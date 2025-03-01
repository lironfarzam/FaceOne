import os
import shutil
import cv2
import numpy as np
from deepface import DeepFace
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import DBSCAN
from matplotlib.patches import Rectangle


def process_images(
    images_folder,
    output_folder="faces_output",
    face_confidence=0.3,
    face_size=20,
    face_aspect_ratio=2.0,
):
    """
    Process all images in a folder, detect faces, and cluster them by identity.

    Args:
        images_folder (str): The path to the folder containing the images to process.
        output_folder (str): The path to the folder where the output will be saved.
        face_confidence (float): The confidence threshold for a face to be considered valid.
        face_size (int): The minimum size of a face to be considered valid.
        face_aspect_ratio (float): The aspect ratio of a face to be considered valid.
    """
    print(f"Processing images from {images_folder}...")

    # Create output directories
    if os.path.exists(output_folder):
        # Delete folder
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)

    faces_folder = os.path.join(output_folder, "detected_faces")
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)

    # Get all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]  # Added webp
    image_files = [
        f
        for f in os.listdir(images_folder)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]

    print(f"Found {len(image_files)} images to process")

    # Lists to store face data
    all_faces = []
    all_embeddings = []
    face_sources = []
    face_locations = []
    source_filenames = []

    # Keep track of processed files to avoid duplicates
    processed_filenames = set()

    # Extract faces from images
    for img_file in tqdm(image_files, desc="Detecting faces"):
        img_path = os.path.join(images_folder, img_file)

        # Skip if this file has been processed already
        if img_file in processed_filenames:
            continue

        processed_filenames.add(img_file)

        try:
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            # Try multiple detection backends in order of accuracy
            backends = ["retinaface", "mtcnn", "opencv", "ssd"]
            face_objs = []

            # Optional: basic image enhancement before detection
            try:
                # Normalize image contrast
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                # Save enhanced image temporarily
                enhanced_path = os.path.join(output_folder, "temp_enhanced.jpg")
                cv2.imwrite(enhanced_path, img_enhanced)

                # First try with enhanced image
                for backend in backends:
                    try:
                        detected_faces = DeepFace.extract_faces(
                            img_path=enhanced_path,
                            detector_backend=backend,
                            enforce_detection=False,
                        )
                        if detected_faces and len(detected_faces) > 0:
                            face_objs = detected_faces
                            break
                    except Exception:
                        continue

                # Clean up temp file
                if os.path.exists(enhanced_path):
                    os.remove(enhanced_path)
            except Exception:
                pass

            # If enhanced detection failed, try original image
            if len(face_objs) == 0:
                for backend in backends:
                    try:
                        detected_faces = DeepFace.extract_faces(
                            img_path=img_path,
                            detector_backend=backend,
                            enforce_detection=False,
                        )
                        if detected_faces and len(detected_faces) > 0:
                            face_objs = detected_faces
                            break
                    except Exception as e:
                        # print(f"Detection failed with {backend} backend: {e}")
                        continue

            # Process each detected face
            for i, face_obj in enumerate(face_objs):
                # Filter by confidence - more permissive threshold
                if (
                    face_obj["confidence"] > 0.3
                ):  # Even more permissive for Facebook photos
                    face = face_obj["face"]

                    # Skip very small faces (likely false positives)
                    facial_area = face_obj["facial_area"]
                    face_width = facial_area["w"]
                    face_height = facial_area["h"]
                    min_dimension = min(face_width, face_height)

                    # Less strict minimum size
                    if min_dimension < 20:
                        continue

                    # More permissive face validation - wider aspect ratio range
                    face_aspect_ratio = face_width / face_height
                    if not (
                        0.4 <= face_aspect_ratio <= 2.0
                    ):  # Wider range for profile pics
                        continue

                    # Ensure face is in uint8 format (0-255 range)
                    if face.dtype != np.uint8:
                        # Normalize and convert if not in uint8 format
                        if face.dtype == np.float64 or face.dtype == np.float32:
                            face = (face * 255).astype(np.uint8)
                        else:
                            face = face.astype(np.uint8)

                    # Resize face to standard size for better comparison
                    face_resized = cv2.resize(face, (224, 224))

                    # Generate face embedding directly
                    embedding = None
                    valid_face = False
                    temp_face_path = None

                    try:
                        # Save temp file for embedding generation
                        temp_face_path = os.path.join(
                            output_folder, f"temp_face_{i}.jpg"
                        )
                        cv2.imwrite(
                            temp_face_path,
                            (
                                cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
                                if len(face_resized.shape) == 3
                                and face_resized.shape[2] == 3
                                else face_resized
                            ),
                        )

                        # Try multiple models for face verification
                        models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]

                        for model in models:
                            try:
                                # Get embedding
                                embedding_result = DeepFace.represent(
                                    img_path=temp_face_path,
                                    model_name=model,
                                    enforce_detection=False,
                                )

                                if isinstance(embedding_result, list):
                                    embedding = embedding_result[0]["embedding"]
                                else:
                                    embedding = embedding_result["embedding"]

                                valid_face = True
                                break
                            except Exception:
                                continue

                        # If we couldn't get an embedding, try simple validation
                        if not valid_face:
                            validation_methods = ["emotion", "age", "gender"]
                            for method in validation_methods:
                                try:
                                    DeepFace.analyze(
                                        img_path=temp_face_path,
                                        actions=[method],
                                        enforce_detection=False,
                                        silent=True,
                                    )
                                    valid_face = True
                                    break
                                except Exception:
                                    continue
                    except Exception as e:
                        # print(f"Error validating face: {e}")
                        pass
                    finally:
                        # Clean up
                        if temp_face_path and os.path.exists(temp_face_path):
                            os.remove(temp_face_path)

                    # Skip if we couldn't validate this as a face
                    if not valid_face:
                        continue

                    # Save individual face
                    face_filename = f"{os.path.splitext(img_file)[0]}_face_{i}.jpg"
                    face_path = os.path.join(faces_folder, face_filename)

                    try:
                        if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
                            cv2.imwrite(
                                face_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
                            )
                        else:
                            cv2.imwrite(face_path, face_resized)
                    except Exception:
                        # Try alternate saving method if conversion fails
                        cv2.imwrite(face_path, face_resized)

                    # Store face data
                    all_faces.append(face_resized)
                    all_embeddings.append(embedding)
                    face_sources.append(img_path)
                    face_locations.append(face_obj["facial_area"])
                    source_filenames.append(img_file)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    print(f"Detected {len(all_faces)} faces in total")

    if len(all_faces) == 0:
        print("No faces detected. Exiting.")
        return

    # Step 2: Use more advanced clustering for faces
    print("Clustering faces by identity...")

    # Filter out faces without embeddings
    valid_indices = [i for i, emb in enumerate(all_embeddings) if emb is not None]

    if len(valid_indices) < 2:
        print("Not enough valid face embeddings found for clustering.")
        return

    valid_embeddings = [all_embeddings[i] for i in valid_indices]
    valid_face_sources = [face_sources[i] for i in valid_indices]
    valid_faces = [all_faces[i] for i in valid_indices]

    # Convert embeddings to numpy array
    embeddings_array = np.array(valid_embeddings)

    print(f"Using {len(valid_indices)} faces with valid embeddings for clustering")

    # Method 1: DBSCAN Clustering (more robust for varying cluster sizes and shapes)
    # Use a more permissive distance threshold (epsilon)
    dbscan = DBSCAN(eps=0.4, min_samples=1, metric="cosine")
    dbscan_labels = dbscan.fit_predict(embeddings_array)

    # Count unique clusters (excluding noise at -1)
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN found {n_clusters} clusters")

    # Count faces in each cluster
    cluster_counts = {}
    for i, label in enumerate(dbscan_labels):
        if label == -1:  # Skip noise
            continue
        if label not in cluster_counts:
            cluster_counts[label] = []
        cluster_counts[label].append(i)

    # Find most frequent identity
    if not cluster_counts:
        print("No valid clusters found")
        return

    most_frequent_label = max(cluster_counts, key=lambda k: len(cluster_counts[k]))
    most_frequent_indices = cluster_counts[most_frequent_label]
    most_frequent_count = len(most_frequent_indices)

    print(f"Most frequent person appears {most_frequent_count} times")

    # Save original images containing the most frequent person
    most_frequent_folder = os.path.join(output_folder, "most_frequent_person")
    if os.path.exists(most_frequent_folder):
        shutil.rmtree(most_frequent_folder)
    os.makedirs(most_frequent_folder)

    most_frequent_source_images = set(
        [valid_face_sources[i] for i in most_frequent_indices]
    )
    print(
        f"Found {len(most_frequent_source_images)} unique images with the most frequent person"
    )

    # Save images with face boxes highlighted
    for i, source_img in enumerate(most_frequent_source_images):
        try:
            # Get all face locations for this source image
            faces_in_this_image = [
                j for j, src in enumerate(valid_face_sources) if src == source_img
            ]
            relevant_faces = [
                j for j in faces_in_this_image if j in most_frequent_indices
            ]

            if not relevant_faces:
                continue

            # Read original image
            img = cv2.imread(source_img)
            if img is None:
                continue

            # Get original filename
            source_filename = os.path.basename(source_img)

            # Add face boxes for the most frequent person
            for face_idx in relevant_faces:
                face_loc = face_locations[valid_indices[face_idx]]
                x, y = face_loc["x"], face_loc["y"]
                w, h = face_loc["w"], face_loc["h"]

                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save original image to most frequent folder
            cv2.imwrite(
                os.path.join(most_frequent_folder, f"{i}_{source_filename}"), img
            )

        except Exception as e:
            print(f"Error saving {source_img}: {e}")

    print(
        f"Saved {len(most_frequent_source_images)} images of the most frequent person to {most_frequent_folder}"
    )

    # Save a collage of face crops of the most frequent person
    face_collage_path = os.path.join(output_folder, "most_frequent_faces_collage.jpg")
    create_face_collage(
        [valid_indices[i] for i in most_frequent_indices], all_faces, face_collage_path
    )

    # Save a visualization of all clusters for debugging
    visualize_clusters(
        valid_faces, dbscan_labels, os.path.join(output_folder, "all_clusters.jpg")
    )

    print("Processing complete!")
    return most_frequent_folder


def visualize_clusters(faces, labels, output_path):
    """Create a visualization of all face clusters"""
    # Get unique labels
    unique_labels = set(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 0:
        return

    # Create color map
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Group faces by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(faces[i])

    # Sort clusters by size (descending)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    # Create figure
    n_rows = min(8, len(sorted_clusters))
    plt.figure(figsize=(15, 2 * n_rows))

    for i, (label, cluster_faces) in enumerate(sorted_clusters):
        if i >= n_rows:
            break

        # Show up to 8 faces per cluster
        n_faces = min(8, len(cluster_faces))
        for j in range(n_faces):
            plt.subplot(n_rows, 8, i * 8 + j + 1)

            # Get the face
            face = cluster_faces[j]

            # Show face
            plt.imshow(face)
            plt.axis("off")

            # Add colored border for cluster
            if label != -1:  # -1 is noise
                plt.gca().add_patch(
                    Rectangle(
                        (-1, -1),
                        face.shape[1] + 1,
                        face.shape[0] + 1,
                        linewidth=4,
                        edgecolor=color_map[label],
                        facecolor="none",
                    )
                )

            # Add label only to the first face
            if j == 0:
                plt.title(f"Cluster {label}\n({len(cluster_faces)} faces)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_face_collage(face_indices, all_faces, output_path, max_faces=25):
    """Create a collage of faces"""
    # Limit the number of faces to display
    face_indices = face_indices[: min(len(face_indices), max_faces)]

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(len(face_indices))))

    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Handle case when there's only one image
    if grid_size == 1:
        axes.imshow(all_faces[face_indices[0]])
        axes.set_xticks([])
        axes.set_yticks([])
    else:
        # Add each face to the grid
        for i, ax in enumerate(axes.flat):
            if i < len(face_indices):
                ax.imshow(all_faces[face_indices[i]])
            ax.set_xticks([])
            ax.set_yticks([])

    plt.savefig(output_path)
    print(f"Saved face collage to {output_path}")


if __name__ == "__main__":
    # Replace with your folder path
    images_folder = "./downloaded_photos"
    process_images(images_folder)
