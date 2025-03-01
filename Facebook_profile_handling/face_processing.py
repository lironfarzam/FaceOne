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
import logging
from scipy.spatial.distance import pdist, squareform, cosine

#################################################################
# CONSTANTS AND CONFIGURATION
#################################################################

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("face_processing.log"), logging.StreamHandler()],
)

# Face Detection Settings
DETECTION_BACKENDS = ["retinaface", "mtcnn", "opencv", "ssd"]
EMBEDDING_MODELS = ["Facenet512", "VGG-Face", "Facenet", "OpenFace", "DeepFace"]
FACE_CONFIDENCE_THRESHOLD = 0.3
MIN_FACE_SIZE = 35
FACE_ASPECT_RATIO_RANGE = (0.4, 2.0)  # (min, max) aspect ratio
FACE_QUALITY_THRESHOLD = 0.4

# Clustering Settings
CLUSTERING_THRESHOLD = 0.4  # Base threshold for DBSCAN clustering
MIN_CLUSTER_SIZE = 1
MODEL_SPECIFIC_THRESHOLDS = {
    "Facenet512": 0.35,  # Optimized for FaceNet512
    "VGG-Face": 0.45,
    "Facenet": 0.40,
    "OpenFace": 0.35,
    "DeepFace": 0.45,
}

# Merging Settings
MERGE_THRESHOLD = 0.5  # Base threshold for merging similar clusters
MODEL_MERGE_THRESHOLDS = {
    "Facenet512": 0.5,  # More permissive for merging with FaceNet512
    "VGG-Face": 0.6,
    "Facenet": 0.5,
    "OpenFace": 0.4,
    "DeepFace": 0.5,
}
MERGE_VALIDATION_FACTOR = 1.2  # Multiplier for individual face validation threshold
PHASE1_MERGE_FACTOR = 0.9  # Stricter threshold for phase 1 (multiplier)
PHASE2_MERGE_FACTOR = 1.1  # More permissive threshold for phase 2 (multiplier)

# Image Processing Settings
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".gif"]
STANDARD_FACE_SIZE = (224, 224)  # Size to resize faces for consistent comparison

# Visualization Settings
MAX_CLUSTERS_TO_DISPLAY = 20
MAX_FACES_PER_CLUSTER = 6
HIGHLIGHT_COLOR = (0, 1, 0)  # Green for main identity

#################################################################
# HELPER FUNCTIONS AND UTILITIES
#################################################################


def assess_face_quality(face_img, min_size=MIN_FACE_SIZE):
    """
    Assess the quality of a detected face using multiple heuristics.

    Args:
        face_img (numpy.ndarray): The face image to assess
        min_size (int): Minimum size for a high-quality face

    Returns:
        float: Quality score between 0 (low) and 1 (high)
    """
    # Initialize quality score
    quality_score = 0.0

    # Check image dimensions
    h, w = face_img.shape[:2]
    size_score = min(1.0, (h * w) / (min_size * min_size))

    # Check image sharpness
    gray = (
        cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        if len(face_img.shape) > 2
        else face_img
    )
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(
        1.0, laplacian_var / 500
    )  # Normalize (500 is a good sharpness value)

    # Check face symmetry (as a rough proxy for frontal-ness)
    try:
        flipped = cv2.flip(gray, 1)  # Horizontal flip
        similarity = cv2.matchTemplate(gray, flipped, cv2.TM_CCOEFF_NORMED)[0][0]
        symmetry_score = (similarity + 1) / 2  # Convert from [-1,1] to [0,1]
    except Exception:
        symmetry_score = 0.5  # Default if we can't compute

    # Check lighting uniformity
    try:
        # Calculate lighting uniformity by looking at histogram variance
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        hist_var = np.var(hist_norm)
        lighting_score = 1.0 - min(
            1.0, hist_var * 100
        )  # Lower variance = better lighting
    except Exception:
        lighting_score = 0.5

    # Combine scores with different weights
    quality_score = (
        size_score * 0.3
        + sharpness_score * 0.4
        + symmetry_score * 0.2
        + lighting_score * 0.1
    )

    return quality_score


def enhance_image_for_detection(img):
    """
    Enhance an image to improve face detection.

    Args:
        img (numpy.ndarray): Input image

    Returns:
        numpy.ndarray: Enhanced image
    """
    # Normalize image contrast
    try:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # Optional: Denoise the image
        img_enhanced = cv2.fastNlMeansDenoisingColored(img_enhanced, None, 3, 3, 7, 21)

        return img_enhanced
    except Exception:
        return img


def get_optimal_clustering_threshold(model_name, face_count):
    """
    Get the optimal clustering threshold based on the model and dataset size.

    Args:
        model_name (str): Name of the embedding model
        face_count (int): Number of faces in the dataset

    Returns:
        float: Optimal clustering threshold
    """
    base_threshold = MODEL_SPECIFIC_THRESHOLDS.get(model_name, CLUSTERING_THRESHOLD)

    # Adjust threshold based on face count
    if face_count < 10:
        # With few faces, be more strict to avoid false positives
        return base_threshold * 0.9
    elif face_count > 100:
        # With many faces, be more permissive to handle variations
        return base_threshold * 1.1

    return base_threshold


def process_images(
    images_folder,
    output_folder="faces_output",
    face_confidence=FACE_CONFIDENCE_THRESHOLD,
    face_size=MIN_FACE_SIZE,
    face_aspect_ratio=FACE_ASPECT_RATIO_RANGE[1],
    clustering_threshold=CLUSTERING_THRESHOLD,
    min_cluster_size=MIN_CLUSTER_SIZE,
    save_embeddings=True,
    load_embeddings=False,
    backends=DETECTION_BACKENDS,
    models=EMBEDDING_MODELS,
    merge_threshold=MERGE_THRESHOLD,
    face_quality_threshold=FACE_QUALITY_THRESHOLD,
    enhanced_merging=True,
    verify_identity=True,
    visualize_before_merge=True,
    extract_frames=True,
):
    """
    Process images to detect, validate, and cluster faces, identifying the most frequent person.

    Args:
        images_folder (str): Path to folder containing images to process.
        output_folder (str): Path where all outputs will be saved.
        face_confidence (float): Minimum confidence score (0-1) for detected faces.
        face_size (int): Minimum size in pixels for a face to be considered valid.
        face_aspect_ratio (float): Maximum allowed ratio between width and height for a face.
        clustering_threshold (float): Distance threshold for DBSCAN clustering (lower = stricter).
        min_cluster_size (int): Minimum number of faces required to form a cluster.
        save_embeddings (bool): Whether to save face embeddings to disk for later reuse.
        load_embeddings (bool): Whether to try loading previously saved embeddings.
        backends (list): List of detection backends to try, in order of preference.
        models (list): List of face embedding models to try, in order of preference.
        merge_threshold (float): Threshold for merging clusters (higher = more merging).
        face_quality_threshold (float): Minimum quality score for face validation.
        enhanced_merging (bool): Whether to use enhanced cluster merging algorithm.
        verify_identity (bool): Whether to perform final identity verification on clusters.
        visualize_before_merge (bool): Whether to visualize clusters before merging.
        extract_frames (bool): Whether to extract face frames from the most frequent person's images.

    Returns:
        str: Path to the folder containing images of the most frequent person.
    """
    print(f"Processing images from {images_folder}...")

    # Create output directories
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    faces_folder = os.path.join(output_folder, "detected_faces")
    os.makedirs(faces_folder)

    # Path for saved embeddings
    embeddings_path = os.path.join(output_folder, "face_embeddings.pkl")

    # Try to load previous embeddings if requested
    if load_embeddings and os.path.exists(embeddings_path):
        try:
            print("Loading previously saved face embeddings...")
            with open(embeddings_path, "rb") as f:
                loaded_data = pickle.load(f)

            all_faces = loaded_data.get("faces", [])
            all_embeddings = loaded_data.get("embeddings", [])
            face_sources = loaded_data.get("sources", [])
            face_locations = loaded_data.get("locations", [])
            source_filenames = loaded_data.get("filenames", [])

            print(f"Loaded {len(all_faces)} faces with embeddings")

            # Proceed to clustering directly
            if len(all_faces) > 0:
                print(f"Loaded {len(all_faces)} previously processed faces\n")
                goto_clustering = True
        except Exception as e:
            print(f"Error loading embeddings: {e}. Processing images from scratch.")
            goto_clustering = False
    else:
        goto_clustering = False

    # If we didn't successfully load embeddings, process images
    if not goto_clustering:
        # Get all image files
        image_files = [
            f
            for f in os.listdir(images_folder)
            if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
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

        # Process all images with progress bar
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

                # Enhanced face detection strategy:
                face_objs = []

                # 1. Try enhanced image first
                try:
                    img_enhanced = enhance_image_for_detection(img)
                    enhanced_path = os.path.join(output_folder, "temp_enhanced.jpg")
                    cv2.imwrite(enhanced_path, img_enhanced)

                    # Try each backend with enhanced image
                    for backend in backends:
                        try:
                            detected_faces = DeepFace.extract_faces(
                                img_path=enhanced_path,
                                detector_backend=backend,
                                enforce_detection=False,
                                align=True,
                            )
                            if detected_faces and len(detected_faces) > 0:
                                face_objs = detected_faces
                                break
                        except Exception as e:
                            # Just continue to the next backend
                            continue

                    # Clean up temp file
                    if os.path.exists(enhanced_path):
                        os.remove(enhanced_path)
                except Exception as e:
                    # If enhanced detection fails completely, continue to original image
                    print(f"Enhanced detection failed for {img_file}: {e}")

                # 2. If enhanced detection failed, try original image
                if len(face_objs) == 0:
                    for backend in backends:
                        try:
                            detected_faces = DeepFace.extract_faces(
                                img_path=img_path,
                                detector_backend=backend,
                                enforce_detection=False,
                                align=True,
                            )
                            if detected_faces and len(detected_faces) > 0:
                                face_objs = detected_faces
                                break
                        except Exception as e:
                            # Just continue to the next backend
                            continue

                # Process each detected face with improved validation
                for i, face_obj in enumerate(face_objs):
                    try:
                        # Check confidence score
                        if face_obj["confidence"] < face_confidence:
                            continue

                        face = face_obj["face"]
                        facial_area = face_obj["facial_area"]

                        # Get face dimensions and validate size
                        face_width = facial_area["w"]
                        face_height = facial_area["h"]
                        min_dimension = min(face_width, face_height)

                        if min_dimension < face_size:
                            continue

                        # Validate face aspect ratio
                        face_ar = face_width / face_height
                        if not (
                            FACE_ASPECT_RATIO_RANGE[0]
                            <= face_ar
                            <= FACE_ASPECT_RATIO_RANGE[1]
                        ):
                            continue

                        # Ensure face is in uint8 format
                        if face.dtype != np.uint8:
                            if face.dtype == np.float64 or face.dtype == np.float32:
                                face = (face * 255).astype(np.uint8)
                            else:
                                face = face.astype(np.uint8)

                        # Check face quality using our improved function
                        quality_score = assess_face_quality(face, min_size=face_size)
                        if quality_score < face_quality_threshold:
                            continue

                        # Resize face to standard size for consistent shape
                        face_resized = cv2.resize(face, STANDARD_FACE_SIZE)

                        # Generate face embedding
                        embedding = None

                        # Save temp file for embedding generation
                        temp_face_path = os.path.join(
                            output_folder, f"temp_face_{i}.jpg"
                        )
                        cv2.imwrite(temp_face_path, face_resized)

                        # Try each model until one works
                        for model in models:
                            try:
                                embedding_obj = DeepFace.represent(
                                    img_path=temp_face_path,
                                    model_name=model,
                                    enforce_detection=False,
                                )

                                if embedding_obj and len(embedding_obj) > 0:
                                    embedding = embedding_obj[0]["embedding"]
                                    used_model = model
                                    break
                            except Exception:
                                continue

                        # Clean up temp file
                        os.remove(temp_face_path)

                        # Skip if no embedding could be generated
                        if embedding is None:
                            continue

                        # If all checks pass, add the face and its data
                        all_faces.append(face_resized)
                        all_embeddings.append(embedding)
                        face_sources.append(img_path)
                        source_filenames.append(img_file)

                        # Save the detected face
                        face_filename = f"{os.path.splitext(img_file)[0]}_face_{i}.jpg"
                        cv2.imwrite(
                            os.path.join(faces_folder, face_filename), face_resized
                        )

                        face_locations.append(facial_area)
                    except Exception as e:
                        print(f"Error processing face {i} in {img_file}: {e}")
                        continue
            except Exception as e:
                print(f"Error processing image {img_file}: {e}")
                continue

        print(f"Detected {len(all_faces)} faces in total")

        # Convert all_faces to numpy array (now should work because all faces are the same size)
        valid_faces = []
        for face in all_faces:
            # Ensure all faces are properly sized
            if face.shape[:2] != STANDARD_FACE_SIZE:
                face = cv2.resize(face, STANDARD_FACE_SIZE)
            valid_faces.append(face)

        # Only convert embeddings to numpy array
        embeddings_array = np.array(all_embeddings)

        # Save embeddings and metadata
        if all_embeddings and save_embeddings:
            valid_faces = np.array(valid_faces)
            embeddings_array = np.array(embeddings_array)

            # Save embeddings and metadata
            with open(embeddings_path, "wb") as f:
                pickle.dump(
                    {
                        "faces": valid_faces,
                        "embeddings": embeddings_array,
                        "sources": face_sources,
                        "locations": face_locations,
                        "filenames": source_filenames,
                    },
                    f,
                )
            print(f"Saved {len(valid_faces)} face embeddings to {embeddings_path}")

    # Clustering and identification
    if len(all_faces) == 0:
        print("No valid faces found")
        return None

    # Convert to numpy arrays for clustering
    valid_faces = np.array(valid_faces)
    embeddings_array = np.array(embeddings_array)
    valid_face_sources = np.array(face_sources)

    # Perform clustering with optimized parameters
    print("Clustering faces by identity...")
    print(f"Using {len(valid_faces)} faces with valid embeddings for clustering")

    # Get optimal threshold for clustering based on model and dataset size
    optimized_threshold = get_optimal_clustering_threshold(used_model, len(valid_faces))
    print(
        f"Using optimized clustering threshold {optimized_threshold} for {used_model}"
    )

    # Perform DBSCAN clustering
    dbscan = DBSCAN(
        eps=optimized_threshold,
        min_samples=min_cluster_size,
        metric="cosine",
        n_jobs=-1,
    )
    dbscan_labels = dbscan.fit_predict(embeddings_array)

    # Create dictionary of face indices by cluster
    cluster_counts = {}
    for i, label in enumerate(dbscan_labels):
        if label == -1:  # Skip noise
            continue
        if label not in cluster_counts:
            cluster_counts[label] = []
        cluster_counts[label].append(i)

    print(
        "DBSCAN found",
        len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
        "clusters",
    )

    # Visualize clusters before merging if requested
    if visualize_before_merge:
        pre_merge_dir = os.path.join(output_folder, "pre_merge_clusters")
        if not os.path.exists(pre_merge_dir):
            os.makedirs(pre_merge_dir)

        print("Visualizing clusters before merging...")

        # Create dictionary for visualization
        pre_merge_dict = {}
        for label, face_indices in cluster_counts.items():
            pre_merge_dict[label] = face_indices

        try:
            # Visualize all clusters before merging
            visualize_clusters(
                pre_merge_dict,
                valid_faces,
                os.path.join(pre_merge_dir, "all_clusters_pre_merge.jpg"),
            )
            print(
                f"Saved pre-merge visualization to {os.path.join(pre_merge_dir, 'all_clusters_pre_merge.jpg')}"
            )

            # Also save individual large clusters
            for label, face_indices in pre_merge_dict.items():
                if len(face_indices) >= 3:
                    # Create a simple collage of just this cluster
                    create_face_collage(
                        face_indices,
                        valid_faces,
                        os.path.join(pre_merge_dir, f"cluster_{label}.jpg"),
                        max_faces=min(16, len(face_indices)),
                        title=f"Cluster {label}: {len(face_indices)} faces",
                    )
        except Exception as e:
            print(f"Warning: Could not create pre-merge visualization: {e}")

    # Perform cluster merging with improved algorithm
    print("Performing cluster refinement to merge related identities...")

    # Use the appropriate merging function
    if enhanced_merging:
        merged_clusters = improved_merge_similar_clusters(
            cluster_counts,
            embeddings_array,
            merge_threshold=merge_threshold,
            used_model=used_model,
            valid_faces=valid_faces,
        )
    else:
        merged_clusters = merge_similar_clusters(
            cluster_counts,
            embeddings_array,
            merge_threshold=merge_threshold,
            used_model=used_model,
        )

    print(
        f"Reduced from {len(cluster_counts)} to {len(merged_clusters)} clusters after merging"
    )

    # Verify identity if requested
    if verify_identity:
        verified_clusters = {}
        for cluster_id, face_indices in merged_clusters.items():
            is_consistent, core_indices = verify_cluster_identity(
                face_indices, embeddings_array, threshold=optimized_threshold * 1.2
            )

            if is_consistent:
                verified_clusters[cluster_id] = core_indices
                if len(core_indices) < len(face_indices):
                    print(
                        f"Cluster {cluster_id}: Removed {len(face_indices) - len(core_indices)} outlier faces"
                    )
            else:
                # If cluster is inconsistent, try to salvage the core
                if len(core_indices) >= 3:
                    verified_clusters[cluster_id] = core_indices
                    print(
                        f"Cluster {cluster_id}: Identity inconsistent, keeping only {len(core_indices)} core faces"
                    )
                else:
                    # Discard small inconsistent clusters
                    print(
                        f"Cluster {cluster_id}: Discarded due to identity inconsistency"
                    )

        cluster_counts = verified_clusters
    else:
        cluster_counts = merged_clusters

    # Find the most frequent identity
    cluster_sizes = {label: len(indices) for label, indices in cluster_counts.items()}

    if not cluster_sizes:
        print("No valid clusters found")
        return None

    # Get the most frequent person
    most_frequent_label = max(cluster_sizes, key=cluster_sizes.get)
    most_frequent_count = cluster_sizes[most_frequent_label]

    print(f"Most frequent person appears {most_frequent_count} times")

    # Get the face indices for the most frequent person
    most_frequent_face_indices = cluster_counts[most_frequent_label]

    # Find unique original images containing the most frequent person
    most_frequent_sources = [valid_face_sources[i] for i in most_frequent_face_indices]
    unique_sources = set(most_frequent_sources)

    print(f"Found {len(unique_sources)} unique images with the most frequent person")

    # Create a folder for the most frequent person's images
    most_frequent_folder = os.path.join(output_folder, "most_frequent_person")
    os.makedirs(most_frequent_folder, exist_ok=True)

    # Copy unique images to the output folder
    for i, source in enumerate(unique_sources):
        shutil.copy(
            source,
            os.path.join(
                most_frequent_folder, f"image_{i+1}{os.path.splitext(source)[1]}"
            ),
        )

    print(
        f"Saved {len(unique_sources)} images of the most frequent person to {most_frequent_folder}"
    )

    # Extract face frames if requested
    if extract_frames:
        frames_folder = os.path.join(most_frequent_folder, "face_frames")
        os.makedirs(frames_folder, exist_ok=True)

        # Extract frames
        frame_count = extract_face_frames(
            most_frequent_folder,
            frames_folder,
            face_size=(400, 400),  # Larger size for higher quality
            padding_factor=0.3,
            enhance_quality=True,
        )

        print(f"Extracted {frame_count} face frames to {frames_folder}")

    # Create a visualization of all clusters after merging
    try:
        # Convert list of cluster labels to dictionary format expected by visualize_clusters
        cluster_dict = {}
        for cluster_id, face_indices in cluster_counts.items():
            cluster_dict[cluster_id] = face_indices

        visualize_clusters(
            cluster_dict,  # Pass a dictionary instead of a list
            valid_faces,
            os.path.join(output_folder, "all_clusters.jpg"),
            identity_to_highlight=most_frequent_label,
        )
        print(
            f"Saved cluster visualization to {os.path.join(output_folder, 'all_clusters.jpg')}"
        )
    except Exception as e:
        print(f"Warning: Could not create cluster visualization: {e}")

    print("Processing complete!")

    return most_frequent_folder


def visualize_clusters(
    cluster_face_indices, valid_faces, output_path, identity_to_highlight=None
):
    """
    Visualize the clustered faces with color-coded borders by cluster.
    """
    # Convert list format to dictionary if needed
    if isinstance(cluster_face_indices, list):
        # Convert list of labels to a dictionary
        clusters_dict = {}
        for i, label in enumerate(cluster_face_indices):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(i)
        cluster_face_indices = clusters_dict

    # Skip if no clusters
    if not cluster_face_indices:
        print("No clusters to visualize")
        return

    # Calculate total faces and grid size
    total_faces = sum(len(indices) for indices in cluster_face_indices.values())
    grid_size = int(np.ceil(np.sqrt(total_faces)))

    # Create a color map for clusters - use a different method for matplotlib compatibility
    unique_clusters = sorted(cluster_face_indices.keys())
    num_clusters = len(unique_clusters)

    # Use cm.rainbow instead of get_cmap for better compatibility
    import matplotlib.cm as cm

    colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    color_map = {
        cluster_id: colors[i][:3] for i, cluster_id in enumerate(unique_clusters)
    }

    # If there's a specific identity to highlight, give it a special color
    if identity_to_highlight is not None and identity_to_highlight in color_map:
        color_map[identity_to_highlight] = (0, 1, 0)  # Green for the main identity

    # Create the figure
    fig = plt.figure(figsize=(15, 15))

    # Track current position
    pos = 1

    # Add each face to the grid
    for label, face_indices in cluster_face_indices.items():
        for face_idx in face_indices:
            # Create subplot
            ax = fig.add_subplot(grid_size, grid_size, pos)
            ax.imshow(valid_faces[face_idx])
            ax.set_xticks([])
            ax.set_yticks([])

            # Add colored border based on cluster
            rect = plt.Rectangle(
                (0, 0),
                valid_faces[face_idx].shape[1],
                valid_faces[face_idx].shape[0],
                linewidth=4,
                edgecolor=color_map[label],
                facecolor="none",
            )
            ax.add_patch(rect)

            # Add label for the first face in each cluster
            if face_idx == face_indices[0]:
                label_text = f"Cluster {label}"
                if label == identity_to_highlight:
                    label_text += " (Main)"
                ax.set_title(label_text, fontsize=9)

            pos += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

    try:
        plt.close(fig)  # Close the specific figure
    except NameError:
        plt.close()  # Fall back to closing the current figure if fig is not defined


def create_face_collage(face_indices, all_faces, output_path, max_faces=25, title=None):
    """Create a collage of faces"""
    # Limit the number of faces to display
    face_indices = face_indices[: min(len(face_indices), max_faces)]

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(len(face_indices))))

    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)

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
    plt.close(fig)


def extract_face_with_margin(img, face_location, margin_percent=20):
    """
    Extract a face from an image with an additional margin around it.

    Args:
        img (numpy.ndarray): The source image.
        face_location (dict): Dictionary with x, y, w, h coordinates of the face.
        margin_percent (int): Percentage of the face dimensions to add as margin.

    Returns:
        numpy.ndarray: The extracted face image with margin.
    """
    x, y = face_location["x"], face_location["y"]
    w, h = face_location["w"], face_location["h"]

    # Calculate margins
    margin_x = int(w * margin_percent / 100)
    margin_y = int(h * margin_percent / 100)

    # Calculate new coordinates with margin
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img.shape[1], x + w + margin_x)
    y2 = min(img.shape[0], y + h + margin_y)

    # Extract face with margin
    return img[y1:y2, x1:x2]


def analyze_face_attributes(face_path, attributes=None):
    """
    Analyze facial attributes like age, gender, emotion, etc.

    Args:
        face_path (str): Path to the face image.
        attributes (list, optional): List of attributes to analyze.
                                    Default: ["age", "gender", "emotion"]

    Returns:
        dict: Dictionary of detected attributes.
    """
    if attributes is None:
        attributes = ["age", "gender", "emotion"]

    try:
        result = DeepFace.analyze(
            img_path=face_path,
            actions=attributes,
            enforce_detection=False,
            silent=True,
        )

        # Format the result
        if isinstance(result, list):
            return result[0]
        return result
    except Exception as e:
        print(f"Error analyzing face attributes: {e}")
        return {}


def export_cluster_data(clusters, faces, sources, output_folder):
    """
    Export data about each cluster for further analysis.

    Args:
        clusters (dict): Dictionary mapping cluster IDs to lists of face indices.
        faces (list): List of all face images.
        sources (list): List of source image paths for each face.
        output_folder (str): Folder where to save the cluster data.
    """
    report_path = os.path.join(output_folder, "cluster_report.txt")

    with open(report_path, "w") as f:
        f.write("Face Clustering Report\n")
        f.write("=====================\n\n")

        for cluster_id, face_indices in sorted(
            clusters.items(), key=lambda x: len(x[1]), reverse=True
        ):
            f.write(f"Cluster {cluster_id}: {len(face_indices)} faces\n")

            # Get unique source images
            unique_sources = set([sources[i] for i in face_indices])
            f.write(f"  From {len(unique_sources)} unique images\n")

            # List source images
            for i, source in enumerate(unique_sources):
                if i < 10:  # Limit to 10 sources in the report
                    f.write(f"  - {os.path.basename(source)}\n")
                elif i == 10:
                    f.write(f"  - ... and {len(unique_sources) - 10} more\n")

            f.write("\n")

        f.write("\nEnd of Report\n")

    print(f"Cluster report saved to {report_path}")


def compare_face_embeddings(
    embedding1, embedding2, model_name="Facenet512", metric="cosine"
):
    """
    Compare two face embeddings directly.

    Args:
        embedding1 (numpy.ndarray): First face embedding vector
        embedding2 (numpy.ndarray): Second face embedding vector
        model_name (str): Name of the model used to generate embeddings
        metric (str): Distance metric to use (cosine, euclidean, euclidean_l2)

    Returns:
        dict: Dictionary with verification result and distance
    """
    # Convert embeddings to numpy arrays if they aren't already
    if not isinstance(embedding1, np.ndarray):
        embedding1 = np.array(embedding1)
    if not isinstance(embedding2, np.ndarray):
        embedding2 = np.array(embedding2)

    # Ensure embeddings have the same shape
    if embedding1.shape != embedding2.shape:
        raise ValueError(
            f"Embedding shapes don't match: {embedding1.shape} vs {embedding2.shape}"
        )

    # Model-specific thresholds for verification
    thresholds = {
        "VGG-Face": 0.4,
        "Facenet": 0.4,
        "Facenet512": 0.3,  # Lower threshold = stricter matching for FaceNet512
        "OpenFace": 0.1,
        "DeepFace": 0.23,
        "DeepID": 0.015,
        "ArcFace": 0.68,
        "Dlib": 0.7,
    }

    # Get threshold for the model
    threshold = thresholds.get(model_name, 0.4)

    # Calculate distance based on the metric
    if metric == "cosine":
        distance = cosine(embedding1, embedding2)
    elif metric == "euclidean":
        distance = euclidean(embedding1, embedding2) / 100  # Normalize
    elif metric == "euclidean_l2":
        distance = euclidean_distances([embedding1], [embedding2])[0][0]
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    # Verify if the distance is below the threshold
    verified = distance <= threshold

    return {
        "verified": verified,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "metric": metric,
    }


def improved_merge_similar_clusters(
    clusters,
    embeddings,
    merge_threshold=MERGE_THRESHOLD,
    used_model="Facenet512",
    valid_faces=None,
):
    """
    Improved version of cluster merging with advanced similarity metrics.

    Args:
        clusters (dict): Dictionary mapping cluster IDs to lists of face indices
        embeddings (numpy.ndarray): Array of face embeddings
        merge_threshold (float): Threshold for merging (higher = more merging)
        used_model (str): Name of the model used for embeddings
        valid_faces (list): List of face images for visual comparison

    Returns:
        dict: Dictionary of merged clusters
    """
    # Use model-specific thresholds
    if used_model in MODEL_MERGE_THRESHOLDS:
        base_threshold = MODEL_MERGE_THRESHOLDS[used_model]
    else:
        base_threshold = merge_threshold

    print(f"Using base merge threshold {base_threshold} for model {used_model}")

    # Calculate cluster centers with better handling of outliers
    cluster_centers = {}
    cluster_center_faces = {}  # Representative face for each cluster

    for cluster_id, face_indices in clusters.items():
        cluster_embeddings = embeddings[face_indices]

        # For larger clusters, use a robust estimation of the center
        if len(face_indices) >= 3:
            # Compute pairwise distances within cluster
            distances = squareform(pdist(cluster_embeddings, "cosine"))

            # Calculate average distance for each face to all others
            avg_distances = np.mean(distances, axis=1)

            # Find faces with lowest average distance (most central)
            central_idx = np.argsort(avg_distances)[: max(1, len(face_indices) // 2)]

            # Use weighted mean of central faces as cluster center
            weights = 1.0 - avg_distances[central_idx] / (
                np.max(avg_distances[central_idx]) + 1e-10
            )
            weights = weights / np.sum(weights)

            center = np.zeros_like(cluster_embeddings[0])
            for i, idx in enumerate(central_idx):
                center += weights[i] * cluster_embeddings[idx]

            cluster_centers[cluster_id] = center

            # Store most central face as representative
            most_central_idx = np.argmin(avg_distances)
            cluster_center_faces[cluster_id] = face_indices[most_central_idx]
        else:
            # For small clusters, use simple mean
            cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)
            cluster_center_faces[cluster_id] = face_indices[0]

    # Initialize merged clusters and tracking
    merged_clusters = {k: v.copy() for k, v in clusters.items()}
    merged_to = {k: k for k in clusters.keys()}
    merged_away = set()

    # Two-phase merging: first with strict threshold, then more permissive
    for phase in [1, 2]:
        phase_threshold = base_threshold * (
            PHASE1_MERGE_FACTOR if phase == 1 else PHASE2_MERGE_FACTOR
        )
        print(f"Phase {phase} merging with threshold {phase_threshold:.3f}")

        # Sort clusters by size for priority in merging
        clusters_by_size = sorted(
            [(k, v) for k, v in merged_clusters.items() if k not in merged_away],
            key=lambda x: len(x[1]),
            reverse=True,
        )

        anchor_ids = [c[0] for c in clusters_by_size]

        # For each anchor cluster, find candidates to merge
        for anchor_id in anchor_ids:
            if anchor_id in merged_away:
                continue

            # Try each candidate for merging
            for candidate_id in anchor_ids:
                # Skip self, already merged, or already processed clusters
                if (
                    candidate_id == anchor_id
                    or candidate_id in merged_away
                    or merged_to[candidate_id] != candidate_id
                ):
                    continue

                # Calculate similarity between cluster centers
                center_distance = cosine(
                    cluster_centers[anchor_id], cluster_centers[candidate_id]
                )

                # Enhanced merging criteria: Check overlap of source images too
                if valid_faces is not None:
                    # Get representative faces for verification
                    anchor_face = valid_faces[cluster_center_faces[anchor_id]]
                    candidate_face = valid_faces[cluster_center_faces[candidate_id]]

                    # Only use DeepFace verification for borderline cases
                    face_verified = False
                    if (
                        0.8 * phase_threshold
                        <= center_distance
                        <= 1.2 * phase_threshold
                    ):
                        try:
                            # Save representative faces temporarily
                            anchor_path = os.path.join(
                                os.path.dirname(os.path.abspath(__file__)),
                                f"temp_anchor_{anchor_id}.jpg",
                            )
                            candidate_path = os.path.join(
                                os.path.dirname(os.path.abspath(__file__)),
                                f"temp_candidate_{candidate_id}.jpg",
                            )

                            cv2.imwrite(
                                anchor_path,
                                (
                                    cv2.cvtColor(anchor_face, cv2.COLOR_RGB2BGR)
                                    if len(anchor_face.shape) == 3
                                    else anchor_face
                                ),
                            )
                            cv2.imwrite(
                                candidate_path,
                                (
                                    cv2.cvtColor(candidate_face, cv2.COLOR_RGB2BGR)
                                    if len(candidate_face.shape) == 3
                                    else candidate_face
                                ),
                            )

                            # Verify faces
                            result = DeepFace.verify(
                                img1_path=anchor_path,
                                img2_path=candidate_path,
                                model_name=used_model,
                                enforce_detection=False,
                                distance_metric="cosine",
                            )

                            face_verified = result["verified"]

                            # Clean up
                            os.remove(anchor_path)
                            os.remove(candidate_path)
                        except Exception:
                            face_verified = False

                # Decide whether to merge based on metrics
                should_merge = (
                    center_distance <= phase_threshold  # Distance below threshold
                    or face_verified  # Or verified by DeepFace
                )

                if should_merge:
                    print(
                        f"Phase {phase}: Merging cluster {candidate_id} into {anchor_id} "
                        + f"(distance: {center_distance:.3f}, verified: {face_verified})"
                    )

                    # Merge candidate into anchor
                    merged_clusters[anchor_id].extend(merged_clusters[candidate_id])
                    merged_away.add(candidate_id)
                    del merged_clusters[candidate_id]

                    # Update tracking
                    merged_to[candidate_id] = anchor_id

                    # Recompute anchor center
                    anchor_embeddings = embeddings[merged_clusters[anchor_id]]

                    # For large clusters, use robust center calculation
                    if len(merged_clusters[anchor_id]) >= 5:
                        distances = squareform(pdist(anchor_embeddings, "cosine"))
                        avg_distances = np.mean(distances, axis=1)
                        central_idx = np.argsort(avg_distances)[
                            : max(3, len(merged_clusters[anchor_id]) // 3)
                        ]
                        cluster_centers[anchor_id] = np.mean(
                            anchor_embeddings[central_idx], axis=0
                        )
                    else:
                        cluster_centers[anchor_id] = np.mean(anchor_embeddings, axis=0)

    # Final validation of merged clusters
    validated_clusters = {}

    for cluster_id, face_indices in merged_clusters.items():
        # Small clusters don't need validation
        if len(face_indices) <= 3:
            validated_clusters[cluster_id] = face_indices
            continue

        # For larger clusters, validate all faces against the center
        cluster_embeddings = embeddings[face_indices]

        # Recompute cluster center
        distances = squareform(pdist(cluster_embeddings, "cosine"))
        avg_distances = np.mean(distances, axis=1)

        # Find the central face (index within the cluster's faces, not global index)
        central_idx_local = np.argmin(avg_distances)

        # Convert to the global index in the embeddings array
        central_idx_global = face_indices[central_idx_local]

        # Use the global index to get the embedding
        central_embedding = embeddings[central_idx_global]

        # Keep faces that are similar enough to the central face
        valid_indices = []
        for idx in face_indices:
            distance = cosine(central_embedding, embeddings[idx])
            if (
                distance <= base_threshold * MERGE_VALIDATION_FACTOR
            ):  # Slightly more permissive for final validation
                valid_indices.append(idx)

        # If we have enough valid faces, use them; otherwise use all faces
        if len(valid_indices) >= 3:
            validated_clusters[cluster_id] = valid_indices
            if len(valid_indices) < len(face_indices):
                print(
                    f"Cluster {cluster_id}: Removed {len(face_indices) - len(valid_indices)} outlier faces"
                )
        else:
            validated_clusters[cluster_id] = face_indices

    return validated_clusters


def verify_cluster_identity(cluster_indices, embeddings, threshold=0.4):
    """
    Verify that a cluster truly represents a single identity.

    Args:
        cluster_indices (list): Indices of faces in the cluster
        embeddings (numpy.ndarray): Array of face embeddings
        threshold (float): Similarity threshold

    Returns:
        tuple: (is_consistent, core_indices) - Boolean indicating if cluster is consistent,
               and indices of the core faces in the cluster
    """
    if len(cluster_indices) < 3:
        return True, cluster_indices  # Too small to analyze

    # Extract embeddings for this cluster
    cluster_embeddings = embeddings[cluster_indices]

    # Calculate pairwise similarities within cluster
    distances = squareform(pdist(cluster_embeddings, "cosine"))

    # Calculate average distance for each face to all others
    avg_distances = np.mean(distances, axis=1)

    # Find the face with minimum average distance (most central)
    central_idx = np.argmin(avg_distances)

    # Count how many faces are similar to the central face
    similar_count = sum(
        distances[central_idx, j] <= threshold for j in range(len(cluster_indices))
    )
    similar_ratio = similar_count / len(cluster_indices)

    # If at least 80% of faces are similar to the central face, cluster is consistent
    is_consistent = similar_ratio >= 0.8

    # Return the core faces (those similar to central face)
    core_indices = [
        cluster_indices[j]
        for j in range(len(cluster_indices))
        if distances[central_idx, j] <= threshold
    ]

    return is_consistent, core_indices


def visualize_all_clusters(
    clusters, faces, output_path, max_clusters=20, max_faces_per_cluster=4
):
    """
    Create a visualization of all clusters in a single image.

    Args:
        clusters (dict): Dictionary mapping cluster IDs to lists of face indices
        faces (list): List of all face images
        output_path (str): Path where to save the visualization
        max_clusters (int): Maximum number of clusters to visualize
        max_faces_per_cluster (int): Maximum number of faces to show per cluster
    """
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    # Limit the number of clusters to display
    sorted_clusters = sorted_clusters[: min(len(sorted_clusters), max_clusters)]

    # Calculate grid dimensions
    n_clusters = len(sorted_clusters)

    if n_clusters == 0:
        print("No clusters to visualize")
        return

    # Create figure with subplots for each cluster
    fig, axes = plt.subplots(
        n_clusters,
        max_faces_per_cluster,
        figsize=(max_faces_per_cluster * 2, n_clusters * 2),
    )

    # If there's only one cluster, ensure axes is a 2D array
    if n_clusters == 1:
        axes = axes.reshape(1, -1)

    # Add each cluster to the visualization
    for i, (cluster_id, face_indices) in enumerate(sorted_clusters):
        # Get faces for this cluster (limit to max_faces_per_cluster)
        cluster_faces = [faces[idx] for idx in face_indices[:max_faces_per_cluster]]

        # Fill in empty spaces if needed
        while len(cluster_faces) < max_faces_per_cluster:
            cluster_faces.append(np.zeros_like(faces[0]))

        # Add faces to the visualization
        for j, face in enumerate(cluster_faces):
            axes[i, j].imshow(face)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            # Add border around the image
            color = "green" if j < len(face_indices) else "white"
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

        # Add cluster info to the first image
        axes[i, 0].set_title(f"Cluster {cluster_id}\n({len(face_indices)} faces)")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved cluster visualization to {output_path}")


def merge_similar_clusters(
    clusters, embeddings, merge_threshold=0.5, used_model="Facenet512"
):
    """
    Basic version of cluster merging function.

    Args:
        clusters (dict): Dictionary mapping cluster IDs to lists of face indices
        embeddings (numpy.ndarray): Array of face embeddings
        merge_threshold (float): Threshold for merging (higher = more merging)
        used_model (str): Name of the model used for embeddings

    Returns:
        dict: Dictionary of merged clusters
    """
    # Adjust threshold based on model
    model_merge_thresholds = {
        "Facenet512": 0.5,  # More permissive for merging
        "VGG-Face": 0.6,
        "Facenet": 0.5,
        "OpenFace": 0.4,
        "DeepFace": 0.5,
    }

    if used_model in model_merge_thresholds:
        merge_threshold = model_merge_thresholds[used_model]

    print(f"Using merge threshold {merge_threshold} for model {used_model}")

    # Calculate cluster centers (mean embeddings)
    cluster_centers = {}
    for cluster_id, face_indices in clusters.items():
        cluster_embeddings = embeddings[face_indices]
        cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)

    # Initialize merged clusters with original clusters
    merged_clusters = {k: v.copy() for k, v in clusters.items()}
    merged_to = {k: k for k in clusters.keys()}  # Track where each cluster got merged

    # List clusters by size (descending) for priority in merging
    clusters_by_size = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    anchor_clusters = [c[0] for c in clusters_by_size]

    # Track which clusters have been merged
    merged_away = set()

    # For each cluster, check if it should be merged with another
    for i, anchor_id in enumerate(anchor_clusters):
        if anchor_id in merged_away:
            continue

        for j, candidate_id in enumerate(anchor_clusters[i + 1 :], i + 1):
            # Skip self or already merged clusters
            if (
                candidate_id == anchor_id
                or candidate_id in merged_away
                or merged_to[candidate_id] != candidate_id
            ):
                continue

            # Calculate similarity between cluster centers
            distance = cosine(cluster_centers[anchor_id], cluster_centers[candidate_id])

            # If similar enough, merge the candidate into the anchor
            if distance <= merge_threshold:
                print(
                    f"Merging cluster {candidate_id} into {anchor_id} (distance: {distance:.3f})"
                )

                # Add all faces from candidate to anchor
                merged_clusters[anchor_id].extend(merged_clusters[candidate_id])
                merged_away.add(candidate_id)
                del merged_clusters[candidate_id]

                # Update merge tracking
                merged_to[candidate_id] = anchor_id

                # Update the anchor's center after merging
                anchor_embeddings = embeddings[merged_clusters[anchor_id]]
                cluster_centers[anchor_id] = np.mean(anchor_embeddings, axis=0)

    # Perform basic validation
    validated_clusters = {}

    for cluster_id, face_indices in merged_clusters.items():
        # If the cluster is small, just keep it as is
        if len(face_indices) <= 3:
            validated_clusters[cluster_id] = face_indices
            continue

        # For larger clusters, validate by checking face-to-face similarity
        valid_faces = []

        # Use the most central face as reference
        center_embedding = cluster_centers[cluster_id]

        distances_to_center = [
            cosine(center_embedding, embeddings[idx]) for idx in face_indices
        ]

        reference_idx = face_indices[np.argmin(distances_to_center)]
        reference_embedding = embeddings[reference_idx]
        valid_faces.append(reference_idx)

        # Compare all other faces to the reference
        for idx in face_indices:
            if idx == reference_idx:
                continue

            distance = cosine(reference_embedding, embeddings[idx])
            # Use a slightly more permissive threshold for individual face validation
            if distance <= merge_threshold * 1.2:
                valid_faces.append(idx)

        validated_clusters[cluster_id] = valid_faces

    return validated_clusters


def extract_face_frames(
    image_folder,
    output_folder,
    face_size=(224, 224),
    padding_factor=0.3,
    enhance_quality=True,
):
    """
    Extract face frames from all images in a folder and save them as separate files.

    Args:
        image_folder (str): Folder containing images
        output_folder (str): Folder to save extracted face frames
        face_size (tuple): Size to resize extracted faces
        padding_factor (float): Amount of padding around the face (relative to face size)
        enhance_quality (bool): Whether to enhance the quality of extracted faces

    Returns:
        int: Number of frames extracted
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files in the folder
    image_files = [
        f
        for f in os.listdir(image_folder)
        if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
    ]

    frame_count = 0

    # Process each image
    for img_file in tqdm(image_files, desc="Extracting face frames"):
        img_path = os.path.join(image_folder, img_file)

        try:
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            # Detect faces with high confidence threshold
            for backend in DETECTION_BACKENDS:
                try:
                    detected_faces = DeepFace.extract_faces(
                        img_path=img_path,
                        detector_backend=backend,
                        enforce_detection=False,
                        align=True,
                    )
                    if detected_faces and len(detected_faces) > 0:
                        break
                except Exception:
                    continue

            # Process each detected face
            for i, face_obj in enumerate(detected_faces):
                if face_obj["confidence"] < 0.9:  # Only high confidence faces
                    continue

                # Get face and area
                face = face_obj["face"]
                facial_area = face_obj["facial_area"]

                # Get area with padding
                x = max(0, facial_area["x"] - int(facial_area["w"] * padding_factor))
                y = max(0, facial_area["y"] - int(facial_area["h"] * padding_factor))
                w = min(
                    img.shape[1] - x,
                    facial_area["w"] + int(facial_area["w"] * padding_factor * 2),
                )
                h = min(
                    img.shape[0] - y,
                    facial_area["h"] + int(facial_area["h"] * padding_factor * 2),
                )

                # Extract face with padding
                face_padded = img[y : y + h, x : x + w]

                # Enhance quality if requested
                if enhance_quality:
                    # Color correction
                    face_padded_yuv = cv2.cvtColor(face_padded, cv2.COLOR_BGR2YUV)
                    face_padded_yuv[:, :, 0] = cv2.equalizeHist(
                        face_padded_yuv[:, :, 0]
                    )
                    face_padded = cv2.cvtColor(face_padded_yuv, cv2.COLOR_YUV2BGR)

                    # Slight sharpening
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    face_padded = cv2.filter2D(face_padded, -1, kernel)

                # Resize face
                face_resized = cv2.resize(face_padded, face_size)

                # Save frame
                frame_filename = f"{os.path.splitext(img_file)[0]}_frame_{i}.jpg"
                cv2.imwrite(
                    os.path.join(output_folder, frame_filename),
                    face_resized,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],  # High quality JPEG
                )

                frame_count += 1

        except Exception as e:
            print(f"Error processing image for frames: {img_file} - {e}")

    return frame_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process images to find faces and identify the most frequent person"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="./downloaded_photos",
        help="Input folder containing images",
    )
    parser.add_argument(
        "--output", "-o", default="faces_output", help="Output folder for results"
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.3,
        help="Face detection confidence threshold",
    )
    parser.add_argument(
        "--min-size", "-s", type=int, default=20, help="Minimum face size in pixels"
    )
    parser.add_argument(
        "--aspect-ratio",
        "-a",
        type=float,
        default=2.0,
        help="Maximum face aspect ratio",
    )
    parser.add_argument(
        "--clustering",
        "-t",
        type=float,
        default=0.4,
        help="Clustering distance threshold",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save face embeddings for future use",
    )
    parser.add_argument(
        "--load-embeddings",
        action="store_true",
        help="Try to load previously saved embeddings",
    )
    parser.add_argument(
        "--model",
        choices=["Facenet512", "VGG-Face", "Facenet", "OpenFace", "DeepFace"],
        default="Facenet512",
        help="Face embedding model to use",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.5,  # Increased from 0.45 to 0.5 for better merging
        help="Threshold for merging similar clusters (higher = more merging)",
    )
    parser.add_argument(
        "--face-quality",
        type=float,
        default=0.4,  # Lowered from 0.5 to 0.4 for more permissive face acceptance
        help="Minimum face quality threshold (0-1)",
    )
    parser.add_argument(
        "--visualize-clusters",
        action="store_true",
        help="Visualize clusters before merging",
    )
    parser.add_argument(
        "--improved-merging",
        action="store_true",
        default=True,  # Make it true by default
        help="Use improved cluster merging algorithm (default: enabled)",
    )
    parser.add_argument(
        "--basic-merging",
        action="store_true",
        help="Use basic cluster merging instead of improved algorithm",
    )
    parser.add_argument(
        "--verify-identity",
        action="store_true",
        help="Perform final identity verification on clusters",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        default=True,  # Enable by default
        help="Extract face frames from the most frequent person's images",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        default=False,  # Disable by default
        help="Don't extract face frames",
    )

    args = parser.parse_args()

    # Parse the extract frames option
    extract_frames = args.extract_frames and not args.no_frames

    # If basic-merging is specified, it overrides improved-merging
    use_improved_merging = not args.basic_merging  # Simplified logic

    process_images(
        images_folder=args.input,
        output_folder=args.output,
        face_confidence=args.confidence,
        face_size=args.min_size,
        face_aspect_ratio=args.aspect_ratio,
        clustering_threshold=args.clustering,
        save_embeddings=args.save_embeddings,
        load_embeddings=args.load_embeddings,
        models=[args.model] + ["VGG-Face", "Facenet", "OpenFace", "DeepFace"],
        merge_threshold=args.merge_threshold,
        face_quality_threshold=args.face_quality,
        enhanced_merging=use_improved_merging,  # Use improved merging by default
        verify_identity=args.verify_identity,
        visualize_before_merge=args.visualize_clusters,
        extract_frames=args.extract_frames,
    )
