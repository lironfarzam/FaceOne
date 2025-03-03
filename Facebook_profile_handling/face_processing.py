"""
Face Processing Module
======================

This module provides functionality for detecting, analyzing, and clustering faces 
in a collection of images. It's designed to identify the most frequently occurring 
person across multiple photos.

Key features:
- Multi-backend face detection for improved accuracy
- Face quality assessment to filter out low-quality detections
- Multiple embedding models for face representation
- DBSCAN clustering to group similar faces
- Advanced cluster merging algorithms to consolidate identities
- Identity verification to ensure cluster consistency
- Visualization tools for debugging and analysis
- Face frame extraction for the most frequent person

The module is built on top of DeepFace and offers a comprehensive pipeline for 
face processing tasks, with configurable parameters at each stage.

Usage:
    from Facebook_profile_handling.face_processing import process_images
    
    most_frequent_person_folder = process_images(
        images_folder="./photos",
        output_folder="./results",
        face_confidence=0.4,
        enhanced_merging=True
    )

Authors: Liron Farzam
"""

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
from scipy.spatial.distance import pdist, squareform, cosine, euclidean
from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing import Pool, cpu_count
from functools import partial

#################################################################
# CONSTANTS AND CONFIGURATION
#################################################################


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
    "Facenet512": 0.3,  # More permissive for merging with FaceNet512
    "VGG-Face": 0.4,
    "Facenet": 0.3,
    "OpenFace": 0.2,
    "DeepFace": 0.3,
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

    This function evaluates face quality using several metrics:
    1. Size - Larger faces typically have more detail
    2. Sharpness - Blurry faces are less useful for recognition
    3. Symmetry - As a proxy for how frontal the face is
    4. Lighting uniformity - Even lighting improves recognition accuracy

    Each metric is weighted differently in the final score calculation.

    Args:
        face_img (numpy.ndarray): The face image to assess
        min_size (int): Minimum size (in pixels) for a high-quality face

    Returns:
        float: Quality score between 0.0 (lowest quality) and 1.0 (highest quality)
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


def process_single_photo(args):
    """Process a single photo for face detection and embedding generation"""
    (
        img_file,
        img_folder,
        face_confidence,
        face_size,
        face_aspect_ratio,
        backends,
        models,
        face_quality_threshold,
    ) = args

    img_path = os.path.join(img_folder, img_file)
    results = {
        "faces": [],
        "embeddings": [],
        "sources": [],
        "locations": [],
        "filenames": [],
        "used_models": [],  # Add this to track which model was used
    }

    try:
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            return results

        # Enhanced face detection strategy:
        face_objs = []

        # Try each backend with enhanced image
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
            except Exception:
                continue

        # Process each detected face
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
                    FACE_ASPECT_RATIO_RANGE[0] <= face_ar <= FACE_ASPECT_RATIO_RANGE[1]
                ):
                    continue

                # Ensure face is in uint8 format
                face = ensure_valid_image(face)

                # Check face quality
                quality_score = assess_face_quality(face, min_size=face_size)
                if quality_score < face_quality_threshold:
                    continue

                # Resize face to standard size
                face_resized = cv2.resize(face, STANDARD_FACE_SIZE)

                # Generate face embedding
                temp_face_path = f"temp_face_{os.getpid()}_{i}.jpg"
                cv2.imwrite(temp_face_path, face_resized)

                # Try each model until one works
                embedding = None
                used_model = None  # Track which model succeeded
                for model in models:
                    try:
                        embedding_obj = DeepFace.represent(
                            img_path=temp_face_path,
                            model_name=model,
                            enforce_detection=False,
                        )
                        if embedding_obj and len(embedding_obj) > 0:
                            embedding = embedding_obj[0]["embedding"]
                            used_model = model  # Store the successful model
                            break
                    except Exception:
                        continue

                # Clean up temp file
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)

                # Skip if no embedding could be generated
                if embedding is None:
                    continue

                # Store results
                results["faces"].append(face_resized)
                results["embeddings"].append(embedding)
                results["sources"].append(img_path)
                results["locations"].append(facial_area)
                results["filenames"].append(img_file)
                results["used_models"].append(used_model)  # Store the model used

            except Exception as e:
                print(f"Error processing face {i} in {img_file}: {e}")
                continue

    except Exception as e:
        print(f"Error processing image {img_file}: {e}")

    return results


def process_batch(args):
    """Process a batch of images in parallel"""
    (
        image_batch,
        images_folder,
        face_confidence,
        face_size,
        face_aspect_ratio,
        backends,
        models,
        face_quality_threshold,
    ) = args

    batch_results = {
        "faces": [],
        "embeddings": [],
        "sources": [],
        "locations": [],
        "filenames": [],
        "used_models": [],  # Add this field
    }

    for img_file in image_batch:
        result = process_single_photo(
            (
                img_file,
                images_folder,
                face_confidence,
                face_size,
                face_aspect_ratio,
                backends,
                models,
                face_quality_threshold,
            )
        )

        # Combine results from this image
        batch_results["faces"].extend(result["faces"])
        batch_results["embeddings"].extend(result["embeddings"])
        batch_results["sources"].extend(result["sources"])
        batch_results["locations"].extend(result["locations"])
        batch_results["filenames"].extend(result["filenames"])
        batch_results["used_models"].extend(result["used_models"])  # Add this line

    return batch_results


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
    extract_frames=False,  # Legacy parameter
    save_best_crops=True,
    max_best_crops=10,
):
    """Process images in parallel batches"""
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
            used_models = loaded_data.get("used_models", [])

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

        # Calculate optimal batch size based on CPU count
        num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
        batch_size = max(
            1, len(image_files) // (num_processes * 4)
        )  # Smaller batches for better load balancing

        # Split images into batches
        batches = [
            image_files[i : i + batch_size]
            for i in range(0, len(image_files), batch_size)
        ]

        # Prepare arguments for parallel processing
        process_args = [
            (
                batch,
                images_folder,
                face_confidence,
                face_size,
                face_aspect_ratio,
                backends,
                models,
                face_quality_threshold,
            )
            for batch in batches
        ]

        print(f"Processing {len(batches)} batches using {num_processes} processes")

        # Process batches in parallel
        with Pool(num_processes) as pool:
            batch_results = list(
                tqdm(
                    pool.imap(process_batch, process_args),
                    total=len(batches),
                    desc="Processing image batches",
                )
            )

        # Combine all results
        all_faces = []
        all_embeddings = []
        face_sources = []
        face_locations = []
        source_filenames = []
        used_models = []  # Add this list

        for result in batch_results:
            all_faces.extend(result["faces"])
            all_embeddings.extend(result["embeddings"])
            face_sources.extend(result["sources"])
            face_locations.extend(result["locations"])
            source_filenames.extend(result["filenames"])
            used_models.extend(result["used_models"])  # Add this line

        print(f"Detected {len(all_faces)} faces in total")

        # Save embeddings if requested
        if all_embeddings and save_embeddings:
            with open(embeddings_path, "wb") as f:
                pickle.dump(
                    {
                        "faces": np.array(all_faces),
                        "embeddings": np.array(all_embeddings),
                        "sources": face_sources,
                        "locations": face_locations,
                        "filenames": source_filenames,
                        "used_models": used_models,  # Add this line
                    },
                    f,
                )
            print(f"Saved {len(all_faces)} face embeddings to {embeddings_path}")

    # Clustering and identification
    if len(all_faces) == 0:
        print("No valid faces found")
        return None

    # Use the most common model for clustering
    model_counts = Counter(used_models)
    used_model = model_counts.most_common(1)[0][0]
    print(f"Using {used_model} for clustering as it was the most commonly used model")

    # Convert to numpy arrays for clustering
    valid_faces = np.array(all_faces)
    embeddings_array = np.array(all_embeddings)
    valid_face_sources = np.array(face_sources)

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

    # Save best face crops if requested
    if save_best_crops:
        best_crops_folder = os.path.join(output_folder, "best_face_crops")
        os.makedirs(best_crops_folder, exist_ok=True)

        crop_count = save_best_face_crops(
            most_frequent_folder,
            best_crops_folder,
            max_images=max_best_crops,
            padding_factor=0.5,  # More padding for better context
            min_confidence=0.85,
            model=used_model,
            detection_backend=backends[0],
            prefer_profile=True,  # Prefer profile views
            enhance_quality=True,
            crop_size=(800, 800),  # High-quality crops
        )

        print(f"Saved {crop_count} best face crops to {best_crops_folder}")

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
        metric (str): Distance metric to use (cosine, euclidean, euclidean_l2, l1)

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
    elif metric == "l1":
        # Manhattan/L1 distance (sum of absolute differences)
        distance = np.sum(np.abs(embedding1 - embedding2)) / len(
            embedding1
        )  # Normalize by dimension
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

    This algorithm uses a two-phase approach:
    1. First phase: Strict merging with a lower threshold to merge highly similar clusters
    2. Second phase: More permissive merging to catch additional matches

    The algorithm also:
    - Uses robust center calculation to handle outliers
    - Performs face verification for borderline cases
    - Validates merged clusters to ensure consistency
    - Prioritizes larger clusters during merging

    This approach provides better results than the basic merging algorithm,
    especially for datasets with variations in lighting, pose, and expression.

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


def highlight_main_person_faces(
    images_folder,
    output_folder,
    frame_color=(0, 255, 0),
    frame_thickness=3,
    model="Facenet512",
    detection_backend="retinaface",
    min_confidence=0.8,  # Slightly lower threshold for better recall
):
    """
    Highlight the main person's face in each image with a colored frame.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files in the folder
    image_files = [
        f
        for f in os.listdir(images_folder)
        if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
    ]

    if len(image_files) == 0:
        print(f"No images found in {images_folder}")
        return 0

    # Use more images for the reference
    num_reference_images = min(10, len(image_files))
    print(
        f"Using {num_reference_images} images to create reference embedding for highlighting"
    )

    # Create reference embeddings
    reference_embeddings = []

    for img_file in image_files[:num_reference_images]:
        img_path = os.path.join(images_folder, img_file)
        try:
            faces = safe_face_detection(
                img_path,
                detector_backend=detection_backend,
                enforce_detection=False,
            )

            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x["confidence"], reverse=True)
                main_face = faces[0]

                if main_face["confidence"] >= min_confidence:
                    temp_face_path = os.path.join(
                        output_folder, f"temp_ref_face_{img_file}"
                    )
                    face_img = ensure_valid_image(main_face["face"])
                    cv2.imwrite(temp_face_path, face_img)

                    embedding = safe_represent(
                        temp_face_path,
                        model_name=model,
                        enforce_detection=False,
                    )

                    if embedding and len(embedding) > 0:
                        reference_embeddings.append(embedding[0]["embedding"])
                        print(f"Added reference embedding from {img_file}")

                    if os.path.exists(temp_face_path):
                        os.remove(temp_face_path)
        except Exception as e:
            print(f"Error processing reference image {img_file}: {e}")
            continue

    if len(reference_embeddings) == 0:
        print("Could not create reference embeddings for the main person")
        return 0

    print(f"Created {len(reference_embeddings)} reference embeddings for highlighting")

    # Create average reference embedding
    reference_embedding = np.mean(reference_embeddings, axis=0)

    # Process all images
    processed_count = 0

    # Adjust the verification threshold to be slightly more lenient
    model_threshold = MODEL_SPECIFIC_THRESHOLDS.get(model, 0.4)
    verification_threshold = model_threshold * 1.2  # Increase threshold by 20%

    print(f"Using verification threshold of {verification_threshold} for model {model}")

    for img_file in tqdm(image_files, desc="Highlighting faces"):
        img_path = os.path.join(images_folder, img_file)

        try:
            # Read the image safely
            img = safe_imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            # Make a copy to draw on
            img_with_frame = img.copy()

            # Detect faces using our safe function
            faces = safe_face_detection(
                img_path,
                detector_backend=detection_backend,
                enforce_detection=False,
            )

            if not faces:
                print(f"No faces detected in {img_file}")
                continue

            main_person_found = False

            # Process each detected face
            for face_obj in faces:
                if face_obj["confidence"] < min_confidence:
                    continue

                # Get face area
                facial_area = face_obj["facial_area"]
                x, y = facial_area["x"], facial_area["y"]
                w, h = facial_area["w"], facial_area["h"]

                # Save face temporarily to get embedding
                temp_face_path = os.path.join(
                    output_folder, f"temp_face_{processed_count}.jpg"
                )
                face_img = ensure_valid_image(face_obj["face"])
                cv2.imwrite(temp_face_path, face_img)

                # Get face embedding
                embedding = safe_represent(
                    temp_face_path,
                    model_name=model,
                    enforce_detection=False,
                )

                # Clean up temp file
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)

                if embedding and len(embedding) > 0:
                    face_embedding = embedding[0]["embedding"]

                    # Compare with reference directly using our adjusted threshold
                    distance = 0
                    if model in ["VGG-Face", "Facenet", "Facenet512", "DeepID"]:
                        distance = cosine(reference_embedding, face_embedding)
                    else:  # Euclidean distance models
                        distance = euclidean(reference_embedding, face_embedding) / 100

                    # If this is the main person, draw a frame
                    if distance <= verification_threshold:
                        cv2.rectangle(
                            img_with_frame,
                            (x, y),
                            (x + w, y + h),
                            frame_color,
                            frame_thickness,
                        )
                        main_person_found = True
                        print(
                            f"Found main person in {img_file} with distance {distance}"
                        )

            # Save the image with frames
            if main_person_found:
                output_path = os.path.join(output_folder, f"highlighted_{img_file}")
                cv2.imwrite(output_path, img_with_frame)
                processed_count += 1

        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            continue

    print(f"Highlighted main person in {processed_count} images")
    return processed_count


def save_best_face_crops(
    images_folder,
    output_folder,
    max_images=10,
    padding_factor=0.5,
    min_confidence=0.85,
    model="Facenet512",
    detection_backend="retinaface",
    prefer_profile=True,
    enhance_quality=True,
    crop_size=(800, 800),
    min_face_size=(30, 30),
):
    """
    Select and save the best face crops from the main person's images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files in the folder
    image_files = [
        f
        for f in os.listdir(images_folder)
        if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
    ]

    if len(image_files) == 0:
        print(f"No images found in {images_folder}")
        return 0

    # Use more images for the reference embeddings to improve accuracy
    num_reference_images = min(10, len(image_files))
    print(f"Using {num_reference_images} images to create reference embedding")

    # First create a reference embedding for the main person
    reference_embeddings = []

    # Use several images to create a robust reference
    for img_file in image_files[:num_reference_images]:
        img_path = os.path.join(images_folder, img_file)
        try:
            # Use our safe face detection function
            faces = safe_face_detection(
                img_path,
                detector_backend=detection_backend,
                enforce_detection=False,
            )

            if len(faces) > 0:
                # Sort by confidence
                faces = sorted(faces, key=lambda x: x["confidence"], reverse=True)
                main_face = faces[0]

                if main_face["confidence"] >= min_confidence:
                    temp_face_path = os.path.join(
                        output_folder, f"temp_ref_face_{img_file}"
                    )
                    face_img = ensure_valid_image(main_face["face"])
                    cv2.imwrite(temp_face_path, face_img)

                    embedding = safe_represent(
                        temp_face_path,
                        model_name=model,
                        enforce_detection=False,
                    )

                    if embedding and len(embedding) > 0:
                        reference_embeddings.append(embedding[0]["embedding"])
                        print(f"Added reference embedding from {img_file}")
                    else:
                        print(f"Failed to get embedding for {img_file}")

                    if os.path.exists(temp_face_path):
                        os.remove(temp_face_path)
                else:
                    print(
                        f"Low confidence detection in {img_file}: {main_face['confidence']}"
                    )
            else:
                print(f"No faces detected in {img_file}")
        except Exception as e:
            print(f"Error processing reference image {img_file}: {e}")
            continue

    if len(reference_embeddings) == 0:
        print("Could not create reference embeddings for the main person")
        return 0

    print(f"Created {len(reference_embeddings)} reference embeddings")

    # Create average reference embedding
    reference_embedding = np.mean(reference_embeddings, axis=0)

    # Analyze all faces in images
    face_candidates = []

    for img_file in tqdm(image_files, desc="Analyzing faces"):
        img_path = os.path.join(images_folder, img_file)

        try:
            # Use our safe face detection
            faces = safe_face_detection(
                img_path,
                detector_backend=detection_backend,
                enforce_detection=False,
            )

            # Count valid faces (confidence > threshold)
            valid_faces_count = sum(
                1 for face in faces if face["confidence"] >= min_confidence
            )

            # Bonus for single-person images
            single_person_bonus = 1.5 if valid_faces_count == 1 else 1.0

            # Process each detected face
            for face_idx, face_obj in enumerate(faces):
                if face_obj["confidence"] < min_confidence:
                    continue

                # Get face area
                facial_area = face_obj["facial_area"]
                x, y = facial_area["x"], facial_area["y"]
                w, h = facial_area["w"], facial_area["h"]

                # Skip very small faces
                if w < min_face_size[0] or h < min_face_size[1]:
                    continue

                # Save face temporarily in proper format
                temp_face_path = os.path.join(
                    output_folder, f"temp_face_{face_idx}.jpg"
                )
                face_img = ensure_valid_image(face_obj["face"])
                cv2.imwrite(temp_face_path, face_img)

                # Get face embedding
                embedding = None
                try:
                    embedding_result = safe_represent(
                        temp_face_path,
                        model_name=model,
                        enforce_detection=False,
                    )
                    if embedding_result and len(embedding_result) > 0:
                        embedding = embedding_result[0]["embedding"]
                except Exception:
                    pass

                # Clean up temp file
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)

                # Skip if no embedding could be generated
                if embedding is None:
                    continue

                # Verify this is the main person
                verification = compare_face_embeddings(
                    reference_embedding, embedding, model_name=model
                )

                if not verification["verified"]:
                    continue

                # Calculate quality score
                quality_score = assess_face_quality(face_img)

                # Check for sunglasses or face masks
                try:
                    # Make sure face image is in proper format
                    face_img = ensure_valid_image(face_obj["face"])

                    # Convert to grayscale safely
                    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                    # Eye region detection
                    h, w = face_img.shape[:2]
                    eye_region = face_gray[int(h * 0.2) : int(h * 0.5), :]

                    # Check brightness variance in eye region
                    eye_variance = np.var(eye_region)

                    # If variance is too low, eyes might be covered
                    if eye_variance < 300:  # Threshold determined empirically
                        continue

                    # Mouth region
                    mouth_region = face_gray[int(h * 0.6) : int(h * 0.9), :]
                    mouth_variance = np.var(mouth_region)

                    # If variance is too low, mouth might be covered
                    if mouth_variance < 200:  # Threshold determined empirically
                        continue
                except Exception:
                    # If we can't check for glasses/masks, we'll proceed anyway
                    pass

                # Calculate profile score
                profile_score = 0.5  # Default neutral score

                # Use face symmetry as a proxy for profile detection
                try:
                    face_img = ensure_valid_image(face_obj["face"])
                    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    flipped = cv2.flip(gray, 1)
                    similarity = cv2.matchTemplate(gray, flipped, cv2.TM_CCOEFF_NORMED)[
                        0
                    ][0]
                    # Convert to 0-1 scale where 0 means symmetric (frontal) and 1 means asymmetric (profile)
                    profile_score = 1.0 - ((similarity + 1) / 2)
                except Exception:
                    pass

                # Calculate combined score with single person bonus
                combined_score = quality_score * 0.5
                if prefer_profile:
                    # Reward profile views
                    combined_score += profile_score * 0.3
                else:
                    # Reward frontal views
                    combined_score += (1.0 - profile_score) * 0.3

                # Apply single person bonus
                combined_score *= single_person_bonus

                # Store candidate
                face_candidates.append(
                    {
                        "img_path": img_path,
                        "facial_area": facial_area,
                        "combined_score": combined_score,
                        "quality_score": quality_score,
                        "profile_score": profile_score,
                        "confidence": face_obj["confidence"],
                        "img_file": img_file,
                        "single_person": valid_faces_count == 1,
                    }
                )

        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            continue

    # If no candidates were found, return
    if not face_candidates:
        print("No suitable face candidates found")
        return 0

    # Sort candidates by score (higher is better)
    face_candidates.sort(key=lambda x: x["combined_score"], reverse=True)

    # Select top N faces with diversity constraint (different source images)
    selected_files = set()
    selected_candidates = []

    # First, prioritize single-person images
    single_person_candidates = [
        c for c in face_candidates if c.get("single_person", False)
    ]

    for candidate in single_person_candidates:
        if candidate["img_file"] in selected_files:
            continue

        selected_candidates.append(candidate)
        selected_files.add(candidate["img_file"])

        if len(selected_candidates) >= max_images:
            break

    # Then add other candidates if needed
    if len(selected_candidates) < max_images:
        remaining_candidates = [
            c for c in face_candidates if c["img_file"] not in selected_files
        ]

        for candidate in remaining_candidates:
            selected_candidates.append(candidate)
            selected_files.add(candidate["img_file"])

            if len(selected_candidates) >= max_images:
                break

    # Crop and save selected faces
    saved_count = 0

    for i, candidate in enumerate(selected_candidates):
        try:
            img = safe_imread(candidate["img_path"])
            if img is None:
                continue

            # Get face area with padding
            facial_area = candidate["facial_area"]
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
            face_crop = img[y : y + h, x : x + w]
            face_crop = ensure_valid_image(face_crop)

            # Enhance quality if requested
            if enhance_quality:
                try:
                    # Color correction
                    face_crop_yuv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YUV)
                    face_crop_yuv[:, :, 0] = cv2.equalizeHist(face_crop_yuv[:, :, 0])
                    face_crop = cv2.cvtColor(face_crop_yuv, cv2.COLOR_YUV2BGR)

                    # Slight sharpening - this can produce floating point results
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    face_crop = cv2.filter2D(face_crop, -1, kernel)

                    # Ensure 8-bit format after all processing
                    face_crop = ensure_valid_image(face_crop)
                except Exception as e:
                    print(f"Warning: Could not enhance image quality: {e}")

            # Resize face
            face_resized = cv2.resize(face_crop, crop_size)

            # Save crop
            crop_type = "profile" if candidate["profile_score"] > 0.6 else "frontal"
            persons = "single" if candidate.get("single_person", False) else "multi"
            quality_text = f"{int(candidate['quality_score']*100)}"
            crop_filename = f"best_face_{i+1}_{crop_type}_{persons}_q{quality_text}.jpg"

            cv2.imwrite(
                os.path.join(output_folder, crop_filename),
                face_resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],  # High quality JPEG
            )

            saved_count += 1

        except Exception as e:
            print(f"Error saving crop {i}: {e}")
            continue

    print(f"Saved {saved_count} best face crops to {output_folder}")
    return saved_count


# At the top of the file, add these functions
def safe_imread(img_path):
    """Safely read an image and ensure it's in proper 8-bit format."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        # Make sure image is 8-bit (uint8)
        if img.dtype != np.uint8:
            if img.dtype == np.float64 or img.dtype == np.float32:
                img = (
                    np.clip(img, 0, 1.0) * 255
                    if img.max() <= 1.0
                    else np.clip(img, 0, 255)
                )
            img = img.astype(np.uint8)

        return img
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        return None


def ensure_valid_image(img):
    """
    Ensure an image is in valid 8-bit format for OpenCV operations.

    Args:
        img (numpy.ndarray): Input image

    Returns:
        numpy.ndarray: Image in 8-bit format or None if invalid
    """
    if img is None:
        return None

    # Make sure image is 8-bit (uint8)
    if img.dtype != np.uint8:
        # Convert to 8-bit unsigned integer with proper scaling
        if img.dtype == np.float64 or img.dtype == np.float32:
            img = (
                np.clip(img, 0, 1.0) * 255 if img.max() <= 1.0 else np.clip(img, 0, 255)
            )
        img = img.astype(np.uint8)
    return img


def safe_face_detection(
    img_path, detector_backend="retinaface", enforce_detection=False
):
    """
    Perform face detection with proper error handling and image format validation.

    Args:
        img_path (str): Path to the image
        detector_backend (str): Face detection backend
        enforce_detection (bool): Whether to enforce detection

    Returns:
        list: List of detected faces or empty list if error
    """
    try:
        # First read and ensure proper format
        img = safe_imread(img_path)
        if img is None:
            return []

        # Save a temp copy in proper format
        temp_dir = os.path.dirname(img_path)
        temp_path = os.path.join(temp_dir, "temp_safe_detect.jpg")
        cv2.imwrite(temp_path, img)

        # Perform detection
        try:
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=True,
            )

            # Process each face to ensure proper format
            for face in faces:
                if "face" in face and face["face"] is not None:
                    face_img = face["face"]
                    if face_img.dtype != np.uint8:
                        if face_img.dtype == np.float64 or face_img.dtype == np.float32:
                            face_img = (
                                np.clip(face_img, 0, 1.0) * 255
                                if face_img.max() <= 1.0
                                else np.clip(face_img, 0, 255)
                            )
                        face["face"] = face_img.astype(np.uint8)
        except Exception as e:
            print(f"Face detection error: {e}")
            faces = []

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return faces
    except Exception as e:
        print(f"Error in face detection pipeline: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return []


def safe_represent(img_path, model_name="Facenet512", enforce_detection=False):
    """
    Get face embedding with proper error handling and image format validation.

    Args:
        img_path (str): Path to the image
        model_name (str): Name of the embedding model
        enforce_detection (bool): Whether to enforce detection

    Returns:
        list: List of embedding dictionaries or empty list if error
    """
    try:
        # First read and ensure proper format
        img = safe_imread(img_path)
        if img is None:
            return []

        # Save a temp copy in proper format
        temp_dir = os.path.dirname(img_path)
        temp_path = os.path.join(temp_dir, "temp_safe_represent.jpg")
        cv2.imwrite(temp_path, img)

        # Get embedding
        try:
            embedding = DeepFace.represent(
                img_path=temp_path,
                model_name=model_name,
                enforce_detection=enforce_detection,
            )
        except Exception as e:
            print(f"Face embedding error: {e}")
            embedding = []

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return embedding
    except Exception as e:
        print(f"Error in embedding pipeline: {e}")
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process images to find faces and identify the most frequent person",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python face_processing.py --input ./my_photos --output ./results
  
  # Use improved merging with high-quality face detection
  python face_processing.py --input ./photos --output ./results --face-quality 0.6 --improved-merging
  
  # Load previously saved embeddings to save processing time
  python face_processing.py --input ./photos --output ./results --load-embeddings
  
  # Highlight faces in images and apply a strict clustering threshold
  python face_processing.py --input ./photos --output ./results --clustering 0.35 --highlight-faces
  
Workflow:
  1. Images are processed to detect faces
  2. Face embeddings are generated using the specified model
  3. Faces are clustered to identify similar identities
  4. Similar clusters are merged to consolidate identities
  5. The most frequent person is identified
  6. Results are saved to the output directory
        """,
    )

    # Input/Output Arguments
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument(
        "--input",
        "-i",
        default="./downloaded_photos",
        help="Input folder containing images to process (default: ./downloaded_photos)",
    )
    io_group.add_argument(
        "--output",
        "-o",
        default="faces_output",
        help="Output folder for all results including detected faces, clusters, and the most frequent person (default: faces_output)",
    )

    # Face Detection Arguments
    detection_group = parser.add_argument_group("Face Detection Options")
    detection_group.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.3,
        help="Face detection confidence threshold (0.0-1.0). Lower values detect more faces but may include false positives (default: 0.3)",
    )
    detection_group.add_argument(
        "--min-size",
        "-s",
        type=int,
        default=20,
        help="Minimum face size in pixels. Smaller faces will be ignored (default: 20)",
    )
    detection_group.add_argument(
        "--aspect-ratio",
        "-a",
        type=float,
        default=2.0,
        help="Maximum face aspect ratio (width/height). Helps filter out non-face detections (default: 2.0)",
    )
    detection_group.add_argument(
        "--face-quality",
        type=float,
        default=0.4,
        help="Minimum face quality threshold (0.0-1.0). Higher values keep only better quality face images (default: 0.4)",
    )

    # Clustering Arguments
    clustering_group = parser.add_argument_group("Clustering Options")
    clustering_group.add_argument(
        "--clustering",
        "-t",
        type=float,
        default=0.4,
        help="Clustering distance threshold (0.0-1.0). Lower values create more clusters with stricter matching (default: 0.4)",
    )
    clustering_group.add_argument(
        "--merge-threshold",
        type=float,
        default=0.5,
        help="Threshold for merging similar clusters (0.0-1.0). Higher values result in more aggressive merging (default: 0.5)",
    )
    clustering_group.add_argument(
        "--improved-merging",
        action="store_true",
        default=True,
        help="Use improved cluster merging algorithm with two-phase approach (default: enabled)",
    )
    clustering_group.add_argument(
        "--basic-merging",
        action="store_true",
        help="Use basic cluster merging instead of improved algorithm (overrides --improved-merging)",
    )
    clustering_group.add_argument(
        "--verify-identity",
        action="store_true",
        help="Perform final identity verification on clusters to ensure consistency (default: disabled)",
    )

    # Model Selection
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model",
        choices=["Facenet512", "VGG-Face", "Facenet", "OpenFace", "DeepFace"],
        default="Facenet512",
        help="Face embedding model to use. Facenet512 provides the best balance of accuracy and speed (default: Facenet512)",
    )

    # Embedding Management
    embed_group = parser.add_argument_group("Embedding Management")
    embed_group.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save face embeddings to disk for later reuse (default: disabled)",
    )
    embed_group.add_argument(
        "--load-embeddings",
        action="store_true",
        help="Try to load previously saved embeddings to skip face detection and embedding generation (default: disabled)",
    )

    # Visualization Options
    viz_group = parser.add_argument_group("Visualization Options")
    viz_group.add_argument(
        "--visualize-clusters",
        action="store_true",
        help="Visualize clusters before merging to help debug clustering results (default: disabled)",
    )

    # Output Processing
    output_group = parser.add_argument_group("Output Processing")
    output_group.add_argument(
        "--extract-frames",
        action="store_true",
        default=False,  # Changed default to False
        help="Extract face frames from the most frequent person's images (legacy option)",
    )
    output_group.add_argument(
        "--highlight-faces",
        action="store_true",
        default=True,  # Default to highlighting faces
        help="Highlight the main person's face in the original images (default: enabled)",
    )
    output_group.add_argument(
        "--no-highlight",
        action="store_true",
        default=False,
        help="Don't highlight faces (overrides --highlight-faces)",
    )
    output_group.add_argument(
        "--save-best-crops",
        action="store_true",
        default=True,  # Default to saving best crops
        help="Save the best face crops of the main person (default: enabled)",
    )
    output_group.add_argument(
        "--no-best-crops",
        action="store_true",
        default=False,
        help="Don't save best face crops (overrides --save-best-crops)",
    )
    output_group.add_argument(
        "--max-crops",
        type=int,
        default=10,
        help="Maximum number of best face crops to save (default: 10)",
    )

    args = parser.parse_args()

    # Parse the options
    extract_frames = args.extract_frames
    highlight_faces = args.highlight_faces and not args.no_highlight
    save_best_crops = args.save_best_crops and not args.no_best_crops

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
        enhanced_merging=use_improved_merging,
        verify_identity=args.verify_identity,
        visualize_before_merge=args.visualize_clusters,
        extract_frames=extract_frames,
        save_best_crops=save_best_crops,
        max_best_crops=args.max_crops,
    )
