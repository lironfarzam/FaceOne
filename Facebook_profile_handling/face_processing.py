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
import mediapipe as mp
import logging

# Configure logging to suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)  # Suppress MediaPipe logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)  # Suppress matplotlib warnings
logging.getLogger("PIL").setLevel(logging.ERROR)  # Suppress PIL warnings

# Disable scientific notation for numpy
np.set_printoptions(suppress=True)

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
MODEL_SPECIFIC_THRESHOLDS = {
    "Facenet512": 0.35,  # Optimized for FaceNet512
    "VGG-Face": 0.45,
    "Facenet": 0.40,
    "OpenFace": 0.35,
    "DeepFace": 0.45,
}

# Merging Settings
MERGE_THRESHOLD = 0.1  # Base threshold for merging similar clusters
MODEL_MERGE_THRESHOLDS = {
    "Facenet512": 0.2,  # More permissive for merging with FaceNet512
    "VGG-Face": 0.2,
    "Facenet": 0.2,
    "OpenFace": 0.2,
    "DeepFace": 0.2,
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
    Returns a quality score between 0.0 (lowest) and 1.0 (highest).

    Quality metrics:
    - Size: Larger faces have more detail (30%)
    - Sharpness: Clear, non-blurry faces (40%)
    - Symmetry: How frontal the face is (20%)
    - Lighting: Even illumination (10%)
    """
    try:
        # Convert to grayscale if needed
        gray = (
            cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            if len(face_img.shape) > 2
            else face_img
        )

        # Size score (30%)
        h, w = face_img.shape[:2]
        size_score = min(1.0, (h * w) / (min_size * min_size))

        # Sharpness score (40%)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500)

        # Symmetry score (20%)
        flipped = cv2.flip(gray, 1)
        similarity = cv2.matchTemplate(gray, flipped, cv2.TM_CCOEFF_NORMED)[0][0]
        symmetry_score = (similarity + 1) / 2

        # Lighting uniformity score (10%)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        hist_var = np.var(hist_norm)
        lighting_score = 1.0 - min(1.0, hist_var * 100)

        # Weighted combination
        quality_score = (
            size_score * 0.3
            + sharpness_score * 0.4
            + symmetry_score * 0.2
            + lighting_score * 0.1
        )

        return max(0.0, min(1.0, quality_score))

    except Exception as e:
        print(f"Error assessing face quality: {e}")
        return 0.0


def safe_face_detection(
    img_path, detector_backend="retinaface", enforce_detection=False
):
    """
    Safely detect faces in an image using specified backend.
    Includes quality assessment and proper error handling.
    """
    try:
        # Read image safely
        img = safe_imread(img_path)
        if img is None:
            return []

        # Detect faces
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=True,
        )

        if not faces:
            return []

        # Add quality scores and filter faces
        valid_faces = []
        for face_obj in faces:
            if not isinstance(face_obj, dict) or "face" not in face_obj:
                continue

            # Assess quality
            quality_score = assess_face_quality(face_obj["face"])
            face_obj["quality_score"] = quality_score

            # Add to valid faces if meets criteria
            if (
                face_obj.get("confidence", 0) >= FACE_CONFIDENCE_THRESHOLD
                and quality_score >= FACE_QUALITY_THRESHOLD
            ):
                valid_faces.append(face_obj)

        return valid_faces

    except Exception as e:
        print(f"Error in face detection for {img_path}: {e}")
        return []


def enhance_image_for_detection(img):
    """
    Enhance image quality for better face detection.
    Applies color correction, contrast enhancement and noise reduction.
    """
    try:
        # Convert to float32 for processing
        img_float = img.astype(np.float32) / 255.0

        # Color correction and contrast enhancement
        lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Enhance L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(np.uint8(l * 255)) / 255.0

        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Convert back to uint8
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        # Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            luminance=3,
            color=3,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        return enhanced

    except Exception as e:
        print(f"Error enhancing image: {e}")
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
    backends=DETECTION_BACKENDS,
    models=EMBEDDING_MODELS,
    merge_threshold=MERGE_THRESHOLD,
    face_quality_threshold=FACE_QUALITY_THRESHOLD,
    enhanced_merging=True,
    verify_identity=True,
    visualize_before_merge=True,
    save_best_crops=True,
    max_best_crops=10,
):
    """
    Process a folder of images to detect, cluster, and analyze faces.

    Args:
        images_folder (str): Path to folder containing input images
        output_folder (str): Path to save output files and visualizations
        face_confidence (float): Minimum confidence threshold for face detection
        face_size (int): Minimum size in pixels for detected faces
        face_aspect_ratio (float): Maximum allowed aspect ratio for faces
        clustering_threshold (float): DBSCAN clustering distance threshold
        backends (list): List of face detection backends to try
        models (list): List of face embedding models to use
        merge_threshold (float): Threshold for merging similar clusters
        face_quality_threshold (float): Minimum quality score for faces
        enhanced_merging (bool): Whether to use enhanced cluster merging
        verify_identity (bool): Whether to verify identity consistency in clusters
        visualize_before_merge (bool): Whether to visualize clusters before merging
        save_best_crops (bool): Whether to save best quality face crops
        max_best_crops (int): Maximum number of best crops to save per identity

    Returns:
        str: Path to folder containing most frequent person's images
    """
    print(f"Processing images from {images_folder}...")

    # Create output directories
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

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
        image_files[i : i + batch_size] for i in range(0, len(image_files), batch_size)
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

    # Clustering and identification
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

    # Use improved merging algorithm
    merged_clusters = improved_merge_similar_clusters(
        cluster_counts,
        embeddings_array,
        merge_threshold=merge_threshold,
        used_model=used_model,
        valid_faces=valid_faces if enhanced_merging else None,
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

    # Copy unique images to the output folder with non-main faces blurred
    print(f"Processing {len(unique_sources)} images with face blurring...")

    # Collect all detected faces for synchronized display
    all_detected_faces = []

    for i, source in tqdm(
        enumerate(unique_sources),
        total=len(unique_sources),
        desc="Blurring non-main faces",
    ):
        # Read the source image
        img = safe_imread(source)
        if img is None:
            continue

        # Detect all faces in the image
        faces = safe_face_detection(
            source, detector_backend=backends[0], enforce_detection=False
        )

        if faces:
            # Get multiple reference embeddings from the most frequent person's cluster
            # Use up to 5 different faces as reference for better matching
            num_references = min(5, len(most_frequent_face_indices))
            reference_embeddings = [
                embeddings_array[most_frequent_face_indices[j]]
                for j in range(num_references)
            ]

            # Blur non-main faces and get detected face data
            img_with_blur, detected_faces = blur_non_main_faces(
                img,
                faces,
                reference_embeddings,
                model=used_model,
                verification_threshold=optimized_threshold
                * 1.2,  # Slightly more permissive
                most_frequent_label=most_frequent_label,
                all_clusters=cluster_counts,  # Pass all clusters for department assignment
                all_embeddings=embeddings_array,  # Pass all embeddings
            )

            # Add detected faces to collection
            all_detected_faces.extend(detected_faces)

            # Save the processed image
            output_path = os.path.join(
                most_frequent_folder, f"image_{i+1}{os.path.splitext(source)[1]}"
            )
            cv2.imwrite(output_path, img_with_blur)
        else:
            # If no faces detected, just copy the original image
            shutil.copy(
                source,
                os.path.join(
                    most_frequent_folder, f"image_{i+1}{os.path.splitext(source)[1]}"
                ),
            )

    print(
        f"Saved {len(unique_sources)} images of the most frequent person to {most_frequent_folder} (with non-main faces blurred)"
    )

    # Create synchronized face display
    if all_detected_faces:
        print("Creating synchronized face display by departments...")
        synchronized_display_path = os.path.join(output_folder, "face_departments.jpg")

        create_synchronized_face_display(
            all_detected_faces,
            cluster_counts,  # Use all clusters for display
            synchronized_display_path,
        )

        print(f"Face departments display saved to {synchronized_display_path}")

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
        )

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

    # Use cm. rainbow instead of get_cmap for better compatibility
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
    """
    # Convert embeddings to numpy arrays if they aren't already
    if not isinstance(embedding1, np.ndarray):
        embedding1 = np.array(embedding1)
    if not isinstance(embedding2, np.ndarray):
        embedding2 = np.array(embedding2)

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
    }

    threshold = thresholds.get(model_name, 0.4)

    # Calculate distance based on metric
    if metric == "cosine":
        distance = cosine(embedding1, embedding2)
    elif metric == "euclidean":
        distance = euclidean(embedding1, embedding2)
    elif metric == "euclidean_l2":
        distance = euclidean(
            embedding1 / np.linalg.norm(embedding1),
            embedding2 / np.linalg.norm(embedding2),
        )
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Verify if distance is below the threshold
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
    Advanced cluster merging with improved similarity metrics and validation.
    This is the primary cluster merging function that should be used.
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


def enhance_face_crop(face_crop, preserve_skin_tone=True):
    """
    Enhanced face crop processing with better quality preservation.

    Args:
        face_crop: Original face crop image
        preserve_skin_tone: Whether to preserve natural skin tones

    Returns:
        Enhanced face crop
    """
    try:
        # Convert to float32 for better precision in calculations
        face_float = face_crop.astype(np.float32) / 255.0

        # Preserve original colors for reference
        original_colors = face_float.copy()

        # Convert to LAB color space for better color handling
        face_lab = cv2.cvtColor(face_float, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(face_lab)

        # Enhance lighting using CLAHE (more natural than regular histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(np.uint8(l * 255)) / 255.0

        # Adaptive contrast enhancement
        mean_l = np.mean(l)
        alpha = 1.0 + (0.5 - mean_l)  # Adjust contrast based on brightness
        l_enhanced = np.clip(alpha * (l_enhanced - mean_l) + mean_l, 0, 1)

        # Merge back with original color channels
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        if preserve_skin_tone:
            # Create a skin tone mask
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            face_hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(face_hsv, lower_skin, upper_skin)

            # Blend enhanced image with original colors in skin regions
            skin_mask_float = skin_mask.astype(np.float32) / 255.0
            skin_mask_float = np.dstack([skin_mask_float] * 3)
            enhanced_bgr = (
                enhanced_bgr * (1 - skin_mask_float) + original_colors * skin_mask_float
            )

        # Subtle sharpening using unsharp mask (more natural than kernel sharpening)
        gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.0)
        enhanced_bgr = cv2.addWeighted(enhanced_bgr, 1.5, gaussian, -0.5, 0)

        # Convert back to uint8
        enhanced_bgr = np.clip(enhanced_bgr * 255, 0, 255).astype(np.uint8)

        # Final noise reduction (very subtle)
        enhanced_bgr = cv2.fastNlMeansDenoisingColored(enhanced_bgr, None, 3, 3, 7, 21)

        return enhanced_bgr

    except Exception as e:
        print(f"Warning: Enhancement failed, returning original image: {e}")
        return face_crop


def process_face_crop(args):
    """
    Process a single face crop in parallel.
    """
    (
        img_file,
        img_path,
        reference_embedding,
        min_confidence,
        model,
        detection_backend,
        min_face_size,
        padding_factor,
        enhance_quality,
    ) = args

    try:
        faces = safe_face_detection(
            img_path, detector_backend=detection_backend, enforce_detection=False
        )

        if not faces:
            return None

        # Get the largest face as reference
        largest_face = max(
            faces, key=lambda x: x["facial_area"]["w"] * x["facial_area"]["h"]
        )
        if largest_face["confidence"] < min_confidence:
            return None

        # Calculate adaptive padding based on face size
        area = largest_face["facial_area"]
        if area["w"] < min_face_size[0] or area["h"] < min_face_size[1]:
            return None

        size_factor = min(1.0, np.sqrt((area["w"] * area["h"]) / (224 * 224)))
        adaptive_padding = padding_factor * (1.0 + size_factor)

        # Extract face with padding
        img = safe_imread(img_path)
        if img is None:
            return None

        x = max(0, area["x"] - int(area["w"] * adaptive_padding))
        y = max(0, area["y"] - int(area["h"] * adaptive_padding))
        w = min(img.shape[1] - x, area["w"] + int(area["w"] * adaptive_padding * 2))
        h = min(img.shape[0] - y, area["h"] + int(area["h"] * adaptive_padding * 2))

        face_crop = img[y : y + h, x : x + w]

        # Get face embedding
        emb = safe_represent(img_path, model_name=model)
        if not emb:
            return None

        # Compare with reference
        similarity = 1 - cosine(emb[0]["embedding"], reference_embedding)

        # Calculate quality metrics
        quality_score = assess_face_quality(largest_face["face"])

        # Detect if it's a profile shot
        profile_score = detect_profile_angle(face_crop)

        # Check if there are other faces in the image
        single_person = len(faces) == 1

        # Enhance if requested
        if enhance_quality:
            face_crop = enhance_face_crop(face_crop)

        return {
            "crop": face_crop,
            "similarity": similarity,
            "quality_score": quality_score,
            "profile_score": profile_score,
            "single_person": single_person,
            "size": area["w"] * area["h"],
            "filename": img_file,
        }
    except Exception as e:
        print(f"Error processing crop from {img_file}: {e}")
        return None


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
    min_face_size=(30, 30),
):
    """
    Select and save the best face crops from the main person's images using parallel processing.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files
    image_files = [
        f
        for f in os.listdir(images_folder)
        if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
    ]

    if not image_files:
        print(f"No images found in {images_folder}")
        return 0

    # Create reference embeddings from multiple images
    print("Creating reference embeddings...")
    reference_embeddings = []
    for img_file in image_files[: min(30, len(image_files))]:
        img_path = os.path.join(images_folder, img_file)
        try:
            emb = safe_represent(img_path, model_name=model, enforce_detection=False)
            if emb:
                reference_embeddings.append(emb[0]["embedding"])
        except Exception as e:
            print(f"Warning: Could not process reference image {img_file}: {e}")
            continue

    if not reference_embeddings:
        print("Could not create reference embeddings")
        return 0

    # Calculate average reference embedding
    reference_embedding = np.mean(reference_embeddings, axis=0)

    # Prepare arguments for parallel processing
    process_args = [
        (
            img_file,
            os.path.join(images_folder, img_file),
            reference_embedding,
            min_confidence,
            model,
            detection_backend,
            min_face_size,
            padding_factor,
            enhance_quality,
        )
        for img_file in image_files
    ]

    # Process face crops in parallel
    print("Processing face crops in parallel...")
    with Pool(max(1, cpu_count() - 1)) as pool:
        face_candidates = list(
            tqdm(
                pool.imap(process_face_crop, process_args),
                total=len(process_args),
                desc="Processing face crops",
            )
        )

    # Filter out None results and sort candidates
    face_candidates = [fc for fc in face_candidates if fc is not None]

    if not face_candidates:
        print("No valid face crops found")
        return 0

    # Sort candidates by multiple criteria
    face_candidates.sort(
        key=lambda x: (
            x["similarity"],  # Match to reference
            x["quality_score"],  # Image quality
            x["size"],  # Face size
            x["single_person"],  # Prefer single person shots
            -abs(0.5 - x["profile_score"]),  # Mix of profile and frontal shots
        ),
        reverse=True,
    )

    # Save the best crops
    saved_count = 0
    for i, candidate in enumerate(face_candidates):
        if i >= max_images:
            break

        try:
            face_crop = candidate["crop"]

            # Save crop with descriptive filename
            crop_type = "profile" if candidate["profile_score"] > 0.6 else "frontal"
            persons = "single" if candidate["single_person"] else "multi"
            quality_text = f"{int(candidate['quality_score']*100)}"
            crop_filename = f"best_face_{i+1}_{crop_type}_{persons}_q{quality_text}.jpg"

            cv2.imwrite(
                os.path.join(output_folder, crop_filename),
                face_crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            saved_count += 1

        except Exception as e:
            print(f"Error saving crop {i}: {e}")
            continue

    print(f"Saved {saved_count} best face crops to {output_folder}")
    return saved_count


def detect_profile_angle(face_img):
    """
    Estimate if an image is a profile shot based on face landmarks.
    Returns a score between 0 (frontal) and 1 (complete profile).
    """
    try:
        # Convert to grayscale
        gray = (
            cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            if len(face_img.shape) > 2
            else face_img
        )

        # Use facial landmarks to detect profile angle
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )

        results = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return 0.5  # Default to middle if can't detect landmarks

        landmarks = results.multi_face_landmarks[0].landmark

        # Calculate face symmetry using landmark positions
        left_eye = np.mean(
            [(landmarks[33].x, landmarks[33].y), (landmarks[133].x, landmarks[133].y)],
            axis=0,
        )
        right_eye = np.mean(
            [
                (landmarks[362].x, landmarks[362].y),
                (landmarks[263].x, landmarks[263].y),
            ],
            axis=0,
        )

        nose_tip = (landmarks[4].x, landmarks[4].y)

        # Calculate relative position of nose to eye line
        eye_line_center = (left_eye + right_eye) / 2
        eye_line_vector = right_eye - left_eye
        nose_offset = np.cross(eye_line_vector, nose_tip - eye_line_center)

        # Normalize to 0-1 range
        profile_score = min(1.0, max(0.0, abs(nose_offset) * 2))

        return profile_score

    except Exception:
        return 0.5  # Default to middle if detection fails


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


def blur_non_main_faces(
    img,
    faces,
    reference_embedding,
    model="Facenet512",
    verification_threshold=0.4,
    most_frequent_label=None,
    all_clusters=None,
    all_embeddings=None,
):
    """
    Blur all faces in the image except for the main person.
    Uses multiple verification steps to ensure accurate identification.

    Args:
        img: Input image
        faces: List of detected faces with their locations
        reference_embedding: Embedding of the main person's face (or list of embeddings)
        model: Face recognition model to use
        verification_threshold: Base threshold for face verification
        most_frequent_label: Label of the most frequent person's cluster
        all_clusters: Dictionary of all clusters for department assignment
        all_embeddings: Array of all face embeddings

    Returns:
        Image with non-main faces blurred, and list of detected face data
    """
    img_with_blur = img.copy()
    detected_faces_data = []

    # Convert reference_embedding to list if it's a single embedding
    if not isinstance(reference_embedding, list):
        reference_embeddings = [reference_embedding]
    else:
        reference_embeddings = reference_embedding

    # Sort faces by size (larger faces are more likely to be the main person)
    faces = sorted(
        faces, key=lambda x: x["facial_area"]["w"] * x["facial_area"]["h"], reverse=True
    )

    # Keep track of which faces to blur
    faces_to_blur = []

    # First pass: identify faces with high confidence
    for face_idx, face_obj in enumerate(faces):
        if (
            face_obj["confidence"] < 0.1
        ):  # Very low confidence threshold to catch more faces
            faces_to_blur.append(face_idx)
            continue

        # Get face area
        facial_area = face_obj["facial_area"]
        x, y = facial_area["x"], facial_area["y"]
        w, h = facial_area["w"], facial_area["h"]

        # Skip very small faces
        if w < 20 or h < 20:  # Lower threshold to catch more faces
            faces_to_blur.append(face_idx)
            continue

        # Get face embedding
        face_img = face_obj["face"]
        face_img = ensure_valid_image(face_img)

        # Save face temporarily
        temp_face_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"temp_face_blur_{x}_{y}.jpg"
        )
        cv2.imwrite(temp_face_path, face_img)

        # Get embedding from aligned face
        embedding = safe_represent(
            temp_face_path, model_name=model, enforce_detection=False
        )

        # Clean up temp file
        if os.path.exists(temp_face_path):
            os.remove(temp_face_path)

        if embedding and len(embedding) > 0:
            face_embedding = embedding[0]["embedding"]

            # Calculate minimum distance across all reference embeddings
            min_distance = float("inf")
            for ref_emb in reference_embeddings:
                distance = cosine(ref_emb, face_embedding)
                min_distance = min(min_distance, distance)

            # Adaptive threshold based on face size and quality
            size_factor = min(1.0, np.sqrt((w * h) / (224 * 224)))
            quality_score = assess_face_quality(face_img)

            # More permissive threshold for high quality, large faces
            adaptive_threshold = verification_threshold * (
                1.2 + 0.3 * size_factor + 0.2 * quality_score
            )

            # Determine if this is the main person
            is_main_person = min_distance <= adaptive_threshold

            # Determine cluster/department assignment
            assigned_cluster = None
            if is_main_person:
                assigned_cluster = most_frequent_label
            elif all_clusters and all_embeddings is not None:
                # Try to assign to a department/cluster
                best_cluster = None
                best_distance = float("inf")

                # Compare with representatives from each cluster
                for cluster_label, face_indices in all_clusters.items():
                    if cluster_label == most_frequent_label:
                        continue  # Skip main person's cluster

                    # Use up to 3 representatives from each cluster
                    num_representatives = min(3, len(face_indices))
                    for i in range(num_representatives):
                        idx = face_indices[i]
                        cluster_embedding = all_embeddings[idx]
                        distance = cosine(face_embedding, cluster_embedding)

                        if (
                            distance < best_distance and distance < 0.5
                        ):  # Threshold for assignment
                            best_distance = distance
                            best_cluster = cluster_label

                assigned_cluster = best_cluster

            # Store face data for synchronized display
            detected_faces_data.append(
                {
                    "embedding": face_embedding,
                    "face_img": face_img,
                    "position": (x, y, w, h),
                    "quality": quality_score,
                    "distance": min_distance,
                    "is_main_person": is_main_person,
                    "cluster_label": assigned_cluster,
                }
            )

            # If this is not the main person, mark for blurring
            if not is_main_person:
                faces_to_blur.append(face_idx)
        else:
            # If we couldn't get embeddings, blur the face
            faces_to_blur.append(face_idx)

    # Second pass: apply blurring
    for face_idx in faces_to_blur:
        face_obj = faces[face_idx]
        facial_area = face_obj["facial_area"]
        x, y = facial_area["x"], facial_area["y"]
        w, h = facial_area["w"], facial_area["h"]

        # Extract face region
        face_region = img_with_blur[y : y + h, x : x + w]

        # Apply strong Gaussian blur
        kernel_size = max(99, min(w, h) // 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 30)

        # Replace the region with blurred version
        img_with_blur[y : y + h, x : x + w] = blurred

    return img_with_blur, detected_faces_data


def create_synchronized_face_display(
    all_detected_faces, cluster_data, output_path, face_size=(150, 150)
):
    """
    Create a synchronized display of all detected faces organized by departments/clusters.

    Args:
        all_detected_faces: List of detected face data from all images
        cluster_data: Dictionary mapping cluster labels to face indices
        output_path: Path to save the output image
        face_size: Size to resize each face thumbnail

    Returns:
        Path to the saved image
    """
    if not all_detected_faces:
        print("No faces to display")
        return None

    # Group faces by cluster (department)
    faces_by_cluster = {}
    unclustered_faces = []

    # First, process faces with assigned clusters
    for face_data in all_detected_faces:
        if face_data["cluster_label"] is not None:
            if face_data["cluster_label"] not in faces_by_cluster:
                faces_by_cluster[face_data["cluster_label"]] = []
            faces_by_cluster[face_data["cluster_label"]].append(face_data)
        else:
            # Try to assign to a cluster based on similarity
            best_cluster = None
            best_distance = float("inf")

            for cluster_label, cluster_faces in faces_by_cluster.items():
                if cluster_faces:
                    # Compare with all faces in the cluster
                    for cluster_face in cluster_faces:
                        if "embedding" in cluster_face and "embedding" in face_data:
                            distance = cosine(
                                cluster_face["embedding"], face_data["embedding"]
                            )
                            if (
                                distance < best_distance and distance < 0.45
                            ):  # Stricter threshold for assignment
                                best_distance = distance
                                best_cluster = cluster_label

            if best_cluster is not None:
                # Assign to best matching cluster
                face_data["cluster_label"] = best_cluster
                faces_by_cluster[best_cluster].append(face_data)
            else:
                # Keep as unclustered
                unclustered_faces.append(face_data)

    # Sort clusters by size (number of faces)
    sorted_clusters = sorted(
        faces_by_cluster.keys(), key=lambda x: len(faces_by_cluster[x]), reverse=True
    )

    # Calculate layout
    total_faces = sum(len(faces) for faces in faces_by_cluster.values()) + len(
        unclustered_faces
    )

    # Determine grid dimensions based on number of clusters and faces
    num_clusters = len(faces_by_cluster)

    if num_clusters == 0:
        print("No clusters to display")
        return None

    # Estimate rows and columns for the grid
    faces_per_row = min(10, int(np.ceil(np.sqrt(total_faces))))

    # Create a blank canvas with white background
    # First calculate total height needed
    total_height = 0

    # Add title section
    title_height = 60
    total_height += title_height

    # Add department sections
    for cluster in sorted_clusters:
        cluster_faces = faces_by_cluster[cluster]
        rows_needed = int(np.ceil(len(cluster_faces) / faces_per_row))
        total_height += (
            rows_needed * face_size[1] + 60
        )  # Add space for department header

    # Add space for unclustered faces if any
    if unclustered_faces:
        rows_needed = int(np.ceil(len(unclustered_faces) / faces_per_row))
        total_height += rows_needed * face_size[1] + 60

    # Create canvas
    canvas_width = faces_per_row * face_size[0]
    canvas = np.ones((total_height, canvas_width, 3), dtype=np.uint8) * 255

    # Draw title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "Face Departments - Organized by Similarity",
        (10, 40),
        font,
        1.0,
        (0, 0, 0),
        2,
    )

    # Draw faces by department (cluster)
    y_offset = title_height

    # First draw the main person's department
    main_cluster = sorted_clusters[0] if sorted_clusters else None

    for cluster_idx, cluster in enumerate(sorted_clusters):
        cluster_faces = faces_by_cluster[cluster]

        # Determine if this is the main person's cluster
        is_main_cluster = cluster == main_cluster

        # Draw department header with different styling based on importance
        header_color = (0, 100, 0) if is_main_cluster else (0, 0, 0)
        header_text = f"Department {cluster}" + (
            " (Main Person)" if is_main_cluster else ""
        )

        # Draw department separator line
        cv2.line(
            canvas,
            (0, y_offset + 10),
            (canvas_width, y_offset + 10),
            (200, 200, 200),
            2,
        )

        # Draw department header
        cv2.putText(
            canvas, header_text, (10, y_offset + 40), font, 0.9, header_color, 2
        )
        y_offset += 60

        # Sort faces within department by quality and whether they're the main person
        cluster_faces.sort(
            key=lambda x: (x.get("is_main_person", False), x.get("quality", 0)),
            reverse=True,
        )

        # Draw faces in this department
        for i, face_data in enumerate(cluster_faces):
            face_img = face_data["face_img"]

            # Resize face to standard size
            face_resized = cv2.resize(face_img, face_size)

            # Calculate position
            row = i // faces_per_row
            col = i % faces_per_row

            x = col * face_size[0]
            y = y_offset + row * face_size[1]

            # Place face on canvas
            try:
                canvas[y : y + face_size[1], x : x + face_size[0]] = face_resized

                # Add green frame for main person
                if face_data.get("is_main_person", False):
                    cv2.rectangle(
                        canvas,
                        (x, y),
                        (x + face_size[0], y + face_size[1]),
                        (0, 255, 0),
                        3,
                    )

                # Add quality score
                quality = int(face_data.get("quality", 0) * 100)
                cv2.putText(
                    canvas,
                    f"Q:{quality}",
                    (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            except Exception as e:
                print(f"Error placing face on canvas: {e}")
                continue

        # Update y_offset for next department
        rows_used = int(np.ceil(len(cluster_faces) / faces_per_row))
        y_offset += rows_used * face_size[1]

    # Add unclustered faces if any
    if unclustered_faces:
        # Draw department separator line
        cv2.line(
            canvas,
            (0, y_offset + 10),
            (canvas_width, y_offset + 10),
            (200, 200, 200),
            2,
        )

        cv2.putText(
            canvas,
            f"Unassigned Faces",
            (10, y_offset + 40),
            font,
            0.9,
            (100, 100, 100),
            2,
        )
        y_offset += 60

        for i, face_data in enumerate(unclustered_faces):
            face_img = face_data["face_img"]

            # Resize face to standard size
            face_resized = cv2.resize(face_img, face_size)

            # Calculate position
            row = i // faces_per_row
            col = i % faces_per_row

            x = col * face_size[0]
            y = y_offset + row * face_size[1]

            # Place face on canvas
            try:
                canvas[y : y + face_size[1], x : x + face_size[0]] = face_resized

                # Add quality score
                quality = int(face_data.get("quality", 0) * 100)
                cv2.putText(
                    canvas,
                    f"Q:{quality}",
                    (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
            except Exception as e:
                print(f"Error placing unclustered face on canvas: {e}")
                continue

    # Save the result
    cv2.imwrite(output_path, canvas)
    print(f"Saved synchronized face display to {output_path}")

    return output_path


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

    # Visualization Options
    viz_group = parser.add_argument_group("Visualization Options")
    viz_group.add_argument(
        "--visualize-clusters",
        action="store_true",
        default=True,
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
        models=[args.model] + ["VGG-Face", "Facenet", "OpenFace", "DeepFace"],
        merge_threshold=args.merge_threshold,
        face_quality_threshold=args.face_quality,
        enhanced_merging=use_improved_merging,
        verify_identity=args.verify_identity,
        visualize_before_merge=args.visualize_clusters,
        save_best_crops=save_best_crops,
        max_best_crops=args.max_crops,
    )
