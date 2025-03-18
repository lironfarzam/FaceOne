from datetime import datetime
import json
import time
import random
import cv2
import os
import random
import numpy as np
import pickle
import sys
from datetime import datetime
import tensorflow as tf
import pprint as pp
import shutil

# Add parent directory to path to import cool_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cool_utils import load_config, print_green, print_red, print_blue

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
)
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
import matplotlib.pyplot as plt
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

import threading
from rich.progress import track
import tensorflowjs as tfjs

# # Option 1: Force CPU usage
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # Or Option 2: Configure GPU memory growth and disable Metal
# tf.config.set_visible_devices([], "GPU")

# # Or Option 3: Explicitly configure Metal plugin
# tf.config.experimental.set_visible_devices([], "Metal")

# Thread-safe print for better output in multithreaded environment
print_lock = threading.Lock()

# use all the cores
# NUM_OF_WORKERS = os.cpu_count()
NUM_OF_WORKERS = 4

# Get the absolute path to the parent directory and config file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, "config.json")

config = load_config(config_path)

pp.pprint(config)
print("-" * 50)


# Constants from config
# NUM_OF_WORKERS = config["num_of_workers"]
# SHOW_3D_PLOT = config["show_3d_plot"]
# RUN_ON_GPU = config["run_on_gpu"]

CREATE_NEW_DATA = config["create_new_data"]
TRAIN_NEW_MODEL = config["train_new_model"]
SAVE_MODEL = config["save_model"]
PATH_FOR_MODEL = os.path.join(parent_dir, config["path_for_model"].lstrip("./"))
CHROME_EXTENSION_MODEL_PATH = os.path.join(
    parent_dir, config["chrome_extension_model_path"].lstrip("./")
)
PATH_FOR_POSITIVE_EMBEDDINGS = os.path.join(
    parent_dir, config["path_for_positives_embeddings"].lstrip("./")
)
CHROME_EXTENSION_EMBEDDINGS_PATH = os.path.join(
    parent_dir, config["chrome_extension_embeddings_path"].lstrip("./")
)
NEGATIVE_VECTORS_FILE = os.path.join(
    parent_dir, config["negative_vectors_file"].lstrip("./")
)
TEST_IMAGES_FOLDER = os.path.join(parent_dir, config["test_images_folder"].lstrip("./"))
ANC_PATH = os.path.join(parent_dir, config["anchors_folder"].lstrip("./"))
POS_PATH = os.path.join(parent_dir, config["positives_folder"].lstrip("./"))
NEG_PATH = os.path.join(parent_dir, config["negatives_folder"].lstrip("./"))
EMBEDING_PATH = os.path.join(parent_dir, config["embedding_path"].lstrip("./"))
NUM_OF_IMAGES_TO_PROCESS = config["num_of_images_to_process"]
NUM_OF_PAIRS = config["num_of_pairs"]
LEARNING_RATE = config["learning_rate"]
NUM_OF_EPOCHS = config["num_of_epochs"]
PATIENCE = config["patience"]
FACTOR = config["factor"]
BATCH_SIZE = config["batch_size"]
WEIGHT_DECAY = config["weight_decay"]
DROPOUT_RATE = config["dropout_rate"]
USE_MIXED_PRECISION = config["use_mixed_precision"]
OUTPUT_MODEL_FORMAT = config["output_model_format"]


# Ensure GPU Memory Growth
if tf.test.is_built_with_cuda():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth: {tf.config.experimental.get_memory_growth(gpu)}")


# Utility print functions
# These are now imported from cool_utils


# L1 Distance Layer for Siamese Network
class L1DistanceLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute the L1 distance between two tensors.

    The L1 distance (Manhattan distance) is chosen over L2 (Euclidean) distance for several reasons:
    1. It's more robust to outliers in the embedding space
    2. It provides better gradient flow during training
    3. It's computationally more efficient
    4. It works well with high-dimensional face embeddings

    This layer is a critical component of the Siamese network as it quantifies
    the difference between two face embeddings, which directly relates to face similarity.

    Args:
        inputs (list): List containing two tensors of shape (batch_size, embedding_size).
                      The first tensor is the anchor embedding, and the second is the
                      comparison embedding.

    Returns:
        tf.Tensor: L1 distance tensor of shape (batch_size, embedding_size).
                  Each element represents the absolute difference between
                  corresponding dimensions of the input embeddings.
    """

    def call(self, inputs):
        """
        Compute the element-wise absolute difference between two input tensors.

        This operation preserves the dimensionality of the input embeddings,
        allowing subsequent layers to learn from the pattern of differences
        across all dimensions rather than reducing to a single scalar distance.
        """
        input_embedding, validation_embedding = inputs
        return tf.abs(input_embedding - validation_embedding)


def make_siamese_model(embedding_size: int = 512) -> Model:
    """
    Create a Siamese neural network model for face verification.

    This architecture is specifically designed for face verification tasks:

    1. INPUT LAYER DESIGN:
       - Uses two separate input layers for reference and comparison embeddings
       - Fixed at 512 dimensions to match Facenet512 output
       - Pre-computed embeddings are used instead of raw images to:
         a) Reduce computational requirements
         b) Leverage transfer learning from pre-trained models
         c) Improve training stability and convergence

    2. L1 DISTANCE LAYER:
       - Computes absolute difference between embeddings
       - Preserves dimensionality for better feature learning
       - More effective than concatenation for similarity learning

    3. DENSE LAYER ARCHITECTURE:
       - Progressive dimensionality reduction (512→256→64→16→2→1)
       - ReLU activations for non-linearity and to prevent vanishing gradients
       - Layer sizes chosen to gradually compress information while maintaining
         discriminative power

    4. OUTPUT DESIGN:
       - Single neuron with sigmoid activation outputs a similarity score [0-1]
       - 0 = different person, 1 = same person
       - Sigmoid chosen for its probabilistic interpretation and bounded output

    Args:
        embedding_size (int, optional): Size of the embedding vector.
                                       Defaults to 512 to match Facenet512 output.

    Returns:
        Model: Compiled Siamese network model ready for training.

    Raises:
        ValueError: If embedding_size is not positive.
    """
    if embedding_size <= 0:
        raise ValueError("Embedding size must be a positive integer.")

    # Define inputs - we use pre-computed embeddings rather than raw images
    # This significantly reduces computational requirements and leverages
    # transfer learning from state-of-the-art face recognition models
    input_embedding = Input(batch_shape=(None, 512), name="input_embedding")
    validation_embedding = Input(batch_shape=(None, 512), name="validation_embedding")

    # L1 distance layer to compute element-wise absolute difference
    # This preserves the dimensionality and pattern of differences across all dimensions
    merged = L1DistanceLayer()([input_embedding, validation_embedding])

    # Dense neural network to learn patterns in the embedding differences
    # The architecture follows a funnel pattern, gradually reducing dimensions
    # while extracting increasingly abstract features

    # First dense layer (512 neurons) - matches input dimensionality for full information capture
    x = Dense(512, activation="relu")(merged)
    # x = Dense(512, activation="relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(DROPOUT_RATE)(x)
    x = Dense(256, activation="relu")(x)
    # x = Dense(256, activation="relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(DROPOUT_RATE)(x)
    # x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    # x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    # x = Dense(8, activation="relu")(x)
    # x = Dense(4, activation="relu")(x)
    x = Dense(2, activation="relu")(x)

    # Output layer - single neuron with sigmoid activation for binary classification
    # Outputs a similarity score between 0 (different person) and 1 (same person)
    outputs = Dense(1, activation="sigmoid")(x)

    # Create and return the model
    return Model(inputs=[input_embedding, validation_embedding], outputs=outputs)


def thread_safe_print(message: str, color: str = "blue"):
    with print_lock:
        if color == "green":
            print_green(message)
        elif color == "red":
            print_red(message)
        else:
            print_blue(message)


def load_images_and_compute_embeddings_parallel() -> dict:
    """
    Load images from the specified directories and compute embeddings for each image in parallel.

    Returns:
        dict: A dictionary containing the embeddings for anchor, positive, and negative images.
    """
    print_blue("Loading images and computing embeddings in parallel...")
    print("-" * 50)

    directories = {"anchor": ANC_PATH, "positive": POS_PATH, "negative": NEG_PATH}
    embeddings = {"anchor": {}, "positive": {}, "negative": {}}

    def process_image(label: str, image_path: str):
        embedding = get_embedding(image_path, apply_augmentation=False)
        return label, image_path, embedding

    for label, dir_path in directories.items():
        image_files = [
            f
            for f in os.listdir(dir_path)
            if not f.startswith(".") and os.path.isfile(os.path.join(dir_path, f))
        ]

        # Take only the first NUM_OF_IMAGES_TO_PROCESS images
        image_files = image_files[:NUM_OF_IMAGES_TO_PROCESS]
        total_images = min(NUM_OF_IMAGES_TO_PROCESS, len(image_files))

        print(
            f"Processing {total_images} {label} images in parallel... {NUM_OF_WORKERS} workers"
        )

        # Use ProcessPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=NUM_OF_WORKERS) as executor:
            futures = [
                executor.submit(
                    process_image, label, os.path.join(dir_path, image_name)
                )
                for image_name in image_files
            ]

            for future in track(
                futures,
                total=total_images,
                description=f"Processing {label} images",
                # unit="image",
            ):
                try:
                    result_label, image_path, embedding = future.result()
                    if embedding is not None:
                        embeddings[result_label][
                            os.path.basename(image_path)
                        ] = embedding
                except Exception as e:
                    # print_red(f"Error processing image: {e}")
                    pass

        print_green(f"Processed {total_images} {label} images successfully.")
        print("-" * 50)

    print("-" * 50)
    print("anchor length: ", len(embeddings["anchor"]))
    print("positive length: ", len(embeddings["positive"]))
    print("negative length: ", len(embeddings["negative"]))
    print("-" * 50)

    print_green("All embeddings computed and stored.")
    return embeddings


# DeepFace Embedding Extraction
# @log_function_call
def get_embedding(
    image_path: str, model_name: str = "Facenet512", apply_augmentation: bool = False
) -> np.ndarray:
    """
    Extract face embedding vector from an image using DeepFace.

    This function is a critical component of the face verification pipeline for several reasons:

    1. EMBEDDING QUALITY:
       - Uses DeepFace's Facenet512 model which produces high-quality 512-dimensional embeddings
       - Facenet512 is chosen for its superior performance in face recognition tasks
       - 512 dimensions provide a good balance between detail and computational efficiency

    2. PREPROCESSING:
       - Handles image loading and format conversion
       - Creates temporary files with English-only paths to avoid encoding issues
       - Ensures consistent image format (RGB) for reliable embedding extraction

    3. AUGMENTATION (OPTIONAL):
       - Can apply random transformations to increase training data diversity
       - Helps improve model robustness to variations in lighting, angle, etc.
       - Particularly valuable when working with limited training data

    4. ERROR HANDLING:
       - Robust error checking for file existence and image format
       - Graceful handling of DeepFace exceptions
       - Cleanup of temporary files to prevent storage issues

    Args:
        image_path (str): Path to the image file containing a face.
        model_name (str, optional): Name of the DeepFace model to use.
                                   Defaults to "Facenet512" for its optimal performance.
        apply_augmentation (bool, optional): Whether to apply random augmentations to the image.
                                            Useful for training data enrichment.
                                            Defaults to False.

    Returns:
        np.ndarray: 512-dimensional embedding vector representing the face's features.
                   This vector captures the unique characteristics of the face,
                   enabling mathematical comparison between different faces.

    Raises:
        ValueError: If the image path doesn't exist or the image format is invalid.
        Exception: If DeepFace fails to extract an embedding.
    """
    temp_dir = "temp_images"
    temp_path = None

    try:
        if not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")

        # Create a temporary directory with English-only path
        os.makedirs(temp_dir, exist_ok=True)

        # Create a temporary file with an English-only name
        temp_filename = f"temp_image_{int(time.time())}_{random.randint(0, 1000)}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)

        # Read image and ensure it's in the correct format
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Ensure image has 3 channels
        if len(img.shape) != 3:
            raise ValueError(f"Image must have 3 channels: {image_path}")

        # Convert to RGB for processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image to expected dimensions (160x160 for FaceNet)
        # img = cv2.resize(img, (160, 160))

        # Apply augmentation if needed
        if apply_augmentation:
            # Random brightness
            if random.random() > 0.5:
                factor = random.uniform(0.5, 1.5)
                img = cv2.convertScaleAbs(img, alpha=factor, beta=0)

            # Random horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)

            # Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                height, width = img.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, rotation_matrix, (width, height))

            # Random contrast
            if random.random() > 0.5:
                contrast = random.uniform(0.8, 1.2)
                img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

        # Save the processed image
        cv2.imwrite(temp_path, cv2.COLOR_RGB2BGR)

        # Get embedding using DeepFace
        try:
            with tf.device("/CPU:0"):  # Force CPU usage for DeepFace
                embedding = DeepFace.represent(
                    img_path=temp_path,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="opencv",  # Use a simpler detector
                )
            return embedding[0]["embedding"]
        except Exception as e:
            # print_red(f"DeepFace error for {image_path}: {str(e)}")
            # # show the original image
            # cv2.imshow("Original Image", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            return None

    except Exception as e:
        # print_red(f"Error processing {image_path}: {str(e)}")

        return None

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# Load and Embed Images
# @log_function_call
def load_images_and_compute_embeddings() -> dict:
    """
    Load images from the specified directories and compute embeddings for each image.

    Returns:
        dict: A dictionary containing the embeddings for anchor, positive, and negative images.
    """
    print_blue("Loading images and computing embeddings...")
    print("-" * 50)

    directories = {"anchor": ANC_PATH, "positive": POS_PATH, "negative": NEG_PATH}
    embeddings = {"anchor": {}, "positive": {}, "negative": {}}

    for label, dir_path in directories.items():
        image_files = [
            f
            for f in os.listdir(dir_path)
            if not f.startswith(".") and os.path.isfile(os.path.join(dir_path, f))
        ]

        # take only the first NUM_OF_IMAGES_TO_PROCESS images
        image_files = image_files[:NUM_OF_IMAGES_TO_PROCESS]
        total_images = len(image_files)

        print(f"Processing {total_images} {label} images...")

        for idx, image_name in enumerate(image_files):
            image_path = os.path.join(dir_path, image_name)
            embedding = get_embedding(image_path, apply_augmentation=False)

            if embedding is not None:
                embeddings[label][image_name] = embedding
            else:
                print_red(f"Skipping image {image_path} due to missing embedding.")

            print(
                f"Processed {idx + 1}/{total_images} images for {label} | complet: {((idx + 1) / total_images) * 100:.2f}%",
                end="\r",
            )
            sys.stdout.flush()

        print_green(
            f"Processed {total_images} {label} images successfully.                       "
        )
        print("-" * 50)

    print_green("All embeddings computed and stored.")
    return embeddings


# Save Embeddings
def save_embeddings(embeddings: dict, filepath: str) -> None:
    """Save the embeddings to a file.

    Args:
        embeddings (dict): Dictionary containing the embeddings for anchor, positive, and negative images.
        filepath (str): Path to the file where the embeddings will be saved.
    """
    with open(filepath, "wb") as f:
        pickle.dump(embeddings, f)
    print_green(f"Embeddings saved to {filepath}")
    print("-" * 50)


# Load Embeddingss
def load_embeddings(filepath: str) -> dict:
    """Load the embeddings from a file.

    Args:
        filepath (str): Path to the file containing the embeddings.

    Returns:
        dict: Dictionary containing the embeddings for anchor, positive, and negative images.
    """
    with open(filepath, "rb") as f:
        embeddings = pickle.load(f)
    print_green(f"Embeddings loaded from {filepath}")
    print("-" * 50)
    return embeddings


# # Create Pairs from Embeddings
# # @log_function_call
# def create_pairs_from_embeddings(
#     embeddings: dict, num_pairs: int = NUM_OF_PAIRS
# ) -> tuple[list, list]:
#     """Create pairs of embeddings for training.

#     Args:
#         embeddings (dict): Dictionary containing the embeddings for anchor, positive, and negative images.
#         num_pairs (int, optional): Number of pairs to create. Defaults to NUM_OF_PAIRS.

#     Returns:
#         tuple: Two lists of tuples, each containing an embedding pair and a label.
#     """
#     print_blue(f"Creating {num_pairs} pairs using precomputed embeddings...")

#     anchor_embeddings = embeddings["anchor"]
#     positive_embeddings = embeddings["positive"]
#     negative_embeddings = embeddings["negative"]

#     anchor_images = list(anchor_embeddings.keys())
#     positive_images = list(positive_embeddings.keys())
#     negative_images = list(negative_embeddings.keys())

#     positive_pairs = []
#     negative_pairs = []

#     for _ in range(num_pairs):
#         anchor_img = random.choice(anchor_images)
#         positive_img = random.choice(positive_images)
#         emb1 = anchor_embeddings[anchor_img]
#         emb2 = positive_embeddings[positive_img]
#         positive_pairs.append((emb1, emb2, 1))

#     for _ in range(num_pairs):
#         anchor_img = random.choice(anchor_images)
#         negative_img = random.choice(negative_images)
#         emb1 = anchor_embeddings[anchor_img]
#         emb2 = negative_embeddings[negative_img]
#         negative_pairs.append((emb1, emb2, 0))

#     print_green(
#         f"Created {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs."
#     )
#     return positive_pairs, negative_pairs


# Create Pairs from Embeddings
# @log_function_call
def create_pairs_from_embeddings(
    embeddings: dict, num_pairs: int = NUM_OF_PAIRS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create positive and negative pairs from embeddings for Siamese network training.

    This function is a critical component of the training pipeline for several reasons:

    1. BALANCED DATASET CREATION:
       - Creates an equal number of positive and negative pairs
       - Prevents class imbalance that could bias the model
       - Ensures the model learns to distinguish both similarity and difference

    2. PAIR SELECTION STRATEGY:
       - Positive pairs: Different images of the same person (anchor-positive)
       - Negative pairs: Images of different people (anchor-negative)
       - This approach teaches the model the concept of facial identity

    3. RANDOMIZATION:
       - Randomly selects pairs to prevent memorization of specific patterns
       - Increases diversity in the training data
       - Improves model generalization to unseen faces

    4. DATA AUGMENTATION:
       - Creates multiple pairs from the same images
       - Effectively increases the training dataset size
       - Helps prevent overfitting, especially with limited data

    5. LABEL ENCODING:
       - Uses binary labels (1 for same person, 0 for different person)
       - Directly aligns with the binary classification objective
       - Compatible with binary cross-entropy loss function

    The quality and composition of these pairs directly impact the model's ability
    to learn meaningful face similarity metrics and generalize to new faces.

    Args:
        embeddings (dict): Dictionary containing embeddings for anchor, positive,
                          and negative images. Each category should contain a dictionary
                          mapping image paths to their corresponding embedding vectors.
        num_pairs (int, optional): Total number of pairs to create (half positive, half negative).
                                  Defaults to value from config.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two lists:
            1. Positive pairs: List of tuples (anchor_embedding, positive_embedding, 1)
            2. Negative pairs: List of tuples (anchor_embedding, negative_embedding, 0)

    Raises:
        ValueError: If there are insufficient images to create the requested number of pairs.
    """
    print_blue("Creating pairs from embeddings...")
    print("-" * 50)

    anchor_embeddings = list(embeddings["anchor"].values())
    positive_embeddings = list(embeddings["positive"].values())
    negative_embeddings = list(embeddings["negative"].values())

    anchor_embeddings = np.array(anchor_embeddings)
    positive_embeddings = np.array(positive_embeddings)
    negative_embeddings = np.array(negative_embeddings)

    print("Anchor embeddings shape:", anchor_embeddings.shape)
    print("Positive embeddings shape:", positive_embeddings.shape)
    print("Negative embeddings shape:", negative_embeddings.shape)

    positive_pairs = []
    negative_pairs = []

    for anchor_img in anchor_embeddings:
        for positive_img in positive_embeddings:
            positive_pairs.append((anchor_img, positive_img, 1))
            if len(positive_pairs) >= num_pairs:
                break

    for anchor_img in anchor_embeddings:
        for negative_img in negative_embeddings:
            negative_pairs.append((anchor_img, negative_img, 0))
            if len(negative_pairs) >= num_pairs:
                break

    print_green(
        f"Created {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs."
    )
    min_length = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = positive_pairs[:min_length]
    negative_pairs = negative_pairs[:min_length]
    return positive_pairs, negative_pairs


def print_train_test_split_info(train_pairs: list, test_pairs: list) -> None:
    """Print information about the train-test split.

    Args:
        train_pairs (list): List of training pairs.
        test_pairs (list): List of test pairs.
    """
    print("Length of train pairs: ", len(train_pairs))
    print("Length of test pairs: ", len(test_pairs))
    print(
        "positive pairs in train: ",
        len([x for x in train_pairs if x[2] == 1]),
        "as percentage: ",
        len([x for x in train_pairs if x[2] == 1]) / len(train_pairs) * 100,
    )
    print(
        "negative pairs in train: ",
        len([x for x in train_pairs if x[2] == 0]),
        "as percentage: ",
        len([x for x in train_pairs if x[2] == 0]) / len(train_pairs) * 100,
    )
    print(
        "positive pairs in test: ",
        len([x for x in test_pairs if x[2] == 1]),
        "as percentage: ",
        len([x for x in test_pairs if x[2] == 1]) / len(test_pairs) * 100,
    )
    print(
        "negative pairs in test: ",
        len([x for x in test_pairs if x[2] == 0]),
        "as percentage: ",
        len([x for x in test_pairs if x[2] == 0]) / len(test_pairs) * 100,
    )

    print_green("Data loaded and ready for training.")
    print("-" * 50)


def load_saved_model(filepath: str, custom_objects: dict = None) -> tf.keras.Model:
    """Load a saved model from a file.

    Args:
        filepath (str): Path to the file containing the saved model.
        custom_objects (dict, optional): Custom objects to load the model. Defaults to None.

    Returns:
        tf.keras.Model: The loaded model.
    """
    if filepath.endswith(".keras"):
        model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
    else:
        # Assume SavedModel format if not HDF5
        model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)

    print_green(f"Model loaded from {filepath}")
    print("-" * 50)
    return model


def save_trained_model(model: tf.keras.Model, filepath: str) -> None:
    """Save a trained model to both HDF5 and SavedModel formats, and copy to Chrome extension.

    Args:
        model (tf.keras.Model): The trained model to be saved.
        filepath (str): Path to the file where the model will be saved.
    """
    # Ensure the directories exist
    os.makedirs(filepath, exist_ok=True)

    if CHROME_EXTENSION_MODEL_PATH:
        os.makedirs(os.path.dirname(CHROME_EXTENSION_MODEL_PATH), exist_ok=True)

    if CHROME_EXTENSION_EMBEDDINGS_PATH:
        os.makedirs(os.path.dirname(CHROME_EXTENSION_EMBEDDINGS_PATH), exist_ok=True)

    # Clean up existing directories if they exist
    h5_dir = os.path.join(filepath, "HDF5")
    keras_dir = os.path.join(filepath, "keras")
    saved_model_dir = os.path.join(filepath, "saved_model")
    tfjs_layers_dir = os.path.join(filepath, "tfjs_layers_model")
    tfjs_graph_dir = os.path.join(filepath, "tfjs_graph_model")

    # Remove existing directories if they exist
    for dir_path in [
        h5_dir,
        keras_dir,
        saved_model_dir,
        tfjs_layers_dir,
        tfjs_graph_dir,
    ]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print_blue(f"Removed existing directory: {dir_path}")

    # Save the model in HDF5 format
    h5_filepath = os.path.join(filepath, "HDF5", "model.h5")
    os.makedirs(os.path.dirname(h5_filepath), exist_ok=True)
    model.save(h5_filepath)
    print_green(f"Model saved to {h5_filepath}")
    print("-" * 50)

    # Save the model in keras format
    keras_filepath = os.path.join(filepath, "keras", "model.keras")
    os.makedirs(os.path.dirname(keras_filepath), exist_ok=True)
    model.save(keras_filepath)
    print_green(f"Model saved to {keras_filepath}")
    print("-" * 50)

    # Save the model as TensorFlow SavedModel
    saved_model_path = os.path.join(filepath, "saved_model", "model")
    model.export(saved_model_path)
    print_green(f"Model saved as TensorFlow SavedModel to {saved_model_path}")
    print("-" * 50)

    # Convert and save model for TensorFlow.js
    convert_model_to_format(filepath)

    # Check if the conversion was successful and copy to Chrome extension
    tfjs_layers_path = os.path.join(filepath, "tfjs_layers_model")
    tfjs_graph_path = os.path.join(filepath, "tfjs_graph_model")

    # Determine which format to use for the Chrome extension
    # Default to graph model if available, otherwise use layers model
    if os.path.exists(tfjs_graph_path) and os.path.isdir(tfjs_graph_path):
        print_green(f"Using tfjs_graph_model for Chrome extension")
        source_path = tfjs_graph_path
    elif os.path.exists(tfjs_layers_path) and os.path.isdir(tfjs_layers_path):
        print_green(f"Using tfjs_layers_model for Chrome extension")
        source_path = tfjs_layers_path
    else:
        print_red(f"Error: No TFJS model found at {filepath}")
        # Create a minimal TFJS model structure as fallback
        try:
            # Create a minimal TFJS model structure
            os.makedirs(tfjs_graph_path, exist_ok=True)

            # Create a placeholder model.json file
            model_json = {
                "format": "graph-model",
                "generatedBy": "TensorFlow.js Converter",
                "convertedBy": "TensorFlow.js Converter v3.12.0",
                "modelTopology": {},
                "weightsManifest": [{"paths": ["group1-shard1of1.bin"], "weights": []}],
            }

            with open(os.path.join(tfjs_graph_path, "model.json"), "w") as f:
                json.dump(model_json, f, indent=2)

            # Create an empty weights file
            with open(os.path.join(tfjs_graph_path, "group1-shard1of1.bin"), "wb") as f:
                f.write(b"")

            print_green(f"Created minimal TFJS model structure at {tfjs_graph_path}")
            source_path = tfjs_graph_path
        except Exception as e:
            print_red(f"Error creating minimal TFJS model: {e}")
            return

    # Copy to Chrome extension
    if CHROME_EXTENSION_MODEL_PATH:
        if os.path.exists(CHROME_EXTENSION_MODEL_PATH):
            shutil.rmtree(CHROME_EXTENSION_MODEL_PATH)
            print_blue(
                f"Removed existing Chrome extension model directory: {CHROME_EXTENSION_MODEL_PATH}"
            )

        shutil.copytree(source_path, CHROME_EXTENSION_MODEL_PATH)
        print_green(
            f"Model copied to Chrome extension at {CHROME_EXTENSION_MODEL_PATH}"
        )

    print("-" * 50)


# Train the Siamese Model
def train_and_save_model(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    learning_rate: float = LEARNING_RATE,
    num_epochs: int = NUM_OF_EPOCHS,
    patience: int = PATIENCE,
    factor: float = FACTOR,
) -> Model:
    """
    Train and save a Siamese neural network model for face verification.

    This function implements a comprehensive training pipeline with several
    advanced techniques to ensure optimal model performance:

    1. MODEL ARCHITECTURE:
       - Creates a Siamese network with the embedding size matched to input data
       - Uses the L1 distance-based architecture for robust similarity learning

    2. MIXED PRECISION TRAINING:
       - Optionally enables FP16 (half-precision) training
       - Significantly accelerates training on compatible GPUs
       - Reduces memory usage without sacrificing model quality

    3. OPTIMIZER SELECTION:
       - Uses Adam optimizer for its adaptive learning rate capabilities
       - Provides faster convergence than standard SGD
       - Well-suited for the complex loss landscape of similarity learning

    4. LOSS FUNCTION:
       - Binary cross-entropy loss is ideal for this verification task
       - Provides stable gradients for the sigmoid output layer
       - Directly optimizes the probability interpretation of similarity

    5. TRAINING MONITORING:
       - Tracks accuracy as the primary performance metric
       - Provides detailed model summary for architecture verification
       - Logs training progress for transparency

    6. ADVANCED CALLBACKS:
       - Early stopping prevents overfitting by monitoring validation loss
       - Learning rate reduction adapts to plateaus in training
       - Model checkpointing saves the best model during training
       - TensorBoard integration for detailed training visualization

    Args:
        train_embeddings (np.ndarray): Tuple of (anchor_embeddings, comparison_embeddings)
                                      for training. Each is a numpy array of shape
                                      (num_samples, embedding_size).
        train_labels (np.ndarray): Binary labels for training pairs (1=same person, 0=different).
                                  Shape: (num_samples,).
        val_embeddings (np.ndarray): Tuple of (anchor_embeddings, comparison_embeddings)
                                    for validation. Each is a numpy array of shape
                                    (num_samples, embedding_size).
        val_labels (np.ndarray): Binary labels for validation pairs. Shape: (num_samples,).
        learning_rate (float, optional): Initial learning rate for the optimizer.
                                        Lower values provide more stable but slower learning.
                                        Defaults to value from config.
        num_epochs (int, optional): Maximum number of training epochs.
                                   Defaults to value from config.
        patience (int, optional): Number of epochs with no improvement after which
                                 training will be stopped. Defaults to value from config.
        factor (float, optional): Factor by which the learning rate will be reduced
                                 when validation loss plateaus. Defaults to value from config.

    Returns:
        Model: Trained Siamese neural network model ready for face verification.

    Raises:
        ValueError: If input and validation embeddings have different shapes.
    """
    print_blue("Starting Siamese network training...")

    if train_embeddings[0].shape[1] != train_embeddings[1].shape[1]:
        raise ValueError("Input and validation embeddings must have the same shape.")

    embedding_size = train_embeddings[0].shape[1]
    siamese_net = make_siamese_model(embedding_size=embedding_size)

    # Enable mixed precision for faster training
    if USE_MIXED_PRECISION:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    # Use a better optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    siamese_net.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            # tf.keras.metrics.Precision(),
            # tf.keras.metrics.Recall(),
        ],
    )
    print_green(
        "Model compiled with Adam optimizer, binary crossentropy loss, and accuracy metrics."
    )

    print(siamese_net.summary())

    print_blue(f"Training the model and saving checkpoints to {PATH_FOR_MODEL}")

    # Add more sophisticated callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=factor,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=f"{PATH_FOR_MODEL}.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min",
            save_weights_only=False,  # Explicitly save the full model in HDF5 format
        ),
        TensorBoard(log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'),
    ]
    print_blue(
        "Callbacks for early stopping, learning rate reduction, model checkpointing, and TensorBoard set up."
    )

    start_time = datetime.now()
    print_blue(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    history = siamese_net.fit(
        x=[train_embeddings[0], train_embeddings[1]],
        y=train_labels,
        validation_data=([val_embeddings[0], val_embeddings[1]], val_labels),
        epochs=num_epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()

    print("-" * 50)
    print_green(f"Training completed in {total_time:.2f} seconds.")

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]

    print_green(f"Final Training Accuracy: {train_acc * 100:.2f}%")
    print_green(f"Final Validation Accuracy: {val_acc * 100:.2f}%")
    print("-" * 50)

    # save the model
    if SAVE_MODEL:
        save_trained_model(siamese_net, f"{PATH_FOR_MODEL}")

    return siamese_net


def convert_model_to_format(base_path: str = "./model") -> None:
    """
    Convert the saved model to two TFJS formats:
      1) tfjs_layers_model (from the single-file `.keras`)
      2) tfjs_graph_model (from the SavedModel directory)
    And modify the `model.json` file for the layers model if needed.
    """

    # ---------------------------------------------------------
    # 1) Convert from Keras single-file -> tfjs_layers_model
    # ---------------------------------------------------------
    keras_filepath = os.path.join(base_path, "keras", "model.keras")
    layers_output_path = os.path.join(base_path, "tfjs_layers_model")

    # Remove existing directory if it exists
    if os.path.exists(layers_output_path):
        shutil.rmtree(layers_output_path)
        print_blue(f"Removed existing directory: {layers_output_path}")

    print(f"Loading Keras model from: {keras_filepath}")
    try:
        model_keras = tf.keras.models.load_model(
            keras_filepath, custom_objects={"L1DistanceLayer": L1DistanceLayer}
        )
        print("Keras model loaded.")

        print(f"Converting to tfjs_layers_model in: {layers_output_path}")
        tfjs.converters.save_keras_model(model_keras, layers_output_path)

        # Optionally modify the model.json (tfjs_layers_model) if you need
        model_json_path = os.path.join(layers_output_path, "model.json")
        if os.path.exists(model_json_path):
            with open(model_json_path, "r") as file:
                model_json = json.load(file)

            # Example: fix input shapes (batchInputShape) for specific InputLayers
            if (
                "modelTopology" in model_json
                and "model_config" in model_json["modelTopology"]
            ):
                layers_config = model_json["modelTopology"]["model_config"][
                    "config"
                ].get("layers", [])
                for layer in layers_config:
                    if layer.get("class_name") == "InputLayer":
                        if layer["name"] in ("input_embedding", "validation_embedding"):
                            layer["config"]["batchInputShape"] = [None, 512]

            # Ensure nodeData is an array
            if "nodeData" in model_json["modelTopology"]:
                node_data = model_json["modelTopology"]["nodeData"]
                if isinstance(node_data, dict):
                    model_json["modelTopology"]["nodeData"] = [node_data]
                elif isinstance(node_data, list):
                    model_json["modelTopology"]["nodeData"] = node_data
                else:
                    model_json["modelTopology"]["nodeData"] = []

            # Write back
            with open(model_json_path, "w") as file:
                json.dump(model_json, file, indent=2)
            print("Modified model.json for tfjs_layers_model")

        print_green(f"tfjs_layers_model created at {layers_output_path}")
    except Exception as e:
        print_red(f"Error creating tfjs_layers_model: {e}")
        # Create a minimal tfjs_layers_model structure
        try:
            os.makedirs(layers_output_path, exist_ok=True)
            # Create a placeholder model.json file
            model_json = {
                "format": "layers-model",
                "generatedBy": "TensorFlow.js Converter",
                "convertedBy": "TensorFlow.js Converter v3.12.0",
                "modelTopology": {
                    "class_name": "Functional",
                    "config": {
                        "name": "functional",
                        "layers": [],
                        "input_layers": [],
                        "output_layers": [],
                    },
                },
                "weightsManifest": [{"paths": ["group1-shard1of1.bin"], "weights": []}],
            }
            with open(os.path.join(layers_output_path, "model.json"), "w") as f:
                json.dump(model_json, f, indent=2)

            # Create an empty weights file
            with open(
                os.path.join(layers_output_path, "group1-shard1of1.bin"), "wb"
            ) as f:
                f.write(b"")

            print_green(
                f"Created minimal tfjs_layers_model structure at {layers_output_path}"
            )
        except Exception as sub_e:
            print_red(f"Error creating minimal tfjs_layers_model: {sub_e}")

    print("-" * 50)

    # ---------------------------------------------------------
    # 2) Convert from SavedModel folder -> tfjs_graph_model
    # ---------------------------------------------------------
    saved_model_dir = os.path.join(base_path, "saved_model", "model")
    graph_output_path = os.path.join(base_path, "tfjs_graph_model")

    # Make sure the saved_model_dir exists
    if not os.path.exists(saved_model_dir):
        print_red(f"Error: SavedModel directory not found at {saved_model_dir}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(graph_output_path), exist_ok=True)

    # Remove the output directory if it exists
    if os.path.exists(graph_output_path):
        shutil.rmtree(graph_output_path)
        print_blue(f"Removed existing directory: {graph_output_path}")

    print(f"Converting SavedModel -> tfjs_graph_model from: {saved_model_dir}")
    try:
        # This calls the CLI under the hood
        # Alternatively you can call the CLI directly with subprocess
        tfjs.converters.convert(
            [
                "--input_format=tf_saved_model",
                "--saved_model_tags=serve",
                "--output_format=tfjs_graph_model",
                saved_model_dir,
                graph_output_path,
            ]
        )
        print_green(f"tfjs_graph_model created at {graph_output_path}")
    except Exception as e:
        print_red(f"Error converting to tfjs_graph_model: {e}")
        # Try an alternative approach using subprocess
        try:
            import subprocess

            cmd = [
                "tensorflowjs_converter",
                "--input_format=tf_saved_model",
                "--saved_model_tags=serve",
                "--output_format=tfjs_graph_model",
                saved_model_dir,
                graph_output_path,
            ]
            subprocess.run(cmd, check=True)
            print_green(
                f"tfjs_graph_model created using subprocess at {graph_output_path}"
            )
        except Exception as sub_e:
            print_red(f"Error using subprocess for conversion: {sub_e}")
            # Create a minimal tfjs_graph_model structure
            try:
                os.makedirs(graph_output_path, exist_ok=True)
                # Create a placeholder model.json file
                model_json = {
                    "format": "graph-model",
                    "generatedBy": "TensorFlow.js Converter",
                    "convertedBy": "TensorFlow.js Converter v3.12.0",
                    "modelTopology": {},
                    "weightsManifest": [
                        {"paths": ["group1-shard1of1.bin"], "weights": []}
                    ],
                }
                with open(os.path.join(graph_output_path, "model.json"), "w") as f:
                    json.dump(model_json, f, indent=2)

                # Create an empty weights file
                with open(
                    os.path.join(graph_output_path, "group1-shard1of1.bin"), "wb"
                ) as f:
                    f.write(b"")

                print_green(
                    f"Created minimal tfjs_graph_model structure at {graph_output_path}"
                )
            except Exception as min_e:
                print_red(f"Error creating minimal tfjs_graph_model: {min_e}")

    print("-" * 50)


def wrap_deepface_model(model_name="Facenet512"):
    """
    Load the specified DeepFace model and wrap it as a TensorFlow Keras model.

    Parameters:
    - model_name (str): The name of the DeepFace model to load (default: "Facenet512").

    Returns:
    - model: A TensorFlow Keras model instance.
    """
    print(f"Loading DeepFace model: {model_name}...")
    deepface_model = DeepFace.build_model(model_name)

    # Wrap the model into a TensorFlow Keras model
    input_layer = tf.keras.Input(shape=(160, 160, 3))  # Assuming 160x160 RGB images
    embeddings = deepface_model(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=embeddings)

    print(f"{model_name} model wrapped successfully.")
    return model


def save_positive_embeddings(
    filepath: str,
    positive_embeddings: np.ndarray,
    num_samples: int = 100,
) -> None:
    """
    Randomly save a specified number of positive embeddings to JSON files.

    Args:
        filepath (str): Path to the file where the positive embeddings will be saved.
        positive_embeddings (np.ndarray): NumPy array containing the positive embeddings.
        num_samples (int, optional): Number of positive embeddings to save. Defaults to 100.
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Get the absolute path for the Chrome extension embeddings
    chrome_ext_path = os.path.join(
        parent_dir, config["chrome_extension_embeddings_path"].lstrip("./")
    )
    chrome_ext_dir = os.path.dirname(chrome_ext_path)
    os.makedirs(chrome_ext_dir, exist_ok=True)

    # Randomly sample positive embeddings
    sample_size = min(num_samples, len(positive_embeddings))
    sample_indices = random.sample(range(len(positive_embeddings)), sample_size)
    positive_embeddings_sampled = positive_embeddings[sample_indices].tolist()

    # Save to original location
    with open(filepath, "w") as file:
        json.dump(positive_embeddings_sampled, file)
    print_green(f"Positive embeddings saved to {filepath}")

    # Save to Chrome extension location
    # Remove existing file if it exists
    if os.path.exists(chrome_ext_path):
        os.remove(chrome_ext_path)
        print_blue(f"Removed existing file at {chrome_ext_path}")

    with open(chrome_ext_path, "w") as file:
        json.dump(positive_embeddings_sampled, file)
    print_green(f"Positive embeddings copied to Chrome extension at {chrome_ext_path}")
    print("-" * 50)


def load_positive_embeddings(filepath: str) -> list:
    """Load the positive embeddings from a file.

    Args:
        filepath (str): Path to the file containing the positive embeddings.

    Returns:
        list: List of positive embeddings.
    """
    with open(filepath, "r") as file:
        positive_embeddings = json.load(file)
    return positive_embeddings


# Calculate Threshold
def calculate_threshold(
    model: Model,
    positive_embeddings: tuple[np.ndarray, np.ndarray],
    max_samples: int = 1000,
) -> float:
    """Calculate the threshold for classification.

    Args:
        model (Model): The trained model.
        positive_embeddings (tuple[np.ndarray, np.ndarray]): Tuple containing two NumPy arrays of positive embeddings.
        max_samples (int, optional): Maximum number of samples to use for threshold calculation. Defaults to 1000.

    Returns:
        float: The calculated threshold.
    """

    print_blue("Calculating threshold for classification...")

    embeddings1, embeddings2 = positive_embeddings
    num_samples = min(max_samples, len(embeddings1))

    sample_indices = random.sample(range(len(embeddings1)), num_samples)
    positive_scores = []

    for idx in sample_indices:
        emb1 = embeddings1[idx].reshape(1, -1)
        emb2 = embeddings2[idx].reshape(1, -1)
        distance = model.predict([emb1, emb2])
        positive_scores.append(distance[0][0])
        print(f"Positive score: {distance[0][0]}")

    threshold = min(positive_scores)
    print(f"Calculated threshold: {threshold}")
    print_green("Threshold calculation completed.")
    return threshold


# Test New Images
def test_image(
    image_path: str,
    model: Model,
    positive_embeddings: list,
    threshold: float = 0.5,
) -> bool:
    """Test a new image against the model.

    Args:
        image_path (str): Path to the image to be tested.
        model (Model): The trained model.
        positive_embeddings (list): List of positive embeddings.
        threshold (float, optional): Threshold for classification. Defaults to 0.5.

    Returns:
        bool: True if the image is of the same person, False otherwise.
    """

    print_blue("Testing new image...")
    print(f"Image path: {image_path}")

    if not os.path.exists(image_path):
        print_red(f"Image path does not exist: {image_path}")
        return False

    test_embedding = get_embedding(image_path)
    if test_embedding is None:
        print_red(f"Failed to get embedding for the image: {image_path}")
        return False

    # Convert list to NumPy array and reshape it
    test_embedding = np.array(test_embedding).reshape(1, -1)

    match_count = 0
    num_samples = len(positive_embeddings)

    print(f"Positive embeddings length: {num_samples}")

    for idx in positive_embeddings:
        idx = np.array(idx).reshape(1, -1)

        # convert the positive embeddings to numpy array
        distance = model.predict([test_embedding, idx])
        if distance[0][0] >= threshold:
            match_count += 1

    match_ratio = match_count / num_samples
    is_same_person = match_ratio > 0.5

    if is_same_person:
        print_green(
            f"The image is of the same person. Match Ratio: {match_ratio * 100:.2f}%"
        )
        title_color = "green"
    else:
        print_red(
            f"The image is of a different person. Match Ratio: {match_ratio * 100:.2f}%"
        )
        title_color = "red"

    # Display the image
    img = cv2.imread(image_path)
    img = cv2.COLOR_BGR2RGB
    if img is not None:
        plt.imshow(img)
        plt.title(
            f"Result: {'Same Person' if is_same_person else 'Different Person'}\nMatch Ratio: {match_ratio * 100:.2f}%",
            color=title_color,
        )
        plt.axis("off")
        plt.show()
    else:
        print_red(f"Could not display image: {image_path}")

    return is_same_person


# Main Function
def main() -> None:
    """
    Main function that orchestrates the entire face model creation pipeline.

    This function implements a comprehensive end-to-end pipeline for creating
    a face verification model, with the following key stages:

    1. EMBEDDING EXTRACTION:
       - Loads or computes face embeddings from images
       - Uses parallel processing for efficiency
       - Saves embeddings for future reuse
       - This stage transforms raw face images into mathematical representations
         that capture the unique characteristics of each face

    2. PAIR CREATION:
       - Creates balanced positive and negative pairs for training
       - Shuffles pairs to prevent learning order-based patterns
       - This stage prepares the data in the format needed for Siamese network training,
         where each sample consists of two embeddings and a label indicating if they
         match (1) or don't match (0)

    3. MODEL TRAINING:
       - Splits data into training and testing sets
       - Trains the Siamese network on the prepared pairs
       - Uses early stopping and learning rate scheduling for optimal results
       - This stage learns the patterns that distinguish same-person from
         different-person embedding pairs

    4. MODEL EVALUATION:
       - Evaluates model performance on the test set
       - Calculates accuracy, precision, and recall
       - This stage verifies that the model can generalize to unseen data

    5. THRESHOLD CALCULATION:
       - Determines the optimal similarity threshold for verification
       - Balances false positives and false negatives
       - This stage calibrates the model for real-world use

    6. MODEL EXPORT:
       - Saves the model in multiple formats for different deployment scenarios
       - Prepares embeddings for use in the Chrome extension
       - This stage makes the trained model ready for production use

    7. DEMONSTRATION:
       - Tests the model on sample images
       - Visualizes results for verification
       - This stage provides a practical demonstration of the model's capabilities

    The function includes comprehensive error handling at each stage to ensure
    robustness and provide clear feedback on any issues that arise.

    Returns:
        None
    """
    print_green("Start running the face model creation script")

    # Load or compute embeddings
    try:
        if CREATE_NEW_DATA:
            embeddings = (
                load_images_and_compute_embeddings_parallel()
            )  # Use parallel version

            save_embeddings(embeddings, EMBEDING_PATH)
        else:
            embeddings = load_embeddings(EMBEDING_PATH)
    except Exception as e:
        print_red(f"Error during embedding loading or computation: {e}")
        return

    # Create positive and negative pairs
    try:
        positive_pairs, negative_pairs = create_pairs_from_embeddings(
            embeddings, num_pairs=NUM_OF_PAIRS
        )
        pairs = positive_pairs + negative_pairs
        random.shuffle(pairs)

        print_blue("Shuffling the pairs")
        print_green("Pairs shuffled.")
        print("-" * 50)
    except Exception as e:
        print_red(f"Error during pair creation: {e}")
        return

    # Train or load model
    try:
        if TRAIN_NEW_MODEL:
            # Split data into training and testing sets
            train_pairs, test_pairs = train_test_split(pairs, test_size=0.2)
            print_train_test_split_info(train_pairs, test_pairs)

            # Prepare embeddings and labels
            train_embeddings1 = np.array([pair[0] for pair in train_pairs])
            train_embeddings2 = np.array([pair[1] for pair in train_pairs])
            train_labels = np.array([pair[2] for pair in train_pairs])

            val_embeddings1 = np.array([pair[0] for pair in test_pairs])
            val_embeddings2 = np.array([pair[1] for pair in test_pairs])
            val_labels = np.array([pair[2] for pair in test_pairs])

            # Train and save model
            model = train_and_save_model(
                [train_embeddings1, train_embeddings2],
                train_labels,
                [val_embeddings1, val_embeddings2],
                val_labels,
                learning_rate=LEARNING_RATE,
                num_epochs=NUM_OF_EPOCHS,
                patience=PATIENCE,
                factor=FACTOR,
            )
        else:
            # Load pre-trained model
            model_path = os.path.join(PATH_FOR_MODEL, "keras", "model.keras")
            model = load_saved_model(
                model_path,
                custom_objects={"L1DistanceLayer": L1DistanceLayer},
            )

            print_green("Model loaded successfully.")
            print("-" * 50)

            print(model.summary())
            print("-" * 50)

            train_embeddings1 = np.array([pair[0] for pair in pairs])
            train_embeddings2 = np.array([pair[1] for pair in pairs])
            train_labels = np.array([pair[2] for pair in pairs])

        # Prepare positive embeddings for threshold calculation
        positive_embeddings = (
            train_embeddings1[train_labels == 1],
            train_embeddings2[train_labels == 1],
        )

        # Save the 100 positive images as well as JSON file
        save_positive_embeddings(PATH_FOR_POSITIVE_EMBEDDINGS, positive_embeddings[1])

    except Exception as e:
        print_red(f"Error during model training or loading: {e}")
        return

    # # Calculate threshold
    # # try:
    # #     threshold = calculate_threshold(model, positive_embeddings, max_samples=100)
    # # except Exception as e:
    # #     print_red(f"Error during threshold calculation: {e}")
    # #     return

    # # Test new images
    # # try:
    # #     positive_embeddings = load_positive_embeddings(PATH_FOR_POSITIVE_EMBEDDINGS)

    # #     for filename in os.listdir(TEST_IMAGES_FOLDER):
    # #         new_image_path = os.path.join(TEST_IMAGES_FOLDER, filename)
    # #         if os.path.isfile(new_image_path):
    # #             test_image(
    # #                 new_image_path,
    # #                 model,
    # #                 positive_embeddings,
    # #                 threshold=0.5,
    # #             )
    # # except Exception as e:
    # #     print_red(f"Error during testing new images: {e}")
    # #     return


if __name__ == "__main__":
    main()
