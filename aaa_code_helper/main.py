from datetime import datetime
import json
import time
import random
from typing import Callable
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


from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Layer,
    Dense,
    Input,
    Lambda,
    Dropout,
    Concatenate,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Reshape,
)
from tensorflow.keras.optimizers import Adam
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
from concurrent.futures import ProcessPoolExecutor

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


# Load configuration
def load_config(config_path: str = "./config.json") -> dict:
    """Load configuration from JSON file.

    Args:
        config_path (str, optional): Path to the configuration file. Defaults to "config.json".

    Returns:
        dict: Configuration dictionary.
    """

    with open(config_path, "r") as f:
        return json.load(f)


config = load_config()

pp.pprint(config)
print("-" * 50)


# Constants from config
# NUM_OF_WORKERS = config["num_of_workers"]
# SHOW_3D_PLOT = config["show_3d_plot"]
# RUN_ON_GPU = config["run_on_gpu"]

CREATE_NEW_DATA = config["create_new_data"]
TRAIN_NEW_MODEL = config["train_new_model"]
SAVE_MODEL = config["save_model"]
PATH_FOR_MODEL = config["path_for_model"]
NEGATIVE_VECTORS_FILE = config["negative_vectors_file"]
TEST_IMAGES_FOLDER = config["test_images_folder"]
ANC_PATH = config["anchors_folder"]
POS_PATH = config["positives_folder"]
NEG_PATH = config["negatives_folder"]
EMBEDING_PATH = config["embedding_path"]
PATH_FOR_POSITIVE_EMBEDDINGS = config["path_for_positives_embeddings"]

NUM_OF_IMAGES_TO_PROCESS = 1000
NUM_OF_PAIRS = 1_000_000
LEARNING_RATE = 0.00001
NUM_OF_EPOCHS = 10
PATIENCE = 7
FACTOR = 0.1
BATCH_SIZE = 128
WEIGHT_DECAY = 0.00001
DROPOUT_RATE = 0.3
USE_MIXED_PRECISION = False


# Ensure GPU Memory Growth
if tf.test.is_built_with_cuda():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth: {tf.config.experimental.get_memory_growth(gpu)}")


# Utility print functions
def print_green(text: str) -> None:
    """Print text in green color."""
    print(f"\033[92m{text}\033[0m")


def print_red(text: str) -> None:
    """Print text in red color."""
    print(f"\033[91m{text}\033[0m")


def print_blue(text: str) -> None:
    """Print text in blue color."""
    print(f"\033[94m{text}\033[0m")


# def log_function_call(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         print_blue(f"Starting {func.__name__}...")
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         print_green(f"Completed {func.__name__} in {end_time - start_time:.2f} seconds")
#         print("-" * 50)
#         return result

#     return wrapper


class L1DistanceLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute the L1 distance between two tensors.

    Args:
        inputs (list): List containing two tensors.

    Returns:
        tf.Tensor: L1 distance tensor.
    """

    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.abs(input_embedding - validation_embedding)


def make_siamese_model(embedding_size: int = 512) -> Model:
    """Create a Siamese network model.

    Args:
        embedding_size (int, optional): Size of the embedding vector. Defaults to 512.

    Returns:
        Model: Siamese network model.
    """
    if embedding_size <= 0:
        raise ValueError("Embedding size must be a positive integer.")

    # Define inputs
    input_embedding = Input(batch_shape=(None, 512), name="input_embedding")
    validation_embedding = Input(batch_shape=(None, 512), name="validation_embedding")

    # Concatenate embeddings and reshape
    # merged = Concatenate(axis=-1)([input_embedding, validation_embedding])  # (1024,)
    merged = L1DistanceLayer()([input_embedding, validation_embedding])
    # Fully connected layers
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

    # Classification Layers
    outputs = Dense(1, activation="sigmoid")(x)

    # Create model
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
    """Get the embedding vector for the given image.

    Args:
        image_path (str): Path to the image file.
        model_name (str, optional): Name of the model to use for embedding extraction. Defaults to "Facenet".
        apply_augmentation (bool, optional): Whether to apply augmentation to the image. Defaults to False.

    Returns:
        np.ndarray: Embedding vector for the image.
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
    """Create pairs of embeddings for training.

    Args:
        embeddings (dict): Dictionary containing the embeddings for anchor, positive, and negative images.
        num_pairs (int, optional): Maximum number of pairs to create. Defaults to NUM_OF_PAIRS.

    Returns:
        tuple: Two lists of tuples, each containing an embedding pair and a label.
    """
    print_blue(f"Creating {num_pairs} pairs using precomputed embeddings...")

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
    # Load config for Chrome extension path
    config = load_config()
    chrome_ext_path = config["chrome_extension_model_path"]
    chrome_ext_embeddings_path = config["chrome_extension_embeddings_path"]

    # Ensure the directories exist
    os.makedirs(filepath, exist_ok=True)
    os.makedirs(os.path.dirname(chrome_ext_path), exist_ok=True)
    os.makedirs(os.path.dirname(chrome_ext_embeddings_path), exist_ok=True)

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
    tfjs_path = os.path.join(filepath, "tfjs_graph_model")
    convert_model_to_format(filepath)
    print_green(f"Model converted to TensorFlow.js format at {tfjs_path}")
    print("-" * 50)

    # Copy TFJS model to Chrome extension directory
    if os.path.exists(tfjs_path):
        # Remove existing files in chrome extension directory if they exist
        if os.path.exists(chrome_ext_path):
            shutil.rmtree(chrome_ext_path)

        # Copy the new files
        shutil.copytree(tfjs_path, chrome_ext_path)
        print_green(f"Model copied to Chrome extension at {chrome_ext_path}")
        print("-" * 50)
    else:
        print_red(f"Error: TFJS model not found at {tfjs_path}")


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
        "Model compiled with AdamW optimizer, binary crossentropy loss, and accuracy metrics."
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

    print(f"Loading Keras model from: {keras_filepath}")
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
            layers_config = model_json["modelTopology"]["model_config"]["config"].get(
                "layers", []
            )
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
    print("-" * 50)

    # ---------------------------------------------------------
    # 2) Convert from SavedModel folder -> tfjs_graph_model
    # ---------------------------------------------------------
    saved_model_dir = os.path.join(base_path, "saved_model", "model")
    graph_output_path = os.path.join(base_path, "tfjs_graph_model")

    print(f"Converting SavedModel -> tfjs_graph_model from: {saved_model_dir}")
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
    config = load_config()
    chrome_ext_embeddings_path = config["chrome_extension_embeddings_path"]

    # Ensure directories exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    os.makedirs(os.path.dirname(chrome_ext_embeddings_path), exist_ok=True)

    sample_indices = random.sample(range(len(positive_embeddings)), num_samples)
    positive_embeddings_sampled = positive_embeddings[sample_indices].tolist()

    # Save to original location
    with open(filepath, "w") as file:
        json.dump(positive_embeddings_sampled, file)
    print_green(f"Positive embeddings saved to {filepath}")

    # Save to Chrome extension location
    with open(chrome_ext_embeddings_path, "w") as file:
        json.dump(positive_embeddings_sampled, file)
    print_green(
        f"Positive embeddings copied to Chrome extension at {chrome_ext_embeddings_path}"
    )
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
    print_green("Start running the main file")

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
        positive_pairs, negative_pairs = create_pairs_from_embeddings(embeddings)
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
            model = load_saved_model(
                f"{PATH_FOR_MODEL}.keras",
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

        # Convert the saved model to the desired format
        convert_model_to_format(f"{PATH_FOR_MODEL}")

        # save the 100 positive images as well as JSON file
        save_positive_embeddings(PATH_FOR_POSITIVE_EMBEDDINGS, positive_embeddings[1])

    except Exception as e:
        print_red(f"Error during model training or loading: {e}")
        return

    # # Calculate threshold
    # try:
    #     threshold = calculate_threshold(model, positive_embeddings, max_samples=100)
    # except Exception as e:
    #     print_red(f"Error during threshold calculation: {e}")
    #     return

    # # Test new images
    # try:
    #     positive_embeddings = load_positive_embeddings(PATH_FOR_POSITIVE_EMBEDDINGS)

    #     for filename in os.listdir(TEST_IMAGES_FOLDER):
    #         new_image_path = os.path.join(TEST_IMAGES_FOLDER, filename)
    #         if os.path.isfile(new_image_path):
    #             test_image(
    #                 new_image_path,
    #                 model,
    #                 positive_embeddings,
    #                 threshold=0.5,
    #             )
    # except Exception as e:
    #     print_red(f"Error during testing new images: {e}")
    #     return


if __name__ == "__main__":
    main()
