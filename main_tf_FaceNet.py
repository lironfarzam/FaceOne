from datetime import datetime
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from deepface import DeepFace
from sklearn.model_selection import train_test_split


SHOW_3D_PLOT = False
CREATE_NEW_DATA = True
TRAIN_NEW_MODEL = True
SAVE_MODEL = True
PATH_FOR_MODEL = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/Network_based_on_images/face_recognition_model.keras"
RUN_ON_GPU = True
POSITIVE_VECTORS_FILE = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/tf/positive_vectors"
NEGATIVE_VECTORS_FILE = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/tf/negative_vectors"
TEST_IMAGES_FOLDER = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/imgs/test_imgs"

ANC_PATH = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/imgs/anchors"
POS_PATH = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/imgs/positives"
NEG_PATH = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/imgs/negatives"


NUM_OF_IMAGES_TO_PROCESS = 1_000_000
NUM_OF_PAIRS = 1_000_000

LEARNING_RATE = 0.0001
NUM_OF_EPOCHS = 30
PATIENCE = 7
FACTOR = 0.1
BATCH_SIZE = 128

# Ensure GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Memory growth: {tf.config.experimental.get_memory_growth(gpu)}")


# Utility print functions
def print_green(text):
    """Print text in green color."""
    print(f"\033[92m{text}\033[0m")


def print_red(text):
    """Print text in red color."""
    print(f"\033[91m{text}\033[0m")


def print_blue(text):
    """Print text in blue color."""
    print(f"\033[94m{text}\033[0m")


# Siamese Network Definition
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.abs(input_embedding - validation_embedding)


def make_siamese_model(embedding_size=512):
    """Create a Siamese network model.

    Args:
        embedding_size (int, optional): Size of the embedding vector. Defaults to 512.

    Returns:
        Model: Siamese network model.
    """
    input_embedding = Input(shape=(embedding_size,), name="input_embedding")
    validation_embedding = Input(shape=(embedding_size,), name="validation_embedding")

    # L1 Distance layer
    L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    distance = L1_layer([input_embedding, validation_embedding])

    # Classification layer
    outputs = Dense(1, activation="sigmoid")(distance)

    model = Model(inputs=[input_embedding, validation_embedding], outputs=outputs)
    return model


# DeepFace Embedding Extraction
def get_embedding(image_path, model_name="Facenet"):
    """Get the embedding vector for the given image.

    Args:
        image_path (str): Path to the image file.
        model_name (str, optional): Name of the model to use for embedding extraction. Defaults to "Facenet".

    Returns:
        numpy.ndarray: Embedding vector for the image.
    """
    try:
        # Read and display the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pass the image array instead of the file path
        embedding = DeepFace.represent(
            img_path=img, model_name=model_name, enforce_detection=False
        )

        return embedding[0]["embedding"]
    except Exception as e:
        print_red(f"Error processing {image_path}: {e}")
        return None


# Load and Embed Images
def load_images_and_compute_embeddings():
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
            embedding = get_embedding(image_path)

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
def save_embeddings(embeddings, filepath):
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
def load_embeddings(filepath):
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


# Create Pairs from Embeddings
def create_pairs_from_embeddings(embeddings, num_pairs=NUM_OF_PAIRS):
    print_blue(f"Creating {num_pairs} pairs using precomputed embeddings...")

    anchor_embeddings = embeddings["anchor"]
    positive_embeddings = embeddings["positive"]
    negative_embeddings = embeddings["negative"]

    anchor_images = list(anchor_embeddings.keys())
    positive_images = list(positive_embeddings.keys())
    negative_images = list(negative_embeddings.keys())

    positive_pairs = []
    negative_pairs = []

    for _ in range(num_pairs):
        anchor_img = random.choice(anchor_images)
        positive_img = random.choice(positive_images)
        emb1 = anchor_embeddings[anchor_img]
        emb2 = positive_embeddings[positive_img]
        positive_pairs.append((emb1, emb2, 1))

    for _ in range(num_pairs):
        anchor_img = random.choice(anchor_images)
        negative_img = random.choice(negative_images)
        emb1 = anchor_embeddings[anchor_img]
        emb2 = negative_embeddings[negative_img]
        negative_pairs.append((emb1, emb2, 0))

    print_green(
        f"Created {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs."
    )
    return positive_pairs, negative_pairs


def print_train_test_split_info(train_pairs, test_pairs):

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


# Train the Siamese Network
def train_and_save_model(
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_OF_EPOCHS,
    patience=PATIENCE,
    factor=FACTOR,
):

    print_blue("Starting Siamese network training...")

    embedding_size = train_embeddings[0].shape[1]
    siamese_net = make_siamese_model(embedding_size=embedding_size)

    siamese_net.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print_green(
        "Model compiled with Adam optimizer, binary crossentropy loss, and accuracy metrics."
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )
    lr_reduction = ReduceLROnPlateau(
        monitor="val_loss",
        factor=factor,
        patience=patience // 2,
        min_lr=1e-6,
        verbose=1,
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        PATH_FOR_MODEL, monitor="val_loss", save_best_only=True, verbose=1, mode="min"
    )

    callbacks = [early_stopping, lr_reduction, model_checkpoint]
    print_blue(
        "Callbacks for early stopping, learning rate reduction, and model checkpointing set up."
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
    print_green(f"Training completed in {total_time:.2f} seconds.")

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    print_green(f"Final Training Accuracy: {train_acc * 100:.2f}%")
    print_green(f"Final Validation Accuracy: {val_acc * 100:.2f}%")

    return siamese_net


# Calculate Threshold
def calculate_threshold(model, positive_embeddings, max_samples=100):
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


import numpy as np


# Test New Images
def test_image(image_path, model, positive_embeddings, num_samples=100, threshold=0.5):
    print_blue("Testing new image...")
    print(f"Image path: {image_path}")

    test_embedding = get_embedding(image_path)
    if test_embedding is None:
        print_red(f"Failed to get embedding for the image: {image_path}")
        return False

    # Convert list to NumPy array and reshape it
    test_embedding = np.array(test_embedding).reshape(1, -1)

    num_positive_embeddings = len(positive_embeddings[0])
    sample_indices = random.sample(
        range(num_positive_embeddings), min(num_samples, num_positive_embeddings)
    )
    match_count = 0

    for idx in sample_indices:
        pos_embedding = positive_embeddings[0][idx].reshape(1, -1)
        distance = model.predict([test_embedding, pos_embedding])
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
def main():
    if CREATE_NEW_DATA:
        embeddings = load_images_and_compute_embeddings()
        save_embeddings(embeddings, "embeddings.pkl")
    else:
        embeddings = load_embeddings("embeddings.pkl")

    positive_pairs, negative_pairs = create_pairs_from_embeddings(embeddings)
    pairs = positive_pairs + negative_pairs
    random.shuffle(pairs)

    print_blue("Shuffling the pairs")
    print_green("Pairs shuffled.")
    print("-" * 50)

    if TRAIN_NEW_MODEL:
        train_pairs, test_pairs = train_test_split(pairs, test_size=0.2)
        print_train_test_split_info(train_pairs, test_pairs)

        train_embeddings1 = np.array([pair[0] for pair in train_pairs])
        train_embeddings2 = np.array([pair[1] for pair in train_pairs])
        train_labels = np.array([pair[2] for pair in train_pairs])

        val_embeddings1 = np.array([pair[0] for pair in test_pairs])
        val_embeddings2 = np.array([pair[1] for pair in test_pairs])
        val_labels = np.array([pair[2] for pair in test_pairs])

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
        model = load_model(PATH_FOR_MODEL)

    positive_embeddings = (
        train_embeddings1[train_labels == 1],
        train_embeddings2[train_labels == 1],
    )
    threshold = calculate_threshold(model, positive_embeddings, max_samples=100)

    for filename in os.listdir(TEST_IMAGES_FOLDER):
        new_image_path = os.path.join(TEST_IMAGES_FOLDER, filename)
        if os.path.isfile(new_image_path):
            test_image(
                new_image_path,
                model,
                positive_embeddings,
                num_samples=100,
                threshold=threshold,
            )


if __name__ == "__main__":
    print_green("Start running the main file")
    main()
