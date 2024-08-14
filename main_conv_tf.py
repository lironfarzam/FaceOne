# import os
# import time
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import cv2
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from facenet_pytorch import InceptionResnetV1
# from scipy.spatial import Delaunay
# from scipy.spatial.distance import cosine, braycurtis
# from scipy.spatial.transform import Rotation as R
# from sklearn.decomposition import PCA
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import OrderedDict
# from mpl_toolkits.mplot3d import Axes3D, art3d
# from matplotlib.patches import Rectangle

# # MediaPipe import for face landmarks
# import mediapipe as mp

# Import standard dependencies
from datetime import datetime
import time
import random
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


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


NUM_OF_IMAGES_TO_PROCESS = 1000
INDEX_OF_PROCESS = 1

LEARNING_RATE = 0.0001
NUM_OF_EPOCHS = 20
PATIENCE = 7
FACTOR = 0.1

BATCH_SIZE = 16
NUM_OF_WORKERS = 8

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print("Memory growth:", tf.config.experimental.get_memory_growth(gpu))


def make_embedding():
    inp = Input(shape=(100, 100, 3), name="input_image")

    # First block
    c1 = Conv2D(64, (10, 10), activation="relu")(inp)
    m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation="relu")(m1)
    m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation="relu")(m2)
    m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation="relu")(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation="sigmoid")(f1)

    return Model(inputs=[inp], outputs=[d1], name="embedding")


class L1Dist(Layer):
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        # Ensure inputs are tensors
        input_embedding = (
            input_embedding[0] if isinstance(input_embedding, list) else input_embedding
        )
        validation_embedding = (
            validation_embedding[0]
            if isinstance(validation_embedding, list)
            else validation_embedding
        )

        # Calculate the absolute difference
        return tf.abs(input_embedding - validation_embedding)


def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name="input_img", shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name="validation_img", shape=(100, 100, 3))

    # Create the embedding model
    embedding = make_embedding()

    # Get the embeddings for both input and validation images
    input_embedding = embedding(input_image)
    validation_embedding = embedding(validation_image)

    # Calculate the L1 Distance between the embeddings
    distances = L1Dist()(input_embedding, validation_embedding)

    # Flatten the distances to ensure a defined shape for the Dense layer
    flattened_distances = tf.keras.layers.Flatten()(distances)

    # Classification layer
    classifier = Dense(1, activation="sigmoid")(flattened_distances)

    return Model(
        inputs=[input_image, validation_image],
        outputs=classifier,
        name="SiameseNetwork",
    )


def print_green(text):
    """Print text in green color."""
    print(f"\033[92m{text}\033[0m")


def print_red(text):
    """Print text in red color."""
    print(f"\033[91m{text}\033[0m")


def print_blue(text):
    """Print text in blue color."""
    print(f"\033[94m{text}\033[0m")


def preprocess(file_path):
    # Read in the image using TensorFlow
    byte_img = tf.io.read_file(file_path)

    # Get the file extension
    file_extension = tf.strings.split(file_path, ".")[-1]
    file_extension = tf.strings.lower(file_extension)

    # Conditional decoding based on file extension
    def decode_jpeg():
        img = tf.io.decode_jpeg(byte_img)
        return tf.cast(img, tf.float32)

    def decode_png():
        img = tf.io.decode_png(byte_img)
        return tf.cast(img, tf.float32)

    def decode_bmp():
        img = tf.io.decode_bmp(byte_img)
        return tf.cast(img, tf.float32)

    def decode_gif():
        img = tf.io.decode_gif(byte_img)
        img = tf.squeeze(img, axis=0)  # Remove the frame dimension if GIF
        return tf.cast(img, tf.float32)

    def unsupported_format():
        # Raise an exception within the TensorFlow graph
        msg = tf.strings.format("Unsupported image format: {}", file_extension)
        # tf.print(msg)
        return tf.zeros([100, 100, 3], dtype=tf.float32)  # Placeholder return value

    img = tf.case(
        [
            (tf.equal(file_extension, "jpg"), decode_jpeg),
            (tf.equal(file_extension, "jpeg"), decode_jpeg),
            (tf.equal(file_extension, "png"), decode_png),
            (tf.equal(file_extension, "bmp"), decode_bmp),
            (tf.equal(file_extension, "gif"), decode_gif),
        ],
        default=unsupported_format,
        exclusive=True,
    )

    # Ensure the image has a valid shape
    img = tf.ensure_shape(img, [None, None, 3])

    # Check the image size
    original_size = tf.shape(img)[:2]
    # If the image is not 250x250, crop and resize it
    if original_size[0] != 250 or original_size[1] != 250:
        # Crop the center square
        min_dim = tf.reduce_min(original_size)
        start = (original_size - min_dim) // 2
        img = img[start[0] : start[0] + min_dim, start[1] : start[1] + min_dim]

    # Resize the image to 100x100x3
    img = tf.image.resize(img, (100, 100))

    # Scale the image to be between 0 and 1
    img = img / 255.0

    return img


def preprocess_twin(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label


def plot_training_results(
    epochs, train_losses, test_losses, train_accuracies, test_accuracies
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(range(epochs), train_losses, label="Train Loss")
    ax1.plot(range(epochs), test_losses, label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()

    ax2.plot(range(epochs), train_accuracies, label="Train Accuracy")
    ax2.plot(range(epochs), test_accuracies, label="Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.legend()

    plt.show()


def create_dataset():
    # Load all image names from the directories
    positive_images = os.listdir(POS_PATH)
    negative_images = os.listdir(NEG_PATH)
    anchor_images = os.listdir(ANC_PATH)

    print_blue("Total number of images available:")
    print(f"Positive images: {len(positive_images)}")
    print(f"Negative images: {len(negative_images)}")
    print(f"Anchor images: {len(anchor_images)}")
    print("-" * 50)

    if SHOW_3D_PLOT:
        show_images(positive_images[:5], negative_images[:5], anchor_images[:5])

    print_blue(f"Creating {NUM_OF_IMAGES_TO_PROCESS} random pairs for the dataset...")

    # Randomly pair positive images with anchors
    positive_pairs = [
        (
            os.path.join(POS_PATH, random.choice(positive_images)),
            os.path.join(ANC_PATH, random.choice(anchor_images)),
            1,
        )
        for _ in range(NUM_OF_IMAGES_TO_PROCESS)
    ]

    # Randomly pair negative images with anchors
    negative_pairs = [
        (
            os.path.join(NEG_PATH, random.choice(negative_images)),
            os.path.join(ANC_PATH, random.choice(anchor_images)),
            0,
        )
        for _ in range(NUM_OF_IMAGES_TO_PROCESS)
    ]

    print_green("Labels dataset created.")
    print(f"Number of positive pairs: {len(positive_pairs)}")
    print(f"Number of negative pairs: {len(negative_pairs)}")
    print("-" * 50)

    # Save the positive and negative pairs to a file
    save_pairs(positive_pairs, negative_pairs)

    return positive_pairs, negative_pairs


def show_images(positive_images, negative_images, anchor_images):
    positive_images_as_image = list(
        map(lambda x: preprocess(os.path.join(POS_PATH, x)), positive_images)
    )
    negative_images_as_image = list(
        map(lambda x: preprocess(os.path.join(NEG_PATH, x)), negative_images)
    )
    anchor_images_as_image = list(
        map(lambda x: preprocess(os.path.join(ANC_PATH, x)), anchor_images)
    )

    fig, ax = plt.subplots(3, 5, figsize=(10, 10))
    ax[0, 0].set_title("Positive images")
    ax[1, 0].set_title("Negative images")
    ax[2, 0].set_title("Anchor images")
    for i in range(5):
        ax[0, i].imshow(positive_images_as_image[i])
        ax[0, i].axis("off")
        ax[1, i].imshow(negative_images_as_image[i])
        ax[1, i].axis("off")
        ax[2, i].imshow(anchor_images_as_image[i])
        ax[2, i].axis("off")
    plt.show()


def save_pairs(positive_pairs, negative_pairs):
    with open(POSITIVE_VECTORS_FILE, "w") as f:
        for item in positive_pairs:
            f.write(f"{item}\n")
    with open(NEGATIVE_VECTORS_FILE, "w") as f:
        for item in negative_pairs:
            f.write(f"{item}\n")
    print_green("Positive and negative vectors saved.")
    print("-" * 50)


def load_pairs():
    print_blue("Loading the positive and negative vectors")
    with open(POSITIVE_VECTORS_FILE, "r") as f:
        positive_pairs = f.readlines()
    with open(NEGATIVE_VECTORS_FILE, "r") as f:
        negative_pairs = f.readlines()
    print_green("Positive and negative vectors loaded.")
    print("-" * 50)
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


def create_dataloaders(
    train_image_pairs,
    train_labels,
    test_image_pairs,
    test_labels,
    batch_size=BATCH_SIZE,
):
    print_blue("Creating DataLoader objects...")

    # Separate the image pairs into two lists: one for input images and one for validation images
    train_images_1 = [pair[0] for pair in train_image_pairs]
    train_images_2 = [pair[1] for pair in train_image_pairs]

    # Create the TensorFlow Dataset for training
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images_1, train_images_2, train_labels)
    )
    train_dataset = train_dataset.map(preprocess_twin)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.repeat()  # Ensure the dataset repeats indefinitely
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_images_1 = [pair[0] for pair in test_image_pairs]
    test_images_2 = [pair[1] for pair in test_image_pairs]

    # Create the TensorFlow Dataset for testing
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_images_1, test_images_2, test_labels)
    )
    test_dataset = test_dataset.map(preprocess_twin)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.repeat()  # Ensure the dataset repeats indefinitely
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print_green("DataLoader objects created.")
    print("-" * 50)
    return train_dataset, test_dataset

def load_model(path):
    """
    Load a pre-trained Siamese network model from the specified path.

    Args:
        path (str): Path to the saved model file.

    Returns:
        Model: Loaded Siamese network model.
    """
    # Load the model from the specified path
    siamese_net = tf.keras.models.load_model(path, custom_objects={'L1Dist': L1Dist})
    print_green(f"Model loaded from {path}")
    return siamese_net

def train_and_save_model(
    train_loader,
    test_loader,
    train_labels,
    test_labels,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_OF_EPOCHS,
    patience=PATIENCE,
    factor=FACTOR,
):
    print_blue("Starting Siamese network training...")

    siamese_net = make_siamese_model()

    siamese_net.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],  # Add accuracy as a metric
    )
    print_green("Model compiled with Adam optimizer, binary crossentropy loss, and accuracy metrics.")

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )
    lr_reduction = ReduceLROnPlateau(
        monitor="val_loss", factor=factor, patience=patience // 2, min_lr=1e-6, verbose=1
    )
    
    # Save the best model based on validation loss
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        PATH_FOR_MODEL, monitor='val_loss', save_best_only=True, verbose=1, mode='min'
    )

    callbacks = [early_stopping, lr_reduction, model_checkpoint]
    print_blue("Callbacks for early stopping, learning rate reduction, and model checkpointing set up.")

    start_time = datetime.now()
    print_blue(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    train_steps = len(train_labels) // BATCH_SIZE
    val_steps = len(test_labels) // BATCH_SIZE

    # Prepare the data for training
    def prepare_data(data_loader):
        for images1, images2, labels in data_loader:
            yield ({"input_img": images1, "validation_img": images2}, labels)

    train_data = prepare_data(train_loader)
    val_data = prepare_data(test_loader)

    history = siamese_net.fit(
        train_data,
        steps_per_epoch=train_steps,
        validation_data=val_data,
        validation_steps=val_steps,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=1,  # Show detailed output for each epoch
    )

    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print_green(f"Training completed in {total_time:.2f} seconds.")

    # Print final training and validation accuracy
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print_green(f"Final Training Accuracy: {train_acc * 100:.2f}%")
    print_green(f"Final Validation Accuracy: {val_acc * 100:.2f}%")

    return siamese_net

def calculate_threshold(model, positive_vectors, max_samples=100):
    """
    Calculate the threshold for classification. The threshold is the minimum score of positive samples
    by running the model on positive samples randomly selected from the positive vectors.

    Args:
        model (Model): Trained Siamese network model.
        positive_vectors (list): List of positive vectors (paths to images).
        max_samples (int): Maximum number of samples to consider.

    Returns:
        float: Calculated threshold.
    """
    print_blue("Calculating threshold for classification...")

    positive_samples = random.sample(
        positive_vectors, min(max_samples, len(positive_vectors))
    )
    if not CREATE_NEW_DATA:
        positive_samples = [eval(x) for x in positive_samples]

    positive_pairs = [
        (
            preprocess(vec[0])[None, ...],
            preprocess(vec[1])[None, ...],
        )
        for vec in positive_samples
    ]

    positive_scores = []
    for pair in positive_pairs:
        distance = model.predict(pair)
        positive_scores.append(distance[0][0])
        print(f"Positive score: {distance[0][0]}")

    # Threshold is the minimum of the positive scores
    threshold = min(positive_scores)
    print(f"Calculated threshold: {threshold}")
    print_green("Threshold calculation completed.")
    return threshold



def test_image(image_path, model, positive_vectors, num_samples=100, threshold=0.5):
    """
    Test a new image using the trained Siamese network model.

    Args:
        image_path (str): Path to the test image.
        model (Model): Trained Siamese network model.
        positive_vectors (list): List of positive vectors (paths to images).
        num_samples (int): Number of samples to compare against.
        threshold (float): Threshold for classification.
    """
    print_blue("Testing new image...")
    print(f"Image path: {image_path}")

    # Preprocess the test image
    test_image_processed = preprocess(image_path)[None, ...]

    # Select random positive samples
    positive_samples = random.sample(
        positive_vectors, min(num_samples, len(positive_vectors))
    )

    if not CREATE_NEW_DATA:
        positive_samples = [eval(x) for x in positive_samples]

    match_count = 0

    for pos_vec in positive_samples:
        # Preprocess the positive sample image paths
        pos_image_processed = preprocess(pos_vec[0])[None, ...]

        distance = model.predict([test_image_processed, pos_image_processed])
        if distance[0][0] <= threshold:
            match_count += 1

    match_ratio = match_count / num_samples
    is_same_person = match_ratio > 0.5

    if is_same_person:
        print_green(f"The image is of the same person. {match_ratio * 100:.2f}%")
        title_color = "green"
    else:
        print_red(f"The image is of a different person. {match_ratio * 100:.2f}%")
        title_color = "red"

    # Display the original image
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    plt.imshow(test_image_processed[0])

    plt.title(
        f"Result: {'Same Person' if is_same_person else 'Different Person'}  |  Match Ratio: {match_ratio * 100:.2f}%",
        color=title_color,
    )
    plt.axis("off")
    plt.show()

    return is_same_person


def main():
    if CREATE_NEW_DATA:
        positive_pairs, negative_pairs = create_dataset()
    else:
        positive_pairs, negative_pairs = load_pairs()

    pairs = positive_pairs + negative_pairs
    random.shuffle(pairs)

    print_blue("Shuffling the pairs")
    print_green("Pairs shuffled.")
    print("-" * 50)

    if TRAIN_NEW_MODEL:
        train_pairs, test_pairs = train_test_split(pairs, test_size=0.2)
        print_train_test_split_info(train_pairs, test_pairs)

        if CREATE_NEW_DATA:
            # Extract image paths and labels
            train_image_pairs = [(x[0], x[1]) for x in train_pairs]
            train_labels = [x[2] for x in train_pairs]
            test_image_pairs = [(x[0], x[1]) for x in test_pairs]
            test_labels = [x[2] for x in test_pairs]
        else:
            print_blue("Extracting image paths and labels...")
            # convert the string to tuple (img1, img2, label)
            train_pairs = [eval(x) for x in train_pairs]
            test_pairs = [eval(x) for x in test_pairs]

            # Extract image paths and labels directly from the tuple
            train_image_pairs = [(x[0], x[1]) for x in train_pairs]
            train_labels = [x[2] for x in train_pairs]
            test_image_pairs = [(x[0], x[1]) for x in test_pairs]
            test_labels = [x[2] for x in test_pairs]

        train_loader, test_loader = create_dataloaders(
            train_image_pairs,
            train_labels,
            test_image_pairs,
            test_labels,
        )
        # Test the dataloaders by printing the first batch
        for images1, images2, labels in train_loader:
            print("Training batch shapes:")
            print(f"Images1: {images1.shape}")
            print(f"Images2: {images2.shape}")
            print(f"Labels: {labels.shape}")
            break

        model = train_and_save_model(
            train_loader, test_loader, train_labels, test_labels
        )
    else:
        model = load_model(PATH_FOR_MODEL)

    # Calculate the threshold using positive pairs
    threshold = calculate_threshold(model, positive_pairs, max_samples=100)

    # Test the model on new images
    test_images_folder = TEST_IMAGES_FOLDER
    for filename in os.listdir(test_images_folder):
        new_image_path = os.path.join(test_images_folder, filename)
        if os.path.isfile(new_image_path):
            test_image(
                new_image_path,
                model,
                positive_pairs,
                num_samples=100,
                threshold=threshold,
            )


if __name__ == "__main__":
    print_green("Start running the main file")
    main()
