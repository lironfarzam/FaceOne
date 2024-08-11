import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from facenet_pytorch import InceptionResnetV1
from scipy.spatial import Delaunay
from scipy.spatial.distance import cosine, braycurtis
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle

# MediaPipe import for face landmarks
import mediapipe as mp


SHOW_3D_PLOT = False
CREATE_NEW_DATA = True
TRAIN_NEW_MODEL = True
SAVE_MODEL = True
PATH_FOR_MODEL = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/Network_based_on_images/face_recognition_model.pth"
RUN_ON_GPU = True
POSITIVE_VECTORS_FILE = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/positive_vectors"
NEGATIVE_VECTORS_FILE = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/negative_vectors"
TEST_IMAGES_FOLDER = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/imgs/test_imgs"

# Paths
# POS_PATH = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/Live_portrait/NN_LiveProtrait/src/positiv_images"
# ANC_PATH = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/Live_portrait/NN_LiveProtrait/src/positiv_images"
# NEG_PATH = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/imgs/negatives"


POS_PATH = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/Network_based_on_images/positive_from_camera"
ANC_PATH = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/Network_based_on_images/positive_from_camera"
NEG_PATH = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/imgs/negatives"

NUM_OF_IMAGES_TO_PROCESS = 300
INDEX_OF_PROCESS = 1

LEARNING_RATE = 0.00001
NUM_OF_EPOCHS = 25
PATIENCE = 7
FACTOR = 0.1

BATCH_SIZE = 16
NUM_OF_WORKERS = 8

# Check for MPS (Metal Performance Shaders) for MacBook Pro with M3 chip
device = torch.device(
    "mps" if torch.backends.mps.is_available() and RUN_ON_GPU else "cpu"
)


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=(10, 10)
        )  # Output: (None, 91, 91, 64)
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=(7, 7)
        )  # Output: (None, 40, 40, 128)
        self.conv3 = nn.Conv2d(
            128, 128, kernel_size=(4, 4)
        )  # Output: (None, 17, 17, 128)
        self.conv4 = nn.Conv2d(
            128, 256, kernel_size=(4, 4)
        )  # Output: (None, 6, 6, 256)
        self.fc1 = nn.Linear(2 * 2 * 256, 4096)  # Corrected input size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2), stride=2)  # Output: (None, 45, 45, 64)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2), stride=2)  # Output: (None, 19, 19, 128)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2), stride=2)  # Output: (None, 8, 8, 128)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, (2, 2), stride=2)  # Output: (None, 2, 2, 256)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.sigmoid(self.fc1(x))  # Output: (None, 4096)
        return x


class L1Dist(nn.Module):
    def __init__(self):
        super(L1Dist, self).__init__()

    def forward(self, input_embedding, validation_embedding):
        return torch.abs(input_embedding - validation_embedding)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.l1_layer = L1Dist()
        self.fc1 = nn.Linear(4096, 1)

    def forward(self, input_image, validation_image):
        input_embedding = self.embedding_net(input_image)
        validation_embedding = self.embedding_net(validation_image)
        distances = self.l1_layer(input_embedding, validation_embedding)
        output = torch.sigmoid(self.fc1(distances))
        return output


class FaceDataset(Dataset):
    def __init__(self, image_pairs, labels, transform=None):
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image1_path, image2_path = self.image_pairs[idx]
        label = self.labels[idx]

        # Use the preprocess function to process each image
        image1 = preprocess(image1_path)
        image2 = preprocess(image2_path)

        # Apply any additional transformations if provided
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Convert the images and label to float32 tensors
        return (
            torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1),
            torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1),
            torch.tensor(label, dtype=torch.float32),
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


import cv2


def preprocess(image_path):
    """Preprocess the image before feeding it to the network."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at path: {image_path}")

    if image.shape[0] > 250 or image.shape[1] > 250:

        h, w = image.shape[:2]

        if h > w:
            start = (h - w) // 2
            image = image[start : start + w, :]

        elif w > h:
            start = (w - h) // 2
            image = image[:, start : start + h]

        image = cv2.resize(image, (250, 250))

    # else:
    #     cv2.imshow("image after crop", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     print("image after crop size: ", image.shape)

    # Resize the image to the desired size of 100x100
    image = cv2.resize(image, (100, 100))

    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to the range [0, 1]
    # image = image / 255.0

    return image


def to_tensor(data_pairs):
    images1 = [item[0] for item in data_pairs]
    images2 = [item[1] for item in data_pairs]
    labels = [item[2] for item in data_pairs]
    images1 = torch.tensor(images1, dtype=torch.float32).permute(0, 3, 1, 2)
    images2 = torch.tensor(images2, dtype=torch.float32).permute(0, 3, 1, 2)
    labels = torch.tensor(labels, dtype=torch.float32)
    return images1, images2, labels


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
    num_workers=NUM_OF_WORKERS,
):
    print_blue("Creating DataLoader objects...")

    # Create the custom dataset for training and testing
    train_dataset = FaceDataset(train_image_pairs, train_labels)
    test_dataset = FaceDataset(test_image_pairs, test_labels)

    # Create the DataLoader for training and testing
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print_green("DataLoader objects created.")
    print("-" * 50)
    return train_loader, test_loader


def load_model(path, device):
    embedding_net = EmbeddingNet()
    siamese_net = SiameseNet(embedding_net)
    siamese_net.load_state_dict(torch.load(path, map_location=device))
    siamese_net.to(device)
    print_green(f"Model loaded from {path}")
    return siamese_net


def train_and_save_model(
    train_loader,
    test_loader,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_OF_EPOCHS,
    patience=PATIENCE,
    factor=FACTOR,
):
    print_blue("Starting Siamese network training...")

    embedding_net = EmbeddingNet()
    siamese_net = SiameseNet(embedding_net)
    siamese_net.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(siamese_net.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience
    )

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    best_loss = float("inf")
    early_stopping_counter = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        siamese_net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images1, images2, labels) in enumerate(train_loader):
            print(
                f"\rProcessing: Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx+1}/{len(train_loader)}]",
                end="",
                flush=True,
            )

            images1, images2, labels = (
                images1.to(device, dtype=torch.float32),
                images2.to(device, dtype=torch.float32),
                labels.to(device, dtype=torch.float32),
            )

            optimizer.zero_grad()
            outputs = siamese_net(images1, images2)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                predictions = (outputs > 0.5).float()
                correct_train += (predictions == labels.unsqueeze(1)).sum().item()
                total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        siamese_net.eval()
        running_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images1, images2, labels in test_loader:
                images1, images2, labels = (
                    images1.to(device, dtype=torch.float32),
                    images2.to(device, dtype=torch.float32),
                    labels.to(device, dtype=torch.float32),
                )
                outputs = siamese_net(images1, images2)
                loss = criterion(outputs, labels.unsqueeze(1))
                running_loss += loss.item()

                predictions = (outputs > 0.5).float()
                correct_test += (predictions == labels.unsqueeze(1)).sum().item()
                total_test += labels.size(0)

        test_loss = running_loss / len(test_loader)
        test_losses.append(test_loss)
        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)

        scheduler.step(test_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        remaining_time = epoch_duration * (num_epochs - (epoch + 1))

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\rEpoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy*100:.2f}% | "
            f"Time: {epoch_duration:.2f} seconds | Time left: {remaining_time:.2f} seconds | "
            f"Learning Rate: {current_lr:.6f}"
        )

        if test_loss < best_loss:
            best_loss = test_loss
            early_stopping_counter = 0
            if SAVE_MODEL:
                torch.save(siamese_net.state_dict(), PATH_FOR_MODEL)
                print(f"\rModel improved and saved to {PATH_FOR_MODEL}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print_red("\rEarly stopping triggered. Stopping training.")
                break

    total_time = time.time() - start_time
    print_green(f"Training completed in {total_time:.2f} seconds.")

    completed_epochs = len(train_losses)
    plot_training_results(
        completed_epochs, train_losses, test_losses, train_accuracies, test_accuracies
    )

    return siamese_net


def calculate_threshold(model, positive_vectors, max_samples=100):
    """
    Calculate the threshold for classification. The threshold is the average score of positive samples
    by running the model on positive samples randomly selected from the positive vectors.

    Args:
        model (nn.Module): Trained Siamese network model.
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
            torch.tensor(preprocess(vec[0]), dtype=torch.float32)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .to(device),
            torch.tensor(preprocess(vec[1]), dtype=torch.float32)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .to(device),
        )
        for vec in positive_samples
    ]

    model.eval()
    with torch.no_grad():
        positive_scores = []
        for pair in positive_pairs:
            distance = model(pair[0], pair[1])
            positive_scores.append(distance.item())
            # print(f"Positive score: {distance.item()}")
        print("-" * 50)

    # min of the positive scores
    threshold = min(positive_scores)
    print(f"Calculated threshold: {threshold}")
    print_green("Threshold calculation completed.")
    return threshold


def test_image(image_path, model, positive_vectors, num_samples=100, threshold=0.5):
    """
    Test a new image using the trained Siamese network model.

    Args:
        image_path (str): Path to the test image.
        model (nn.Module): Trained Siamese network model.
        positive_vectors (list): List of positive vectors (paths to images).
        num_samples (int): Number of samples to compare against.
        threshold (float): Threshold for classification.
    """
    print_blue("Testing new image...")
    print(f"Image path: {image_path}")

    # Preprocess the test image
    test_image_processed = preprocess(image_path)
    test_image_embedding = (
        torch.tensor(test_image_processed, dtype=torch.float32)
        .unsqueeze(0)
        .permute(0, 3, 1, 2)
        .to(device)
    )

    # Select random positive samples
    positive_samples = random.sample(
        positive_vectors, min(num_samples, len(positive_vectors))
    )

    if not CREATE_NEW_DATA:
        positive_samples = [eval(x) for x in positive_samples]

    match_count = 0

    model.eval()
    with torch.no_grad():
        for pos_vec in positive_samples:
            # Preprocess the positive sample image paths
            pos_image_processed = preprocess(pos_vec[0])
            pos_embedding = (
                torch.tensor(pos_image_processed, dtype=torch.float32)
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            distance = model(test_image_embedding, pos_embedding)
            print(f"Distance: {distance.item()}")
            if distance.item() <= threshold:
                match_count += 1

    match_ratio = match_count / num_samples
    is_same_person = match_ratio > 0.5

    if is_same_person:
        print_green(f"The image is of the same person. {match_ratio * 100:.2f}%")
        title_color = "green"
    else:
        print_red(f"The image is of a different person. {match_ratio * 100:.2f}%")
        title_color = "red"

    # Display the image with the result
    plt.imshow(test_image_processed)
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

        model = train_and_save_model(train_loader, test_loader)
    else:
        model = load_model(PATH_FOR_MODEL, device)

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
                threshold=0.5,
            )


if __name__ == "__main__":
    print_green("Start running the main file")
    main()
