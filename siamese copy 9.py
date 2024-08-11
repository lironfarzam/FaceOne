import os
import time
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine, braycurtis
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1
import random


ROOT_FOLDER_PATH = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/imgs/"
SHOW_3D_PLOT = True
CREATE_NEW_DATA = False
PATH_FOR_DATA = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/Siamese_network/rotated_points.txt"
SAVE_MODEL = True
PATH_FOR_MODEL = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/Siamese_network/face_recognition_model.pth"
RUN_ON_GPU = True
POSITIVE_VECTORS_FILE = "positive_vectors.npy"
NEGATIVE_VECTORS_FILE = "negative_vectors.npy"
TEST_IMAGES_FOLDER = "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/imgs/test_imgs"

# Check for MPS (Metal Performance Shaders) for MacBook Pro with M3 chip
device = torch.device(
    "mps" if torch.backends.mps.is_available() and RUN_ON_GPU else "cpu"
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


class FaceLandmarksDataset(Dataset):
    """
    Dataset class for face landmarks pairs and labels.

    Args:
        pairs (list): List of tuples containing pairs of landmarks.
        labels (list): List of labels for each pair.
        transform (callable, optional): Optional transform to be applied on a sample.

    Returns:
        tuple: (pair of tensors, label tensor)
    """

    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        points1, points2 = self.pairs[idx]
        label = self.labels[idx]

        if self.transform:
            points1 = self.transform(points1)
            points2 = self.transform(points2)

        # Reshape points to include channel dimension for Conv3d (e.g., 1xDxHxW)
        points1 = points1.reshape(1, *points1.shape)
        points2 = points2.reshape(1, *points2.shape)

        points1 = torch.tensor(points1, dtype=torch.float32)
        points2 = torch.tensor(points2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return (points1, points2), label


class EmbeddingModel(nn.Module):
    """
    Embedding model using 2D convolutional layers followed by fully connected layers.
    Designed to work with 2D spatial data with an additional depth/channel dimension.

    Args:
        input_shape (tuple): Shape of the input data.
        embedding_dim (int): Dimension of the embedding features.

    Returns:
        Tensor: Output embedding of the input features.
    """

    def __init__(self, input_shape, embedding_dim=128):
        super(EmbeddingModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        # Dynamically calculate the output shape after the convolution layers
        conv_output_shape = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_output_shape, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, *shape
            )  # Create a zero tensor with the input shape
            output_feat = self.conv_layers(dummy_input)
            return int(np.prod(output_feat.size()))

    def forward(self, x):
        conv_out = self.conv_layers(x)
        flatten = conv_out.view(conv_out.size(0), -1)
        return self.fc(flatten)


class L1Dist(nn.Module):
    """
    L1 distance metric for Siamese Network.

    Returns:
        Tensor: L1 distance between the input tensors.
    """

    def __init__(self):
        super(L1Dist, self).__init__()

    def forward(self, x1, x2):
        return torch.abs(x1 - x2)
        # return 1 - nn.functional.cosine_similarity(x1, x2, dim=1).unsqueeze(1)


class SiameseNetwork(nn.Module):
    """
    Siamese Network for face verification with modified embedding model.
    """

    def __init__(self, embedding_model):
        super(SiameseNetwork, self).__init__()
        self.embedding_model = embedding_model
        self.l1_dist = L1Dist()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_model.fc_layers[-2].out_features, 1), nn.Sigmoid()
        )
        # self.classifier = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    def forward(self, x1, x2, embed1, embed2):
        embed1 = self.embedding_model(x1, embed1)
        embed2 = self.embedding_model(x2, embed2)
        l1_dist = self.l1_dist(embed1, embed2)
        return self.classifier(l1_dist)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Siamese Network.

    Args:
        margin (float): Margin for contrastive loss.
        distance_metric (str): Distance metric to use ('euclidean', 'mse', 'cosine', 'braycurtis').

    Returns:
        Tensor: Computed loss value.
    """

    def __init__(self, margin=1.0, distance_metric="mse"):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, output1, output2, label):
        if self.distance_metric == "euclidean":
            distance = nn.functional.pairwise_distance(output1, output2)
        elif self.distance_metric == "mse":
            distance = torch.mean((output1 - output2) ** 2, dim=1)
        elif self.distance_metric == "cosine":
            distance = 1 - nn.functional.cosine_similarity(output1, output2)
        elif self.distance_metric == "braycurtis":
            distance = braycurtis_torch(output1, output2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        loss = label * torch.pow(distance, 2) + (1 - label) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2
        )
        return torch.mean(loss)


def braycurtis_torch(tensor1, tensor2):
    """
    Compute Bray-Curtis distance between two tensors.

    Args:
        tensor1 (Tensor): First tensor.
        tensor2 (Tensor): Second tensor.

    Returns:
        Tensor: Bray-Curtis distance.
    """
    return torch.sum(torch.abs(tensor1 - tensor2), dim=1) / torch.sum(
        torch.abs(tensor1 + tensor2), dim=1
    )


def calculate_accuracy(outputs, labels):
    """
    Calculate accuracy given outputs and labels.

    Args:
        outputs (Tensor): Model outputs.
        labels (Tensor): Ground truth labels.

    Returns:
        float: Accuracy value.
    """
    predicted = (outputs > 0.5).float()
    correct = (predicted == labels).float().sum()
    accuracy = correct / labels.size(0)
    return accuracy


def load_image(image_path):
    """
    Load and preprocess an image.

    Args:
        image_path (str): Path to the image.

    Returns:
        np.ndarray: Loaded and preprocessed image.
    """
    print("Loading image")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h = 512
    w = int(image.shape[1] * h / image.shape[0])
    image = cv2.resize(image, (w, h))
    return image


def train_siamese_network(
    pairs,
    labels,
    batch_size=128,
    epochs=25,
    patience=7,
    factor=0.1,
    learning_rate=0.0001,
    max_grad_norm=1.0,
    distance_metric="euclidean",
):
    """
    Train the Siamese network.

    Args:
        pairs (list): List of pairs of input data.
        labels (list): List of labels corresponding to the pairs.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        patience (int): Patience for early stopping.
        factor (float): Factor by which the learning rate is reduced.
        max_grad_norm (float): Maximum gradient norm for clipping.
        distance_metric (str): Distance metric to use.

    Returns:
        nn.Module: Trained Siamese network model.
    """
    print_blue("Starting Siamese network training...")

    # Calculate the correct input shape
    input_shape = (1, 478, 3, 1)
    embedding_dim = 512

    print(f"Input shape: {input_shape}")

    train_pairs, test_pairs, y_train, y_test = train_test_split(
        pairs, labels, test_size=0.2, random_state=42, stratify=labels, shuffle=True
    )

    print("-" * 50)
    print(f"Train pairs: {len(train_pairs):,}, Test pairs: {len(test_pairs):,}")
    print(f"Train labels: {len(y_train):,}, Test labels: {len(y_test):,}")
    print(f"Batch size: {batch_size:,}, Epochs: {epochs:,}")

    train_pos_pairs = sum(y_train)
    train_neg_pairs = len(y_train) - train_pos_pairs
    test_pos_pairs = sum(y_test)
    test_neg_pairs = len(y_test) - test_pos_pairs

    print("-" * 50)
    print(f"In Train: positive: {train_pos_pairs:,}, negative: {train_neg_pairs:,}")
    print(
        f"In Train: positive: {train_pos_pairs / len(y_train) * 100:.2f}% / negative: {train_neg_pairs / len(y_train) * 100:.2f}%"
    )
    print("-" * 50)
    print(f"In Test: positive: {test_pos_pairs:,}, negative: {test_neg_pairs:,}")
    print(
        f"In Test: positive: {test_pos_pairs / len(y_test) * 100:.2f}% / negative: {test_neg_pairs / len(y_test) * 100:.2f}%"
    )
    print("-" * 50)

    train_dataset = FaceLandmarksDataset(train_pairs, y_train)
    test_dataset = FaceLandmarksDataset(test_pairs, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    embedding_model = EmbeddingModel(input_shape, embedding_dim).to(device)
    model = SiameseNetwork(embedding_model).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=3)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    best_loss = float("inf")
    early_stopping_counter = 0
    no_improvement_counter = 0

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_idx, ((points1, points2), labels) in enumerate(train_loader):
            print(
                f"\rProcessing: Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}]",
                end="",
                flush=True,
            )
            points1, points2, labels = (
                points1.to(device).float(),
                points2.to(device).float(),
                labels.to(device).float(),
            )
            print(f"Batch input shape: {points1.shape}")
            optimizer.zero_grad()
            embed1 = torch.tensor(
                [get_face_embedding(facenet_model, p) for p in points1],
                dtype=torch.float32,
            ).to(device)
            embed2 = torch.tensor(
                [get_face_embedding(facenet_model, p) for p in points2],
                dtype=torch.float32,
            ).to(device)
            outputs = model(points1, points2, embed1, embed2)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                correct_train += (outputs.round() == labels.unsqueeze(1)).sum().item()
                total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        running_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for (test_points1, test_points2), test_labels in test_loader:
                test_points1, test_points2, test_labels = (
                    test_points1.to(device).float(),
                    test_points2.to(device).float(),
                    test_labels.to(device).float(),
                )
                embed1 = torch.tensor(
                    [get_face_embedding(facenet_model, p) for p in test_points1],
                    dtype=torch.float32,
                ).to(device)
                embed2 = torch.tensor(
                    [get_face_embedding(facenet_model, p) for p in test_points2],
                    dtype=torch.float32,
                ).to(device)
                test_outputs = model(test_points1, test_points2, embed1, embed2)
                loss = criterion(test_outputs, test_labels.unsqueeze(1))
                running_loss += loss.item()

                correct_test += (
                    (test_outputs.round() == test_labels.unsqueeze(1)).sum().item()
                )
                total_test += test_labels.size(0)

        test_loss = running_loss / len(test_loader)
        test_losses.append(test_loss)
        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)

        scheduler.step(test_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        remaining_time = epoch_duration * (epochs - (epoch + 1))

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\rEpoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy * 100:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy * 100:.4f} | "
            f"Time: {epoch_duration:.2f} seconds | Time left: {remaining_time:.2f} seconds | "
            f"Learning Rate: {current_lr:.6f}"
        )

        if test_loss < best_loss:
            best_loss = test_loss
            early_stopping_counter = 0
            no_improvement_counter = 0
            if SAVE_MODEL:
                torch.save(model.state_dict(), PATH_FOR_MODEL)
                print(f"\rModel improved and saved to {PATH_FOR_MODEL}")
        else:
            early_stopping_counter += 1
            no_improvement_counter += 1
            if no_improvement_counter >= 3:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.1
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"\rLearning rate reduced to {current_lr:.6f}")
                no_improvement_counter = 0

            if early_stopping_counter >= patience:
                print_red("\rEarly stopping triggered. Stopping training.")
                break

    total_time = time.time() - start_time
    print_green(f"Training completed in {total_time:.2f} seconds.")

    completed_epochs = len(train_losses)
    plot_training_results(
        completed_epochs, train_losses, test_losses, train_accuracies, test_accuracies
    )

    return model


def plot_training_results(
    completed_epochs, train_losses, test_losses, train_accuracies, test_accuracies
):
    """
    Plot training and test loss and accuracy.

    Args:
        completed_epochs (int): Number of completed epochs.
        train_losses (list): List of training losses.
        test_losses (list): List of test losses.
        train_accuracies (list): List of training accuracies.
        test_accuracies (list): List of test accuracies.
    """
    epochs_range = range(1, completed_epochs + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs_range, test_losses, label="Test Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label="Train Accuracy", color="blue")
    plt.plot(epochs_range, test_accuracies, label="Test Accuracy", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def detect_face(image):
    """
    Detect the largest face in an image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        tuple: Cropped face region and bounding box coordinates (x, y, w, h).
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None, None
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    return image[y : y + h, x : x + w], (x, y, w, h)


def get_face_landmarks(image, face_mesh):
    """
    Get face landmarks using MediaPipe Face Mesh.

    Args:
        image (np.ndarray): Input image.
        face_mesh (mp.solutions.face_mesh.FaceMesh): MediaPipe Face Mesh object.

    Returns:
        np.ndarray: Array of face landmarks.
    """
    h = 256
    w = int(image.shape[1] * h / image.shape[0])
    image = cv2.resize(image, (w, h))

    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0]
    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
    return np.array(landmarks)


def create_3d_model(landmarks):
    """
    Create a 3D model using Delaunay triangulation.

    Args:
        landmarks (np.ndarray): Array of face landmarks.

    Returns:
        tuple: Points and Delaunay triangulation object.
    """
    points = np.array(landmarks)
    tri = Delaunay(points[:, :2])
    return points, tri


def plot_3d_model_with_texture(ax, image, points, tri, show_landmarks=False):
    """
    Plot 3D model with texture.

    Args:
        ax (Axes3D): Matplotlib 3D axis object.
        image (np.ndarray): Input image.
        points (np.ndarray): Array of 3D points.
        tri (Delaunay): Delaunay triangulation object.
        show_landmarks (bool): Whether to show landmarks or not.
    """
    ax.clear()
    h, w, _ = image.shape

    # Get the UV coordinates by scaling the points to the image dimensions
    uv_coords = np.array([(point[0], point[1]) for point in points])

    faces = tri.simplices
    face_vertices = points[faces]

    face_colors = []

    for i, face in enumerate(faces):
        triangle_2d = uv_coords[face].astype(np.int32)
        triangle_3d = points[face]

        # Create a mask for the triangle in the 2D image
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, triangle_2d, 1)

        # Extract the texture from the 2D image
        texture = cv2.bitwise_and(image, image, mask=mask)

        # Calculate the average color within the triangle
        mean_color = cv2.mean(image, mask=mask)[:3]
        face_color = np.array(mean_color) / 255.0

        if np.any(face_color < 0) or np.any(face_color > 1):
            print(f"Warning: Face color out of range at face {i}")

        face_colors.append(face_color)

    face_colors = np.array(face_colors)

    poly3d = Poly3DCollection(
        face_vertices, facecolors=face_colors, linewidths=0.1, edgecolors="k"
    )
    ax.add_collection3d(poly3d)

    if show_landmarks:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", s=5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_zlim(points[:, 2].min(), points[:, 2].max())

    ax.view_init(elev=-90, azim=270)

    plt.draw()
    plt.pause(0.00001)


def draw_landmarks_on_image(image, landmarks):
    """
    Draw landmarks on the image.

    Args:
        image (np.ndarray): Input image.
        landmarks (np.ndarray): Array of landmarks.

    Returns:
        np.ndarray: Image with landmarks drawn.
    """
    for x, y, z in landmarks:
        cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)
    return image


def update_view(ax_from, ax_to):
    """
    Update the view of one axis based on another axis.

    Args:
        ax_from (Axes3D): Axis to copy the view from.
        ax_to (Axes3D): Axis to apply the view to.
    """
    elev = ax_from.elev
    azim = ax_from.azim
    ax_to.view_init(elev=elev, azim=azim)


def on_move(event, ax3d_left, ax3d_right):
    """
    Handle mouse movement events for 3D plots.

    Args:
        event (Event): Mouse event.
        ax3d_left (Axes3D): Left 3D axis.
        ax3d_right (Axes3D): Right 3D axis.
    """
    if not event.inaxes:
        return
    for ax in [ax3d_left, ax3d_right]:
        if event.inaxes == ax:
            update_view(ax, ax3d_left if ax == ax3d_right else ax3d_right)
            fig.canvas.draw_idle()
            break


def rotate_points(points, elev, azim, roll):
    """
    Rotate 3D points using Euler angles.

    Args:
        points (np.ndarray): Array of 3D points.
        elev (float): Elevation angle.
        azim (float): Azimuth angle.
        roll (float): Roll angle.

    Returns:
        np.ndarray: Rotated points.
    """
    r = R.from_euler("xyz", [elev, azim, roll], degrees=True)
    return r.apply(points)


def save_rotated_points_to_file(rotated_points, filename, labels):
    """
    Save rotated points to a file.

    Args:
        rotated_points (list): List of rotated points.
        filename (str): Path to the output file.
        labels (list): List of labels corresponding to the points.
    """
    print_green("Saving rotated points to file...")
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "w") as f:
        for (points, embedding), label in zip(rotated_points, labels):
            line = f"{label} " + " ".join(f"{x},{y},{z}" for x, y, z in points)
            line += " " + " ".join(f"{e}" for e in embedding)
            f.write(line + "\n")
    print(f"Rotated points saved to {filename}")


def save_vectors_to_file(vectors, filename):
    """
    Save vectors to a file.

    Args:
        vectors (list): List of tuples where each tuple contains 3D points and an embedding.
        filename (str): Path to the output file.
    """
    print_green(f"Saving vectors to file: {filename}")
    points_list = [v[0] for v in vectors]
    embeddings_list = [v[1] for v in vectors]
    np.savez(filename, points=points_list, embeddings=embeddings_list)
    print(f"Vectors saved to {filename}")


def load_vectors_from_file(filename):
    """
    Load vectors from a file.

    Args:
        filename (str): Path to the input file.

    Returns:
        list: Loaded vectors as a list of tuples where each tuple contains 3D points and an embedding.
    """
    print_green(f"Loading vectors from file: {filename}")
    data = np.load(filename)
    points_list = data["points"]
    embeddings_list = data["embeddings"]
    vectors = [
        (points, embedding) for points, embedding in zip(points_list, embeddings_list)
    ]
    return vectors


def load_rotated_points_from_file(filename):
    """
    Load rotated points from a file.

    Args:
        filename (str): Path to the input file.

    Returns:
        tuple: Loaded rotated points and labels.
    """
    rotated_points = []
    labels = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            labels.append(parts[0])
            points = []
            embedding = []
            for part in parts[1:]:
                if "," in part:
                    x, y, z = map(float, part.split(","))
                    points.append([x, y, z])
                else:
                    embedding.append(float(part))
            rotated_points.append((np.array(points), np.array(embedding)))
    print(f"Rotated points loaded from {filename}")
    return rotated_points, labels


def process_image(image_path, face_mesh, facenet_model):
    """
    Process an image to extract face landmarks and embedding.

    Args:
        image_path (str): Path to the image.
        face_mesh (mp.solutions.face_mesh.FaceMesh): MediaPipe Face Mesh object.
        facenet_model (nn.Module): FaceNet model for embedding extraction.

    Returns:
        tuple: Extracted 3D points and embedding.
    """
    print_blue("Processing image...")
    print(f"Image path: {image_path}")
    image = load_image(image_path)

    face_region, bbox = detect_face(image)
    if face_region is not None:
        landmarks = get_face_landmarks(face_region, face_mesh)
        if landmarks is not None:
            points, tri = create_3d_model(landmarks)
            embedding = get_face_embedding(facenet_model, face_region)

            if SHOW_3D_PLOT:
                plot_3d_results(face_region, bbox, landmarks, points, tri, image)

            points = normalize_points(points)
            print(
                f"Points shape: {points.shape}, Embedding shape: {embedding.shape}, Total size: {points.size + embedding.size}"
            )
            print_green("Processing image completed.")
            print("-" * 50)
            return points, embedding
        else:
            print("No face landmarks detected in the image.")
            return None, None
    else:
        print("No face detected in the image.")
        return None, None


def plot_3d_results(face_region, bbox, landmarks, points, tri, image):
    """
    Plot 3D results of face landmarks and texture.

    Args:
        face_region (np.ndarray): Cropped face region.
        bbox (tuple): Bounding box coordinates.
        landmarks (np.ndarray): Array of landmarks.
        points (np.ndarray): Array of 3D points.
        tri (Delaunay): Delaunay triangulation object.
        image (np.ndarray): Original image.
    """
    global fig, ax3d_left, ax3d_right
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    ax3d_left = fig.add_subplot(223, projection="3d")
    ax3d_right = fig.add_subplot(224, projection="3d")

    plot_3d_model_with_texture(
        ax3d_left, face_region, points, tri, show_landmarks=False
    )
    plot_3d_model_with_texture(
        ax3d_right, face_region, points, tri, show_landmarks=True
    )

    image_with_landmarks = draw_landmarks_on_image(face_region.copy(), landmarks)

    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")
    x, y, w, h = bbox
    rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
    axs[0, 0].add_patch(rect)

    axs[0, 1].imshow(image_with_landmarks)
    axs[0, 1].set_title("Image with Landmarks")
    axs[0, 1].axis("off")

    axs[1, 0].set_title("3D Model without Landmarks")
    axs[1, 1].set_title("3D Model with Landmarks")

    fig.canvas.mpl_connect(
        "motion_notify_event",
        lambda event: on_move(event, ax3d_left, ax3d_right),
    )

    plt.show()


def rotate_3d_points(points, embedding, positive_images=False):
    """
    Rotate 3D points to create multiple views.

    Args:
        points (np.ndarray): Array of 3D points.
        embedding (np.ndarray): Embedding vector.
        positive_images (bool): Whether the points are from positive images.

    Returns:
        list: List of rotated points.
    """
    print_green("Starting rotation model...")

    if SHOW_3D_PLOT:
        ax = plot_initial_3d_points(points)

    list_rotated_points = []

    if positive_images:
        # for x_angle in range(-10, 91, 5):
        #     for y_angle in range(-45, 46, 5):
        #         for z_angle in range(-7, 8, 7):
        for x_angle in range(25, 65, 2):
            for y_angle in range(-25, 25, 2):
                for z_angle in range(-7, 8, 7):
                    rotated_points = rotate_points(points, x_angle, y_angle, z_angle)
                    list_rotated_points.append((rotated_points, embedding))
                    if SHOW_3D_PLOT:
                        plot_rotated_3d_points(ax, points, rotated_points)
    else:
        for x_angle in range(45, 46, 1):
            for y_angle in range(0, 1, 1):
                for z_angle in range(0, 1, 1):
                    rotated_points = rotate_points(points, x_angle, y_angle, z_angle)
                    list_rotated_points.append((rotated_points, embedding))

                    if SHOW_3D_PLOT:
                        plot_rotated_3d_points(ax, points, rotated_points)

    if SHOW_3D_PLOT:
        plt.show()

    print(f"Rotation completed. add {len(list_rotated_points)} points to the list.")
    print("-" * 50)

    return list_rotated_points


def plot_initial_3d_points(points):
    """
    Plot initial 3D points.

    Args:
        points (np.ndarray): Array of 3D points.

    Returns:
        Axes3D: Matplotlib 3D axis object.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", marker=".")

    ax.scatter(0, 0, 0, c="b", marker="o")

    ax.set_xlim([points[:, 0].min(), points[:, 0].max()])
    ax.set_ylim([points[:, 1].min(), points[:, 1].max()])
    ax.set_zlim([points[:, 2].min(), points[:, 2].max()])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("3D Points Rotation")

    ax.view_init(elev=-90, azim=270)

    plt.draw()
    plt.pause(0.00001)
    return ax


def plot_rotated_3d_points(ax, points, rotated_points):
    """
    Plot rotated 3D points.

    Args:
        ax (Axes3D): Matplotlib 3D axis object.
        points (np.ndarray): Original 3D points.
        rotated_points (np.ndarray): Rotated 3D points.
    """
    ax.clear()
    ax.scatter(
        rotated_points[:, 0],
        rotated_points[:, 1],
        rotated_points[:, 2],
        c="r",
        marker=".",
    )
    ax.scatter(0, 0, 0, c="b", marker="o")
    ax.set_xlim([points[:, 0].min(), points[:, 0].max()])
    ax.set_ylim([points[:, 1].min(), points[:, 1].max()])
    ax.set_zlim([points[:, 2].min(), points[:, 2].max()])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Points Rotation")
    ax.view_init(elev=-90, azim=270)
    plt.draw()
    plt.pause(0.00001)


def get_distances(points):
    """
    Get vertical and horizontal distances of points.

    Args:
        points (np.ndarray): Array of 3D points.

    Returns:
        tuple: Vertical and horizontal distances.
    """
    vertical_dist = np.max(points[:, 1]) - np.min(points[:, 1])
    horizontal_dist = np.max(points[:, 0]) - np.min(points[:, 0])
    return vertical_dist, horizontal_dist


def align_face(points):
    """
    Align face points using specific landmarks.

    Args:
        points (np.ndarray): Array of 3D points.

    Returns:
        np.ndarray: Aligned 3D points.
    """
    print("Aligning face...")
    # Assuming the points array is in the shape of (num_points, 3)
    # Indices for landmarks: 10 (forehead), 152 (chin), 1 (nose tip), 33 (left eye), 263 (right eye)
    forehead_index = 10
    chin_index = 152
    nose_index = 1
    left_eye_index = 33
    right_eye_index = 263

    # Get the points for forehead, chin, nose, and eyes
    forehead_point = points[forehead_index]
    chin_point = points[chin_index]
    nose_point = points[nose_index]
    left_eye_point = points[left_eye_index]
    right_eye_point = points[right_eye_index]

    # Calculate the vector from chin to forehead for the y-axis
    y_axis = forehead_point - chin_point
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Ensure the y-axis is pointing upwards
    if y_axis[1] < 0:
        y_axis = -y_axis

    # Calculate the vector from the nose to the midpoint of the eyes for the z-axis
    eye_midpoint = (left_eye_point + right_eye_point) / 2
    z_axis = nose_point - eye_midpoint
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Ensure the z-axis is pointing towards the camera
    if z_axis[2] < 0:
        z_axis = -z_axis

    # Calculate the x-axis to be perpendicular to both the y and z axes
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Recalculate the y-axis to ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Create the rotation matrix
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Apply the rotation
    aligned_points = points @ rotation_matrix

    return aligned_points


def normalize_points(points, target_distance=100.0, reference_index=8):
    """
    Normalize 3D points.

    Args:
        points (np.ndarray): Array of 3D points.
        target_distance (float): Target distance for normalization.
        reference_index (int): Index of the reference point.

    Returns:
        np.ndarray: Normalized 3D points.
    """
    print("Normalizing points...")
    points = points.astype(np.float64)

    # Align the points
    aligned_points = align_face(points)

    # Find the center and scale the points
    center = np.mean(aligned_points, axis=0)
    points_centered = aligned_points - center
    # reference_point = points_centered[reference_index]
    # current_distance = np.linalg.norm(reference_point)
    # scale = target_distance / current_distance
    # normalized_points = points_centered * scale

    return points_centered


def shuffle_pairs_labels(pairs, labels):
    """
    Shuffle pairs and labels.

    Args:
        pairs (list): List of pairs.
        labels (list): List of labels.

    Returns:
        tuple: Shuffled pairs and labels.
    """
    print_blue("Shuffling pairs and labels...")
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    print_green("Shuffling pairs and labels completed.")
    return zip(*combined)


def compare_landmarks(image_path1, image_path2, face_mesh):
    """
    Compare landmarks between two images.

    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.
        face_mesh (mp.solutions.face_mesh.FaceMesh): MediaPipe Face Mesh object.
    """
    print_green("Comparing landmarks from two images...")

    points1 = process_image(image_path1, face_mesh)
    points2 = process_image(image_path2, face_mesh)

    if points1 is not None and points2 is not None:
        plot_comparison_results(image_path1, image_path2, points1, points2)
    else:
        print("Landmarks could not be detected in one or both images.")

    print("Comparison completed.")
    print("-" * 50)


def plot_comparison_results(image_path1, image_path2, points1_aligned, points2_aligned):
    """
    Plot comparison results of landmarks between two images.

    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.
        points1_aligned (np.ndarray): Aligned points of the first image.
        points2_aligned (np.ndarray): Aligned points of the second image.
    """
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(
        points1_aligned[:, 0],
        points1_aligned[:, 1],
        points1_aligned[:, 2],
        c="b",
        marker=".",
    )
    ax1.set_title(f"First Image Landmarks {image_path1}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=-90, azim=270)

    ax2 = fig.add_subplot(133, projection="3d")
    ax2.scatter(
        points2_aligned[:, 0],
        points2_aligned[:, 1],
        points2_aligned[:, 2],
        c="r",
        marker=".",
    )
    ax2.set_title(f"Second Image Landmarks {image_path2}")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=-90, azim=270)

    ax3 = fig.add_subplot(132, projection="3d")
    ax3.scatter(
        points1_aligned[:, 0],
        points1_aligned[:, 1],
        points1_aligned[:, 2],
        c="b",
        marker=".",
    )
    ax3.scatter(
        points2_aligned[:, 0],
        points2_aligned[:, 1],
        points2_aligned[:, 2],
        c="r",
        marker=".",
    )
    ax3.set_title("Both Landmarks (Blue & Red)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.view_init(elev=-90, azim=270)

    plt.show()


def process_folder_images(folder_path, face_mesh, facenet_model, max_files=None):
    """
    Process all images in a folder.

    Args:
        folder_path (str): Path to the folder.
        face_mesh (mp.solutions.face_mesh.FaceMesh): MediaPipe Face Mesh object.
        facenet_model (nn.Module): FaceNet model for embedding extraction.
        max_files (int, optional): Maximum number of files to process.

    Returns:
        dict: Dictionary with image paths as keys and (points, embedding) as values.
    """
    image_landmarks = {}
    file_count = 0

    for filename in os.listdir(folder_path):
        if filename.startswith("."):
            continue
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path):
            points, embedding = process_image(image_path, face_mesh, facenet_model)
            if points is not None and embedding is not None:
                image_landmarks[image_path] = (points, embedding)
                file_count += 1
                if max_files is not None and file_count >= max_files:
                    break

    return image_landmarks


def create_pairs(positive_landmarks, negative_landmarks):
    """
    Create pairs for training.

    Args:
        positive_landmarks (list): List of positive landmarks.
        negative_landmarks (list): List of negative landmarks.

    Returns:
        tuple: Created pairs and labels.
    """
    print_blue("Creating pairs for training...")

    pairs = []
    labels = []

    # Positive-Positive pairs
    for i in range(len(positive_landmarks)):
        for j in range(i + 1, len(positive_landmarks)):
            pairs.append((positive_landmarks[i], positive_landmarks[j]))
            labels.append(1)
    p = len(pairs)
    print(f"Positive-Positive pairs: {len(pairs):,}")

    # Positive-Negative pairs
    for pos_points in positive_landmarks:
        for neg_points in negative_landmarks:
            pairs.append((pos_points, neg_points))
            labels.append(0)
    print(f"Positive-Negative pairs: {(len(pairs) - p):,}")
    print_green(f"Total pairs: {len(pairs):,}")
    print("-" * 50)
    return pairs, labels


def load_facenet_model():
    """
    Load the pre-trained FaceNet model.

    Returns:
        nn.Module: Loaded FaceNet model.
    """
    print_blue("Loading FaceNet model...")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    print_green("FaceNet model loaded.")
    return model


def load_mesh_model():
    """
    Load the pre-trained MediaPipe Face Mesh model.

    Returns:
        mp.solutions.face_mesh.FaceMesh: Loaded Face Mesh model.
    """
    print_blue("Loading MediaPipe Face Mesh model...")
    model = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    )
    print_green("MediaPipe Face Mesh model loaded.")
    return model


def get_face_embedding(model, face):
    """
    Get face embedding from a face region.

    Args:
        model (nn.Module): FaceNet model.
        face (np.ndarray): Face region.

    Returns:
        np.ndarray: Extracted embedding.
    """
    h = 256
    w = int(face.shape[1] * h / face.shape[0])
    face = cv2.resize(face, (w, h))

    face = face.transpose((2, 0, 1))
    face = torch.tensor(face, dtype=torch.float32).unsqueeze(0).to(device)
    embedding = model(face).detach().cpu().numpy().flatten()
    return embedding


def main(root_folder_path):
    """
    Main function to process images and train the Siamese network.

    Args:
        root_folder_path (str): Path to the root folder containing image data.
    """
    mp_face_mesh = load_mesh_model()
    facenet_model = load_facenet_model()

    if CREATE_NEW_DATA:
        print_blue("Processing new data...")
        positives_folder = os.path.join(root_folder_path, "positives")
        negatives_folder = os.path.join(root_folder_path, "negatives")

        print_blue("Processing images from the positives folder")
        positive_landmarks = process_folder_images(
            positives_folder, mp_face_mesh, facenet_model
        )

        print_blue("Processing images from the negatives folder")
        negative_landmarks = process_folder_images(
            negatives_folder, mp_face_mesh, facenet_model, max_files=3
        )

        print_green(f"Positive images processed: {len(positive_landmarks):,}")
        print_green(f"Negative images processed: {len(negative_landmarks):,}")
        print("-" * 50)

        positive_points = []
        negative_points = []
        for idx, (points, embedding) in enumerate(positive_landmarks.values()):
            list_rotated_points_3D_positive = rotate_3d_points(
                points, embedding, positive_images=True
            )
            positive_points.extend(list_rotated_points_3D_positive)

        for idx, (points, embedding) in enumerate(negative_landmarks.values()):
            list_rotated_points_3D_negative = rotate_3d_points(
                points, embedding, positive_images=False
            )
            negative_points.extend(list_rotated_points_3D_negative)

        save_vectors_to_file(positive_points, POSITIVE_VECTORS_FILE)
        save_vectors_to_file(negative_points, NEGATIVE_VECTORS_FILE)
        print(f"Positive samples: {len(positive_points):,}")
        print(f"Negative samples: {len(negative_points):,}")
    else:
        print_green("Loading existing data from files...")
        positive_points = load_vectors_from_file(POSITIVE_VECTORS_FILE)
        negative_points = load_vectors_from_file(NEGATIVE_VECTORS_FILE)
        print(
            f"Loaded {len(positive_points):,} positive points from {POSITIVE_VECTORS_FILE}"
        )
        print(
            f"Loaded {len(negative_points):,} negative points from {NEGATIVE_VECTORS_FILE}"
        )

    pairs, labels = create_pairs(positive_points, negative_points)

    if pairs:
        print(f"Type of pairs: {type(pairs)}")
        print(f"Type of pairs[0]: {type(pairs[0])}")
        print(f"Type of pairs[0][0]: {type(pairs[0][0])}")
        print(pairs[0][0][0].shape)  # Accessing the 3D points shape
        print(f"Type of pairs[0][1]: {type(pairs[0][1])}")
        print(pairs[0][0][1].shape)  # Accessing the embedding shape

        # Shuffle the pairs and labels
        pairs, labels = shuffle_pairs_labels(pairs, labels)

        model = train_siamese_network(pairs, labels)

        if SAVE_MODEL:
            print_blue("Saving the model...")
            torch.save(model.state_dict(), PATH_FOR_MODEL)
            print_green(f"Model saved to {PATH_FOR_MODEL}")
    else:
        print("No pairs available for training.")


# ----------------- Test the model -----------------
def load_trained_model(model_path, input_shape, embedding_dim):
    """
    Load a trained Siamese network model.

    Args:
        model_path (str): Path to the saved model.
        input_shape (tuple): Shape of the input points.
        embedding_dim (int): Dimension of the embedding.

    Returns:
        nn.Module: Loaded Siamese network model.
    """
    print_blue("Loading the trained model...")
    embedding_model = EmbeddingModel(input_shape, embedding_dim).to(device)
    model = SiameseNetwork(embedding_model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print_green("Trained model loaded.")
    return model


def calculate_threshold(model, positive_vectors, max_samples=100):

    print_blue("Calculating threshold...")

    threshold_output = []
    for sample1 in random.sample(
        positive_vectors, min(max_samples, len(positive_vectors))
    ):
        for sample2 in random.sample(
            positive_vectors, min(max_samples, len(positive_vectors))
        ):
            model.eval()
            with torch.no_grad():
                sample1_tensor = torch.tensor(sample1, dtype=torch.float32).to(device)
                sample2_tensor = torch.tensor(sample2, dtype=torch.float32).to(device)
                output = model(sample1_tensor.unsqueeze(0), sample2_tensor.unsqueeze(0))
                threshold_output.append(output.item())
    # fiend the number that greater than 85% of all the output
    threshold_output.sort()
    threshold = threshold_output[int(len(threshold_output) * 0.65)]
    print_green(f"Threshold: {threshold}")
    print("-" * 50)

    return threshold


def test_image(
    image_path,
    model,
    mp_face_mesh,
    facenet_model,
    positive_vectors,
    negative_vectors,
    max_samples_for_comparison=1000,
    threshold=0.5,
):
    """
    Test a new image using the trained Siamese network model.

    Args:
        image_path (str): Path to the test image.
        model (nn.Module): Trained Siamese network model.
        mp_face_mesh (mp.solutions.face_mesh.FaceMesh): MediaPipe Face Mesh object.
        facenet_model (nn.Module): FaceNet model for embedding extraction.
        positive_vectors (list): List of positive vectors.
        negative_vectors (list): List of negative vectors.
        max_samples_for_comparison (int): Maximum number of samples to compare.
    """
    print_blue("Testing new image...")
    points, embedding = process_image(image_path, mp_face_mesh, facenet_model)

    if points is not None and embedding is not None:
        points = np.concatenate((points.flatten(), embedding))
        points_tensor = torch.tensor(points, dtype=torch.float32).to(device)

        # Randomly select samples for comparison
        positive_samples = random.sample(
            positive_vectors, min(max_samples_for_comparison, len(positive_vectors))
        )
        negative_samples = random.sample(
            negative_vectors, min(max_samples_for_comparison, len(negative_vectors))
        )

        positive_pairs = [
            (points_tensor, torch.tensor(vec, dtype=torch.float32).to(device))
            for vec in positive_samples
        ]
        negative_pairs = [
            (points_tensor, torch.tensor(vec, dtype=torch.float32).to(device))
            for vec in negative_samples
        ]

        model.eval()
        with torch.no_grad():
            positive_scores = []
            for pair in positive_pairs:
                model_output = model(pair[0].unsqueeze(0), pair[1].unsqueeze(0))
                positive_scores.append(model_output.item())
                print(f"Positive score: {model_output.item()}")
            print("-" * 50)

            negative_scores = []
            for pair in negative_pairs:
                model_output = model(pair[0].unsqueeze(0), pair[1].unsqueeze(0))
                negative_scores.append(model_output.item())
                print(f"Negative score: {model_output.item()}")

        positive_matches = sum(1 for score in positive_scores if score <= threshold)
        negative_matches = sum(1 for score in negative_scores if score >= threshold)

        positive_percentage = positive_matches / max_samples_for_comparison * 100
        negative_percentage = negative_matches / max_samples_for_comparison * 100

        print(f"Positive matches: {positive_matches} / {max_samples_for_comparison}")
        print(f"Negative matches: {negative_matches} / {max_samples_for_comparison}")
        print(f"Positive percentage: {positive_percentage:.2f}%")
        print(f"Negative percentage: {negative_percentage:.2f}%")
        print("-" * 50)

        is_positive = positive_percentage >= 50

        # Display the image and write a title based on the prediction
        image = load_image(image_path)
        plt.imshow(image)
        title = "Same Person" if is_positive else "Different Person"
        title += f" (Matches: {positive_percentage:.2f}%)"
        color = "green" if is_positive else "red"
        plt.title(title, color=color)
        plt.axis("off")
        plt.show()

        print(f"Model output: {title}")
    else:
        print_red("Failed to process the new image or detect landmarks.")

    print_green("Testing completed.")


if __name__ == "__main__":
    print_green("Starting...")
    print_blue(f"Running on device: {device}")
    main(ROOT_FOLDER_PATH)

    # Load the trained model with the correct input dimension
    input_shape = (1, 478, 3, 1)
    embedding_dim = 512

    model = load_trained_model(PATH_FOR_MODEL, input_shape, embedding_dim)
    mp_face_mesh = load_mesh_model()
    facenet_model = load_facenet_model()

    # Load positive and negative vectors from file
    positive_vectors = load_vectors_from_file(POSITIVE_VECTORS_FILE)
    negative_vectors = load_vectors_from_file(NEGATIVE_VECTORS_FILE)

    # Test on a new image
    test_images_folder = TEST_IMAGES_FOLDER

    # Calculate the threshold
    threshold = calculate_threshold(
        model=model, positive_vectors=positive_vectors, max_samples=100
    )

    for filename in os.listdir(test_images_folder):
        new_image_path = os.path.join(test_images_folder, filename)
        if os.path.isfile(new_image_path):
            test_image(
                new_image_path,
                model,
                mp_face_mesh,
                facenet_model,
                positive_vectors,
                negative_vectors,
                max_samples_for_comparison=100,
                threshold=threshold,
            )
