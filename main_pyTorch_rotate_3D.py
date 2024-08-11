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
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

SHOW_3D_PLOT = False
CREATE_NEW_DATA = True
PATH_FOR_DATA = "./rotated_points.txt"
SAVE_MODEL = True
PATH_FOR_MODEL = "./face_recognition_model.pth"
RUN_ON_GPU = True

# Check for MPS (Metal Performance Shaders) for MacBook Pro with M3 chip
device = torch.device(
    "mps" if torch.backends.mps.is_available() and RUN_ON_GPU else "cpu"
)


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_red(text):
    print(f"\033[91m{text}\033[0m")


class FaceLandmarksDataset(Dataset):
    def __init__(self, points, labels, transform=None):
        self.points = points
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = self.points[idx]
        label = self.labels[idx]

        if self.transform:
            points = self.transform(points)

        points = torch.tensor(points, dtype=torch.float32)
        return points, label


def build_model(input_dim, num_of_classes=2):
    model = nn.Sequential(
        nn.Linear(input_dim, 2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 2048),
        nn.Linear(2048, 1024),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        # nn.ReLU(),
        nn.Linear(256, 256),
        # nn.ReLU(),
        nn.Linear(256, 256),
        # nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        # nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 32),
        nn.Linear(32, num_of_classes),
    )
    return model


def train_face_recognition_model(
    points, labels, num_of_classes, batch_size=128, epochs=25
):
    print_green("Starting model training...")

    points_flattened = np.array([point_set.flatten() for point_set in points])

    X_train, X_test, y_train, y_test = train_test_split(
        points_flattened, labels, test_size=0.1, random_state=42
    )

    train_dataset = FaceLandmarksDataset(X_train, y_train)
    test_dataset = FaceLandmarksDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = build_model(input_dim, num_of_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for test_inputs, test_labels in test_loader:
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)
                test_outputs = model(test_inputs)
                loss = criterion(test_outputs, test_labels)
                running_loss += loss.item()
                _, test_predicted = torch.max(test_outputs, 1)
                total += test_labels.size(0)
                correct += (test_predicted == test_labels).sum().item()

        test_loss = running_loss / len(test_loader)
        test_accuracy = correct / total

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        remaining_time = epoch_duration * (epochs - (epoch + 1))

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f} | "
            f"Time: {epoch_duration:.2f} seconds | Time left: {remaining_time:.2f} seconds"
        )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    plot_training_results(
        epochs, train_losses, test_losses, train_accuracies, test_accuracies
    )

    return model


def plot_training_results(
    epochs, train_losses, test_losses, train_accuracies, test_accuracies
):
    epochs_range = range(1, epochs + 1)

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


def load_image(image_path):
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray,
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
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0]
    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
    return np.array(landmarks)


def create_3d_model(landmarks):
    points = np.array(landmarks)
    tri = Delaunay(points[:, :2])
    return points, tri


def plot_3d_model_with_texture(ax, image, points, tri, show_landmarks=False):
    ax.clear()
    h, w, _ = image.shape

    # Get the UV coordinates by scaling the points to the image dimensions
    uv_coords = np.array([(point[0] * w, point[1] * h) for point in points])

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
    for x, y, z in landmarks:
        cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)
    return image


def update_view(ax_from, ax_to):
    elev = ax_from.elev
    azim = ax_from.azim
    ax_to.view_init(elev=elev, azim=azim)


def on_move(event, ax3d_left, ax3d_right):
    if not event.inaxes:
        return
    for ax in [ax3d_left, ax3d_right]:
        if event.inaxes == ax:
            update_view(ax, ax3d_left if ax == ax3d_right else ax3d_right)
            fig.canvas.draw_idle()
            break


def rotate_points(points, elev, azim, roll):
    r = R.from_euler("xyz", [elev, azim, roll], degrees=True)
    return r.apply(points)


def save_rotated_points_to_file(rotated_points, filename, labels):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "w") as f:
        for points, label in zip(rotated_points, labels):
            line = f"{label} " + " ".join(f"{x},{y},{z}" for x, y, z in points)
            f.write(line + "\n")
    print(f"Rotated points saved to {filename}")


def load_rotated_points_from_file(filename):
    rotated_points = []
    labels = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            labels.append(parts[0])
            points = []
            for point_str in parts[1:]:
                x, y, z = map(float, point_str.split(","))
                points.append([x, y, z])
            rotated_points.append(np.array(points))
    print(f"Rotated points loaded from {filename}")
    return rotated_points, labels


def recognize_face(model, list_rotated_points_3D):
    points_flattened = np.array(
        [point_set.flatten() for point_set in list_rotated_points_3D]
    )

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(points_flattened, dtype=torch.float32).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        predicted_np = predicted.cpu().numpy()
        unique, counts = np.unique(predicted_np, return_counts=True)
        label_counts = dict(zip(unique, counts))

        print_green("Predicted Labels and their counts:")
        print("Label\tCount")
        for key, value in label_counts.items():
            print(f"{key}\t{value}")

        predicted_label = max(label_counts, key=label_counts.get)
        confidence = label_counts[predicted_label] / len(predicted) * 100

    return predicted_label, confidence


def process_image(image_path, face_mesh):
    print_green("Processing image...")
    image = load_image(image_path)
    face_region, bbox = detect_face(image)
    if face_region is not None:
        landmarks = get_face_landmarks(face_region, face_mesh)
        if landmarks is not None:
            points, tri = create_3d_model(landmarks)

            if SHOW_3D_PLOT:
                plot_3d_results(face_region, bbox, landmarks, points, tri, image)

            print("Processing completed.")
            print("-" * 50)
            # points = normalize_points(points)

            return points, tri, face_region
        else:
            print("No face landmarks detected in the image.")
            return None, None, None
    else:
        print("No face detected in the image.")
        return None, None, None


def plot_3d_results(face_region, bbox, landmarks, points, tri, image):
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


def rotate_3d_points(points, image, tri):
    print_green("Starting rotation model...")

    # Initial plot similar to the lower left square in plot_3d_results
    if SHOW_3D_PLOT:
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        plot_3d_model_with_texture(ax, image, points, tri, show_landmarks=True)
        plt.show(block=False)
        plt.pause(1)

    list_rotated_points = []

    for x_angle in range(-10, 110, 5):
        for y_angle in range(-50, 50, 5):
            for z_angle in range(-20, 20, 5):
                rotated_points = rotate_points(points, x_angle, y_angle, z_angle)
                list_rotated_points.append(rotated_points)
                if SHOW_3D_PLOT:
                    ax.clear()
                    rotated_tri = Delaunay(rotated_points[:, :2])
                    plot_3d_model_with_texture(
                        ax, image, points, tri, show_landmarks=True
                    )
                    ax.view_init(elev=-90 + y_angle, azim=270 + x_angle)
                    plt.draw()
                    plt.pause(0.001)

                    # Save the plot as an image
                    fig.savefig("temp_plot.png")

                    # Read the image and apply smoothing and sharpening filters
                    img = cv2.imread("temp_plot.png")

                    # Apply Gaussian blur to smooth the image
                    smoothed_img = cv2.GaussianBlur(img, (5, 5), 0)

                    # Create a sharpening kernel
                    sharpening_kernel = np.array(
                        [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
                    )

                    # Apply the sharpening filter
                    sharpened_img = cv2.filter2D(smoothed_img, -1, sharpening_kernel)

                    # Save the sharpened image
                    cv2.imwrite(
                        f"r/sharpened_plot{x_angle}_{y_angle}_{z_angle}.png",
                        sharpened_img,
                    )

                    # Display the sharpened image using matplotlib
                    sharpened_img_rgb = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB)
                    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
                    ax.imshow(sharpened_img_rgb)
                    plt.draw()
                    plt.pause(0.001)

    if SHOW_3D_PLOT:
        plt.show()

    print("Rotation completed.")
    print("-" * 50)

    return list_rotated_points


def plot_initial_3d_points(points):
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


def plot_rotated_3d_points(ax, rotated_points, image, tri):
    ax.clear()
    rotated_tri = Delaunay(rotated_points[:, :2])
    plot_3d_model_with_texture(
        ax, image, rotated_points, rotated_tri, show_landmarks=False
    )
    plt.draw()
    plt.pause(0.001)


def get_distances(points):
    vertical_dist = np.max(points[:, 1]) - np.min(points[:, 1])
    horizontal_dist = np.max(points[:, 0]) - np.min(points[:, 0])
    return vertical_dist, horizontal_dist


def align_face(points):
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

    # Translate the points so that the nose tip is at the origin
    aligned_points -= aligned_points[nose_index]

    return aligned_points


def normalize_points(points, target_distance=1.0, reference_index=8):
    points = points.astype(np.float64)

    # Align the points
    aligned_points = align_face(points)

    # Find the center and scale the points
    center = np.mean(aligned_points, axis=0)
    points_centered = aligned_points - center
    reference_point = points_centered[reference_index]
    current_distance = np.linalg.norm(reference_point)
    scale = target_distance / current_distance
    normalized_points = points_centered * scale

    if SHOW_3D_PLOT:
        ax = plot_initial_3d_points(normalized_points)
        plt.show()

    return normalized_points


def align_points(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.transform(points)


def compare_landmarks(image_path1, image_path2, face_mesh):
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


def process_folder_images(folder_path, face_mesh):
    image_landmarks = {}
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path):
            points = process_image(image_path, face_mesh)
            if points is not None:
                image_landmarks[image_path] = points
    return image_landmarks


def compare_landmarks_mse(landmarks1, landmarks2):
    mse = mean_squared_error(landmarks1, landmarks2)
    return mse


def POC_compare_landmarks(root_folder_path):
    print_green("Starting face mesh setup...")
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    )
    print("Face mesh setup completed.")

    all_landmarks = {}
    image_to_folder = {}
    for subfolder in os.listdir(root_folder_path):
        subfolder_path = os.path.join(root_folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            subfolder_landmarks = process_folder_images(subfolder_path, mp_face_mesh)
            all_landmarks.update(subfolder_landmarks)
            for image_path in subfolder_landmarks.keys():
                image_to_folder[image_path] = subfolder

    all_image_paths = list(all_landmarks.keys())
    all_image_paths.sort()

    num_images = len(all_image_paths)
    differences = np.zeros((num_images, num_images))
    colors = []

    for i in range(num_images):
        for j in range(num_images):
            landmarks1 = all_landmarks[all_image_paths[i]]
            landmarks2 = all_landmarks[all_image_paths[j]]
            mse = compare_landmarks_mse(landmarks1, landmarks2)
            differences[i, j] = mse
            differences[j, i] = mse
            if (
                image_to_folder[all_image_paths[i]]
                == image_to_folder[all_image_paths[j]]
            ):
                colors.append("red")
            else:
                colors.append("blue")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    xpos, ypos = np.meshgrid(np.arange(num_images), np.arange(num_images))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    dx = dy = np.ones_like(zpos)
    dz = differences.flatten()

    color_map = np.array(colors * (len(dz) // len(colors) + 1))[: len(dz)]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color_map, zsort="average")

    ax.set_xlabel("Image 1")
    ax.set_ylabel("Image 2")
    ax.set_zlabel("Mean Squared Error")
    ax.set_title("3D Histogram of Landmark Differences")

    ax.set_xticks(np.arange(num_images))
    ax.set_yticks(np.arange(num_images))
    ax.set_xticklabels(
        [os.path.basename(path) for path in all_image_paths], rotation=90, fontsize=8
    )
    ax.set_yticklabels([os.path.basename(path) for path in all_image_paths], fontsize=8)
    ax.set_ylim(num_images, 0)

    plt.show()


def main(folder_path):
    print_green("Starting face mesh setup...")
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    )
    print("Face mesh setup completed.")

    if not os.path.isdir(folder_path):
        raise ValueError("Provided path is not a directory")

    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    if CREATE_NEW_DATA:
        all_rotated_points = []
        labels = []

        for idx, image_path in enumerate(image_files):
            points, tri, face_region = process_image(image_path, mp_face_mesh)
            if points is not None:
                list_rotated_points_3D = rotate_3d_points(points, face_region, tri)
                all_rotated_points.extend(list_rotated_points_3D)
                labels.extend(
                    [os.path.basename(image_path)] * len(list_rotated_points_3D)
                )
            else:
                print(f"No landmarks detected in image: {image_path}")

        if all_rotated_points:
            # shuffle the data before saving all_rotated_points and labels in same order
            all_rotated_points, labels = (
                np.array(all_rotated_points),
                np.array(labels),
            )
            indices = np.arange(len(all_rotated_points))
            np.random.shuffle(indices)
            all_rotated_points, labels = (
                all_rotated_points[indices],
                labels[indices],
            )
            save_rotated_points_to_file(all_rotated_points, PATH_FOR_DATA, labels)
        else:
            print("No landmarks detected in any of the images.")
            return

    print(f"Loading rotated points from file: {PATH_FOR_DATA}")
    list_rotated_points, labels = load_rotated_points_from_file(PATH_FOR_DATA)
    print("Rotated points loaded successfully.")
    print(f"Size of list_rotated_points: {len(list_rotated_points)}")
    print(f"Size of list_rotated_points[0]: {list_rotated_points[0].shape}")
    print("*" * 50)

    if list_rotated_points:
        list_of_labels = []
        for label in labels:
            if label not in list_of_labels:
                list_of_labels.append(label)
        num_of_classes = len(list_of_labels)
        print(f"Number of classes: {num_of_classes}")
        labels_class = [list_of_labels.index(label) for label in labels]
        dict_labels = {list_of_labels[i]: i for i in range(num_of_classes)}
        print("-" * 50)
        print(f"Labels and their class number: {dict_labels}")
        print("-" * 50)
        model = train_face_recognition_model(
            list_rotated_points, labels_class, num_of_classes, epochs=100
        )

        if SAVE_MODEL:
            torch.save(model.state_dict(), PATH_FOR_MODEL)
            print(f"Model saved to {PATH_FOR_MODEL}")
        else:
            print("Loading model from file...")
            model = build_model(list_rotated_points[0].shape, num_of_classes).to(device)
            model.load_state_dict(torch.load(PATH_FOR_MODEL))
            model.eval()
            print("Model loaded successfully.")

        new_image_path = "./GG.jpeg"
        print(f"Loading and processing new image: {new_image_path}")
        new_landmarks_3D, tri, face_region = process_image(new_image_path, mp_face_mesh)
        if new_landmarks_3D is not None:
            list_rotated_points_3D = rotate_3d_points(
                new_landmarks_3D, face_region, tri
            )

            predicted_label, confidence = recognize_face(model, list_rotated_points_3D)
            print(
                f"Predicted Label: {predicted_label} -> {list_of_labels[predicted_label]}, Confidence: {confidence:.2f}%"
            )
            print("-" * 50)
            print("Labels and their class number:")
            for key, value in dict_labels.items():
                print(f"{key} -> {value}")
            print("-" * 50)
            image = cv2.imread(f"{folder_path}/{list_of_labels[predicted_label]}")
            # Display the image and the test image side by side for comparison
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Matched Image")
            axs[0].axis("off")
            axs[1].imshow(cv2.cvtColor(cv2.imread(new_image_path), cv2.COLOR_BGR2RGB))
            axs[1].set_title("Test Image")
            axs[1].axis("off")
            plt.show()
        else:
            print("No face landmarks detected in the new image.")
    else:
        print("No rotated points available for training.")


if __name__ == "__main__":
    print_green("Starting...")
    # print_green(f"Running on device: {device}")
    # folder_path = "./imgs"  # Change this to your folder path
    # main(folder_path)

    compare_landmarks(
        "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/comper_face/1/11.JPG",
        "/Users/lfarz/Desktop/לימודים/תואר שני/מעבדות מתקדמות בAI/face-blocker/face-blocker/comper_face/2/21.jpeg",
        mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        ),
    )

    POC_compare_landmarks("./comper_face")
