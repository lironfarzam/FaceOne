import os
import cv2
from deepface import DeepFace
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from scipy.spatial.distance import (
    minkowski,
    hamming,
    correlation,
    jensenshannon,
    mahalanobis,
    directed_hausdorff,
)


SHOW_3D_PLOT = False


def print_green(text):
    print("\033[92m" + text + "\033[0m")


def print_red(text):
    print("\033[91m" + text + "\033[0m")


def get_embedding(image_path: str, model_name: str = "ArcFace") -> np.ndarray:
    """Get the embedding vector for the given image.

    Args:
        image_path (str): Path to the image file.
        model_name (str, optional): Name of the model to use for embedding extraction. Defaults to "Facenet".
        apply_augmentation (bool, optional): Whether to apply augmentation to the image. Defaults to False.

    Returns:
        np.ndarray: Embedding vector for the image.
    """

    try:
        if not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")

        # Read image and ensure it's in the correct format
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Ensure image has 3 channels
        if len(img.shape) != 3:
            raise ValueError(f"Image must have 3 channels: {image_path}")

        # Convert to RGB for processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # # Resize image to expected dimensions (160x160 for FaceNet)
        # img = cv2.resize(img, (160, 160))

        # Get embedding using DeepFace
        try:
            with tf.device("/CPU:0"):  # Force CPU usage for DeepFace
                embedding = DeepFace.represent(
                    img_path=img,
                    model_name=model_name,
                    enforce_detection=True,
                    # detector_backend="opencv",  # Use a simpler detector
                )
            return embedding[0]["embedding"]
        except Exception as e:
            print_red(f"DeepFace error for {image_path}: {str(e)}")
            return None

    except Exception as e:
        # print_red(f"Error processing {image_path}: {str(e)}")
        return None

    finally:
        cv2.destroyAllWindows()


def compare_mse(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between two vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.

    Returns:
        float: Scaled MSE (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        MSE = ((a1 - b1)² + (a2 - b2)² + (a3 - b3)²) / len(data1)

    """
    mse = mean_squared_error(data1, data2)
    return mse


def compare_euclidean(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.

    Returns:
        float: Scaled Euclidean distance (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Euclidean distance = sqrt((a1 - b1)² + (a2 - b2)² + (a3 - b3)²)

    """
    euclidean_distance = np.linalg.norm(data1 - data2)
    return euclidean_distance


def compare_manhattan(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Manhattan distance (L1 norm) between two vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.

    Returns:
        float: Scaled Manhattan distance (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Manhattan distance = |a1 - b1| + |a2 - b2| + |a3 - b3|
    """
    manhattan_distance = np.sum(np.abs(data1 - data2))
    return manhattan_distance


def compare_cosine(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Cosine distance (1 - cosine similarity) between two vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.

    Returns:
        float: Scaled cosine distance (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Cosine distance = 1 - cosine similarity
        cosine similarity = (a1*b1 + a2*b2 + a3*b3) / sqrt((a1*a1 + a2*a2 + a3*a3) * (b1*b1 + b2*b2 + b3*b3))
    """
    similarity = cosine_similarity([data1], [data2])[0][0]
    cosine_difference = 1 - similarity
    return cosine_difference


def compare_chebyshev(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Chebyshev distance (maximum absolute difference) between two vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.

    Returns:
        float: Scaled Chebyshev distance (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Chebyshev distance = max(|a1 - b1|, |a2 - b2|, |a3 - b3|)
    """
    chebyshev_distance = np.max(np.abs(data1 - data2))
    return chebyshev_distance


def compare_canberra(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Canberra distance between two vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.

    Returns:
        float: Scaled Canberra distance (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Canberra distance = sum(abs(a1 - b1) / (abs(a1) + abs(b1)))

    """
    canberra_distance = np.sum(np.abs(data1 - data2) / (np.abs(data1) + np.abs(data2)))
    return canberra_distance


def compare_braycurtis(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Bray-Curtis distance between two vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.

    Returns:
        float: Scaled Bray-Curtis distance (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Bray-Curtis distance = sum(abs(a1 - b1)) / sum(abs(a1 + b1))
    """
    braycurtis_distance = np.sum(np.abs(data1 - data2)) / np.sum(np.abs(data1 + data2))
    return braycurtis_distance


def compare_hausdorff(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Hausdorff distance between two vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.

    Returns:
        float: Scaled Hausdorff distance (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Hausdorff distance = max(hausdorff([a1, a2, a3], [b1, b2, b3]), hausdorff([b1, b2, b3], [a1, a2, a3]))
        hausdorff([a1, a2, a3], [b1, b2, b3]) = max(max(abs(a1 - b1), abs(a2 - b2), abs(a3 - b3)), max(abs(b1 - a1), abs(b2 - a2), abs(b3 - a3)))

    """
    forward_hausdorff = directed_hausdorff([data1], [data2])[0]
    backward_hausdorff = directed_hausdorff([data2], [data1])[0]
    hausdorff_distance = max(forward_hausdorff, backward_hausdorff)
    return hausdorff_distance


def compare_minkowski(data1: np.ndarray, data2: np.ndarray, p=3) -> float:
    """
    Calculate the Minkowski distance with parameter p between two vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.
        p (int): The power parameter.

    Returns:
        float: Scaled Minkowski distance (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Minkowski distance = sum(abs(a1 - b1) ^ p) ^ (1/p)
    """
    return minkowski(data1, data2, p)


def compare_hamming(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Hamming distance between two binary vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.

    Returns:
        float: Scaled Hamming distance (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Hamming distance = sum(abs(a1 - b1))

    """
    return hamming(data1 > 0.5, data2 > 0.5) * len(data1)


def compare_correlation(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Correlation distance between two vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.

    Returns:
        float: Scaled Correlation distance (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Correlation distance = 1 - correlation
        correlation = (a1*b1 + a2*b2 + a3*b3) / sqrt((a1*a1 + a2*a2 + a3*a3) * (b1*b1 + b2*b2 + b3*b3))

    """
    return correlation(data1, data2)


def compare_jensenshannon(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Jensen-Shannon divergence between two probability distributions.

    Args:
        data1 (np.ndarray): First vector (normalized to sum to 1).
        data2 (np.ndarray): Second vector (normalized to sum to 1).

    Returns:
        float: Scaled Jensen-Shannon divergence (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Jensen-Shannon divergence = 1 - Jensen-Shannon similarity
        Jensen-Shannon similarity = (a1*b1 + a2*b2 + a3*b3) / sqrt((a1*a1 + a2*a2 + a3*a3) * (b1*b1 + b2*b2 + b3*b3))
    """
    return jensenshannon(data1, data2)


def compare_mahalanobis(data1: np.ndarray, data2: np.ndarray, inv_cov=None) -> float:
    """
    Calculate the Mahalanobis distance between two vectors.

    Args:
        data1 (np.ndarray): First vector.
        data2 (np.ndarray): Second vector.
        inv_cov (np.ndarray): Inverse covariance matrix.

    Returns:
        float: Scaled Mahalanobis distance (multiplied by 1000).

    Example:
        data1 = [a1, a2, a3] and data2 = [b1, b2, b3]
        Mahalanobis distance = sqrt((a1 - b1) * inv_cov * (a1 - b1) + (a2 - b2) * inv_cov * (a2 - b2) + (a3 - b3) * inv_cov * (a3 - b3))
        inv_cov = (1 / (a1*a1 + a2*a2 + a3*a3))

    """
    if inv_cov is None:
        raise ValueError("Inverse covariance matrix (inv_cov) must be provided.")
    return mahalanobis(data1, data2, inv_cov)


def process_folder_images(folder_path, model):
    image_data = {}
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path):
            points = get_embedding(image_path, model_name=model)
            if points is not None:
                image_data[image_path] = points

    if len(image_data) == 0:
        print_red(f"No images found in {folder_path}")

    return image_data


def process_all_images(root_folder_path, model):
    all_data = {}
    image_to_folder = {}
    for subfolder in os.listdir(root_folder_path):
        subfolder_path = os.path.join(root_folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            subfolder_data = process_folder_images(subfolder_path, model)
            all_data.update(subfolder_data)
            for image_path in subfolder_data.keys():
                image_to_folder[image_path] = subfolder

    return all_data, image_to_folder


def compute_differences(
    all_data, image_to_folder, comparison_functions, all_image_paths
):
    print_green("Computing differences for all metrics...")
    num_images = len(all_image_paths)
    differences = {
        metric: np.zeros((num_images, num_images))
        for metric in comparison_functions.keys()
    }

    same_folder_diffs = {metric: [] for metric in comparison_functions.keys()}
    different_folder_diffs = {metric: [] for metric in comparison_functions.keys()}

    for i in range(num_images):
        for j in range(num_images):
            if i != j:
                # Ensure data1 and data2 are numpy arrays
                data1 = np.array(all_data[all_image_paths[i]])
                data2 = np.array(all_data[all_image_paths[j]])

                for metric, func in comparison_functions.items():
                    try:
                        difference = func(data1, data2)
                        differences[metric][i, j] = difference
                        differences[metric][j, i] = difference
                        if (
                            image_to_folder[all_image_paths[i]]
                            == image_to_folder[all_image_paths[j]]
                        ):
                            same_folder_diffs[metric].append(difference)
                        else:
                            different_folder_diffs[metric].append(difference)
                    except Exception as e:
                        print_red(
                            f"Error computing {metric} for images {i} and {j}: {e}"
                        )
    return differences, same_folder_diffs, different_folder_diffs


def plot_3d_histograms_table(
    differences,
    comparison_functions,
    all_image_paths,
    image_to_folder,
    model_name,
):
    """Plot 3D histograms in a table layout with color coding for same/different folder differences.

    Args:
        differences (dict): Differences for all metrics.
        comparison_functions (dict): Dictionary of comparison functions.
        all_image_paths (list): List of image paths used for comparison.
        image_to_folder (dict): Mapping of image paths to their respective folders.
        model_name (str): Name of the model being compared.
    """
    num_metrics = len(comparison_functions)
    num_images = len(all_image_paths)
    cols = 4  # Number of plots per row
    rows = (num_metrics + cols - 1) // cols  # Calculate required rows

    fig = plt.figure(figsize=(15, 5 * rows), constrained_layout=True)
    metric_names = list(comparison_functions.keys())

    # Calculate global z-axis limits
    global_min = float("inf")
    global_max = float("-inf")
    for metric in metric_names:
        dz = differences[metric].flatten()
        global_min = min(global_min, dz.min())
        global_max = max(global_max, dz.max())

    # Ensure z-axis has consistent limits across all plots
    zlim = (global_min, global_max)

    for idx, metric in enumerate(metric_names):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")

        # Prepare 3D histogram data
        xpos, ypos = np.meshgrid(np.arange(num_images), np.arange(num_images))
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)
        dx = dy = np.ones_like(zpos)
        dz = differences[metric].flatten()

        # Determine color based on folder comparison
        colors = []
        for i, j in zip(xpos, ypos):
            same_folder = (
                image_to_folder[all_image_paths[i]]
                == image_to_folder[all_image_paths[j]]
            )
            colors.append("blue" if same_folder else "red")

        # 3D bar plot
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort="average")

        # Set consistent z-axis limits
        ax.set_zlim(zlim)

        # Labels and titles
        ax.set_xlabel("Image 1", fontsize=8)
        ax.set_ylabel("Image 2", fontsize=8)
        ax.set_zlabel("Difference", fontsize=8)
        ax.set_title(f"3D Histogram - {metric}", fontsize=10)

        # Adjust ticks for readability
        ax.set_xticks(np.arange(num_images))
        ax.set_yticks(np.arange(num_images))
        ax.set_xticklabels(
            [os.path.basename(path) for path in all_image_paths],
            rotation=90,
            fontsize=6,
        )
        ax.set_yticklabels(
            [os.path.basename(path) for path in all_image_paths],
            fontsize=6,
        )

    plt.suptitle(f"3D Histograms of Comparison Metrics {model_name}", fontsize=12)
    plt.show()

    # Save the figure
    fig.savefig(f"3d_histograms/{model_name}.png")


def summarize_comparison_statistics(
    differences,
    comparison_functions,
    same_folder_diffs,
    different_folder_diffs,
    model_name,
):
    """
    Summarize and display statistics for all comparison metrics.

    Args:
        differences (dict): Differences for all metrics.
        comparison_functions (dict): Dictionary of comparison functions.
        same_folder_diffs (dict): Differences for same-folder comparisons.
        different_folder_diffs (dict): Differences for different-folder comparisons.
        model_name (str): Name of the model being evaluated.

    Returns:
        pd.DataFrame: DataFrame with detailed statistics for all metrics.
    """
    # Initialize a DataFrame to store statistics
    columns = [
        "Model",
        "Metric",
        "Mean (Same Folder)",
        "Mean (Different Folder)",
        "Std (Same Folder)",
        "Std (Different Folder)",
        "Min (Same Folder)",
        "Max (Same Folder)",
        "Min (Different Folder)",
        "Max (Different Folder)",
        "Separation (Mean Diff)",
        "Dynamic Range (Same Folder)",
        "Dynamic Range (Different Folder)",
        "SNR",
        "Kurtosis (Same Folder)",
        "Kurtosis (Different Folder)",
        "Skewness (Same Folder)",
        "Skewness (Different Folder)",
        "Overlap Percentage",
        "Separation Ratio",
        "Median (Same Folder)",
        "Median (Different Folder)",
        "Suggested Threshold",
        "False Positive Rate",
        "False Negative Rate",
    ]
    data = []

    # Calculate statistics for each metric
    for metric, func in comparison_functions.items():
        # Convert same-folder and different-folder differences to NumPy arrays
        same_diffs = np.array(same_folder_diffs[metric])
        diff_diffs = np.array(different_folder_diffs[metric])

        if len(same_diffs) == 0 or len(diff_diffs) == 0:
            continue

        # Calculate basic statistics
        mean_same = np.mean(same_diffs)
        mean_diff = np.mean(diff_diffs)
        std_same = np.std(same_diffs)
        std_diff = np.std(diff_diffs)
        min_same = np.min(same_diffs)
        max_same = np.max(same_diffs)
        min_diff = np.min(diff_diffs)
        max_diff = np.max(diff_diffs)
        separation = mean_diff - mean_same

        # Dynamic Range
        dynamic_range_same = max_same - min_same
        dynamic_range_diff = max_diff - min_diff

        # Signal-to-Noise Ratio (SNR)
        snr = separation / (std_same + std_diff)

        # Kurtosis and Skewness
        from scipy.stats import kurtosis, skew

        kurtosis_same = kurtosis(same_diffs)
        kurtosis_diff = kurtosis(diff_diffs)
        skewness_same = skew(same_diffs)
        skewness_diff = skew(diff_diffs)

        # Overlap Percentage (Approximate as Intersection of PDFs)
        overlap_percentage = compute_overlap(same_diffs, diff_diffs)

        # Separation Ratio
        separation_ratio = separation / (dynamic_range_same + dynamic_range_diff)

        # Median and Interquartile Range (IQR)
        median_same = np.median(same_diffs)
        median_diff = np.median(diff_diffs)

        # Suggested Threshold
        suggested_threshold = (mean_same + mean_diff) / 2

        # False Positive and False Negative Rates
        false_positive_rate, false_negative_rate = compute_classification_rates(
            same_diffs, diff_diffs, suggested_threshold
        )

        # Add to the data
        data.append(
            [
                model_name,
                metric,
                mean_same,
                mean_diff,
                std_same,
                std_diff,
                min_same,
                max_same,
                min_diff,
                max_diff,
                separation,
                dynamic_range_same,
                dynamic_range_diff,
                snr,
                kurtosis_same,
                kurtosis_diff,
                skewness_same,
                skewness_diff,
                overlap_percentage,
                separation_ratio,
                median_same,
                median_diff,
                suggested_threshold,
                false_positive_rate,
                false_negative_rate,
            ]
        )

    # Create the DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Sort by Separation (Mean Diff) for better ranking
    df_sorted = df.sort_values(by="Separation (Mean Diff)", ascending=False)

    # Save DataFrame to a CSV file; add header only if the file doesn't exist
    if not os.path.exists("statistics.csv"):
        df_sorted.to_csv("statistics.csv", index=False)
    else:
        df_sorted.to_csv("statistics.csv", index=False, mode="a", header=False)

    return df_sorted


def compute_overlap(same_diffs: np.ndarray, diff_diffs: np.ndarray) -> float:
    """
    Compute the overlap percentage between KDE distributions of same-folder and different-folder distances.

    Args:
        same_diffs (np.ndarray): Differences for same-folder pairs.
        diff_diffs (np.ndarray): Differences for different-folder pairs.

    Returns:
        float: Overlap percentage (0 to 100).

    Raises:
        ValueError: If the input arrays contain NaNs, Infs, or are not valid.
    """
    # Remove invalid values (NaN or Inf)
    same_diffs = same_diffs[np.isfinite(same_diffs)]
    diff_diffs = diff_diffs[np.isfinite(diff_diffs)]

    if len(same_diffs) < 2 or len(diff_diffs) < 2:
        # print_red("Error: Not enough valid data points for KDE computation.")
        return 100.0  # Assume maximum overlap if data is invalid

    # Dimensionality reduction if necessary
    if same_diffs.ndim > 1 or diff_diffs.ndim > 1:
        pca = PCA(n_components=1)
        same_diffs = pca.fit_transform(same_diffs.reshape(-1, 1)).flatten()
        diff_diffs = pca.fit_transform(diff_diffs.reshape(-1, 1)).flatten()

    # Compute KDE for both distributions
    try:
        kde_same = gaussian_kde(same_diffs)
        kde_diff = gaussian_kde(diff_diffs)
    except np.linalg.LinAlgError as e:
        print_red(f"KDE failed due to linear algebra error: {e}")
        return 100.0  # Assume maximum overlap in case of failure

    # Define a common range for the two distributions
    min_range = min(same_diffs.min(), diff_diffs.min())
    max_range = max(same_diffs.max(), diff_diffs.max())
    x = np.linspace(min_range, max_range, 1000)

    # Evaluate KDEs
    kde_same_values = kde_same(x)
    kde_diff_values = kde_diff(x)

    # Calculate overlap percentage
    overlap_area = np.minimum(kde_same_values, kde_diff_values).sum() * (x[1] - x[0])
    return overlap_area * 100.0  # Convert to percentage


def compute_classification_rates(same_diffs, diff_diffs, threshold):
    """
    Compute the false positive and false negative rates based on a classification threshold.

    Args:
        same_diffs (np.ndarray): Same-folder differences.
        diff_diffs (np.ndarray): Different-folder differences.
        threshold (float): Classification threshold.

    Returns:
        tuple: False positive rate and false negative rate.
    """
    false_positives = np.sum(same_diffs > threshold) / len(same_diffs)
    false_negatives = np.sum(diff_diffs <= threshold) / len(diff_diffs)
    return false_positives * 100, false_negatives * 100


def find_best_model_and_metric(statistics_file: str) -> pd.DataFrame:
    """
    Load the comparison statistics file and find the best model and metric based on
    maximizing the separation between different-folder images and minimizing the distance
    within same-folder images.

    Args:
        statistics_file (str): Path to the CSV file containing comparison statistics.

    Returns:
        pd.DataFrame: DataFrame sorted by the best model-metric combinations.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(statistics_file)

        # Validate the file structure
        required_columns = [
            "Model",
            "Metric",
            "Mean (Same Folder)",
            "Mean (Different Folder)",
            "Std (Same Folder)",
            "Std (Different Folder)",
            "Min (Same Folder)",
            "Max (Same Folder)",
            "Min (Different Folder)",
            "Max (Different Folder)",
            "Separation (Mean Diff)",
            "Dynamic Range (Same Folder)",
            "Dynamic Range (Different Folder)",
            "SNR",
            "Kurtosis (Same Folder)",
            "Kurtosis (Different Folder)",
            "Skewness (Same Folder)",
            "Skewness (Different Folder)",
            "Overlap Percentage",
            "Separation Ratio",
            "Median (Same Folder)",
            "Median (Different Folder)",
            "Suggested Threshold",
            "False Positive Rate",
            "False Negative Rate",
        ]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"Invalid CSV structure. Expected columns: {required_columns}"
            )

        # Calculate a composite score for ranking
        df["Composite Score"] = (
            df["Separation (Mean Diff)"]  # Larger is better
            + df["Separation Ratio"]  # Larger is better
            + df["SNR"]  # Larger is better
            - 0.5 * df["Std (Same Folder)"]  # Smaller is better
            - 0.5 * df["Std (Different Folder)"]  # Smaller is better
            - 0.2 * df["Overlap Percentage"]  # Smaller is better
        )

        # Sort by Composite Score in descending order
        sorted_df = df.sort_values(by="Composite Score", ascending=False)

        # Display the best model and metric
        best_combination = sorted_df.iloc[0]
        print("\nBest Model-Metric Combination:")
        print(f"Model: {best_combination['Model']}")
        print(f"Metric: {best_combination['Metric']}")
        print(f"Composite Score: {best_combination['Composite Score']:.4f}")
        print(
            f"Separation (Mean Diff): {best_combination['Separation (Mean Diff)']:.4f}"
        )
        print(f"Separation Ratio: {best_combination['Separation Ratio']:.4f}")
        print(f"Mean (Same Folder): {best_combination['Mean (Same Folder)']:.4f}")
        print(
            f"Mean (Different Folder): {best_combination['Mean (Different Folder)']:.4f}"
        )
        print(f"Std (Same Folder): {best_combination['Std (Same Folder)']:.4f}")
        print(
            f"Std (Different Folder): {best_combination['Std (Different Folder)']:.4f}"
        )
        print(
            f"Dynamic Range (Same Folder): {best_combination['Dynamic Range (Same Folder)']:.4f}"
        )
        print(
            f"Dynamic Range (Different Folder): {best_combination['Dynamic Range (Different Folder)']:.4f}"
        )
        print(f"SNR: {best_combination['SNR']:.4f}")
        print(f"Overlap Percentage: {best_combination['Overlap Percentage']:.2f}%")
        print(f"False Positive Rate: {best_combination['False Positive Rate']:.2f}%")
        print(f"False Negative Rate: {best_combination['False Negative Rate']:.2f}%")
        print(
            f"Kurtosis (Same Folder): {best_combination['Kurtosis (Same Folder)']:.4f}"
        )
        print(
            f"Kurtosis (Different Folder): {best_combination['Kurtosis (Different Folder)']:.4f}"
        )
        print(
            f"Skewness (Same Folder): {best_combination['Skewness (Same Folder)']:.4f}"
        )
        print(
            f"Skewness (Different Folder): {best_combination['Skewness (Different Folder)']:.4f}"
        )
        print(f"Suggested Threshold: {best_combination['Suggested Threshold']:.4f}")

        # Save the sorted results for later analysis
        sorted_df.to_csv("optimized_comparison_statistics.csv", index=False)

        # Return the sorted DataFrame
        return sorted_df

    except FileNotFoundError:
        print(f"Error: The file '{statistics_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    print_green("Starting...")

    # Root folder path
    root_folder_path = "./comper_face"

    # Comparison functions dictionary
    comparison_functions = {
        "mse": compare_mse,
        "euclidean": compare_euclidean,
        "manhattan": compare_manhattan,
        "cosine": compare_cosine,
        "chebyshev": compare_chebyshev,
        "canberra": compare_canberra,
        "braycurtis": compare_braycurtis,
        "hausdorff": compare_hausdorff,
        "minkowski": lambda d1, d2: compare_minkowski(d1, d2, p=3),  # Default p=3
        "hamming": compare_hamming,
        "correlation": compare_correlation,
        "jensenshannon": compare_jensenshannon,
    }

    # List of all models
    models = [
        # "DeepFace", # ! DeepFace model requires LocallyConnected2D but it is no longer supported after tf 2.12 but you have 2.17. You need to downgrade your tf.
        "ArcFace",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "VGG-Face",
        "SFace",
    ]

    for model in models:

        # Process all images
        print_green(f"Processing all images using {model=}...")
        all_data, image_to_folder = process_all_images(root_folder_path, model)
        all_image_paths = list(all_data.keys())
        all_image_paths.sort()
        print_green("Done processing all images.")
        print("-" * 50)

        # Compute differences for all metrics
        print_green("Computing differences for all metrics...")
        differences, same_folder_diffs, different_folder_diffs = compute_differences(
            all_data, image_to_folder, comparison_functions, all_image_paths
        )
        print_green("Done computing differences for all metrics.")
        print("-" * 50)

        if SHOW_3D_PLOT:
            plot_3d_histograms_table(
                differences,
                comparison_functions,
                all_image_paths,
                image_to_folder,
                model_name=model,
            )

        print_green("Starting statistics summary...")

        # Generate and display the statistics table
        comparison_statistics = summarize_comparison_statistics(
            differences,
            comparison_functions,
            same_folder_diffs,
            different_folder_diffs,
            model_name=model,
        )

    print_green("Done statistics summary.")
    print("-" * 50)

    # Find the best metric

    sorted_stats = find_best_model_and_metric("./statistics.csv")

    # Optionally display the top 5 metrics for further analysis
    print("\nTop 5 Metrics:")
    print(sorted_stats.head(5).to_string(index=False))
