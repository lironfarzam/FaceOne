"""
Basic utility functions for LivePortrait.
"""

import os
import numpy as np
import cv2
import torch
from typing import Tuple, Optional, Union


def get_device():
    """
    Get the device to use for PyTorch.

    Returns:
        torch.device: The device to use.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_image(path: str) -> np.ndarray:
    """
    Load an image from a file.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: The loaded image in RGB format.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image could not be loaded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: str, img: np.ndarray) -> None:
    """
    Save an image to a file.

    Args:
        path (str): Path to save the image to.
        img (np.ndarray): The image to save in RGB format.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def resize_image(img: np.ndarray, max_size: int) -> np.ndarray:
    """
    Resize an image to a maximum size while preserving aspect ratio.

    Args:
        img (np.ndarray): The image to resize.
        max_size (int): The maximum size of the larger dimension.

    Returns:
        np.ndarray: The resized image.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img

    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)

    return cv2.resize(img, (new_w, new_h))


def pad_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Pad an image to a target size.

    Args:
        img (np.ndarray): The image to pad.
        target_size (Tuple[int, int]): The target size (height, width).

    Returns:
        np.ndarray: The padded image.
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size

    # Calculate padding
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    # Pad the image
    padded = cv2.copyMakeBorder(
        img,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_w // 2,
        pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    return padded


def crop_center(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop the center of an image to a target size.

    Args:
        img (np.ndarray): The image to crop.
        target_size (Tuple[int, int]): The target size (height, width).

    Returns:
        np.ndarray: The cropped image.
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size

    # Calculate crop coordinates
    start_h = max(0, (h - target_h) // 2)
    start_w = max(0, (w - target_w) // 2)

    # Crop the image
    cropped = img[start_h : start_h + target_h, start_w : start_w + target_w]

    return cropped


def normalize_image(
    img: np.ndarray,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """
    Normalize an image for neural network input.

    Args:
        img (np.ndarray): The image to normalize (0-255 RGB).
        mean (Tuple[float, float, float]): Mean values for normalization.
        std (Tuple[float, float, float]): Standard deviation values for normalization.

    Returns:
        np.ndarray: The normalized image.
    """
    # Convert to float and scale to 0-1
    img = img.astype(np.float32) / 255.0

    # Normalize
    img = (img - np.array(mean)) / np.array(std)

    return img


def denormalize_image(
    img: np.ndarray,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """
    Denormalize an image from neural network output.

    Args:
        img (np.ndarray): The normalized image.
        mean (Tuple[float, float, float]): Mean values used for normalization.
        std (Tuple[float, float, float]): Standard deviation values used for normalization.

    Returns:
        np.ndarray: The denormalized image (0-255 RGB).
    """
    # Denormalize
    img = img * np.array(std) + np.array(mean)

    # Scale to 0-255 and convert to uint8
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    return img
