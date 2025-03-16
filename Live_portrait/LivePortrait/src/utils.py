
# Basic utils module for LivePortrait
import os
import numpy as np
import cv2
import torch

def get_device():
    """Get the device to use for PyTorch."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_image(path):
    """Load an image from a file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(path, img):
    """Save an image to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def resize_image(img, max_size):
    """Resize an image to a maximum size while preserving aspect ratio."""
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
