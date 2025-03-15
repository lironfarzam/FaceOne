#!/usr/bin/env python3
"""
FaceOne: LFW Dataset Downloader
===============================

This script downloads the Labeled Faces in the Wild (LFW) dataset and
prepares it for use as negative examples in the face recognition model.

The LFW dataset contains more than 13,000 images of faces collected from the web,
making it an excellent source of diverse negative examples.

Author: Liron Farzam
"""

import os
import sys
import json
import time
import shutil
import logging
import requests
import tarfile
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path to import config utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from cool_utils import load_config, print_green, print_red, print_blue
except ImportError:
    # Define basic print functions if cool_utils is not available
    def print_green(message):
        print(f"\033[92m{message}\033[0m")

    def print_red(message):
        print(f"\033[91m{message}\033[0m")

    def print_blue(message):
        print(f"\033[94m{message}\033[0m")

    def load_config(config_path=None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config.json",
            )

        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print_red(f"Error loading config from {config_path}: {e}")
            return {}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# LFW dataset URLs - updated with alternative sources
# Original URLs (no longer working)
# LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
# LFW_FUNNELED_URL = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"

# Updated URLs
LFW_URL = "https://www.dropbox.com/s/447wphuhn6gq8n4/lfw.tgz?dl=1"
LFW_FUNNELED_URL = "https://www.dropbox.com/s/e7rosp0gedp3v56/lfw-funneled.tgz?dl=1"

# Alternative URLs if the above fail
ALTERNATIVE_URLS = [
    "https://data.deepai.org/lfw.tgz",
    "https://conradsanderson.id.au/lfwcrop/lfw.tgz",
]

# Choose which version to download (regular or funneled)
DOWNLOAD_URL = LFW_URL  # Change to LFW_FUNNELED_URL if you prefer that version

# Kaggle dataset information for LFW
KAGGLE_DATASET = "jessicali9530/lfw-dataset"


def print_warning(message):
    """Print a warning message."""
    print(f"\033[93m{message}\033[0m")


def download_via_kagglehub(output_dir):
    """Download the LFW dataset using kagglehub."""
    try:
        print_blue("Attempting to download LFW dataset using kagglehub...")

        try:
            import kagglehub
        except ImportError:
            print_warning("kagglehub not installed. Installing now...")
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
            import kagglehub

        # Create temp directory for Kaggle downloads
        kaggle_dir = os.path.join(output_dir, "kaggle_download")
        os.makedirs(kaggle_dir, exist_ok=True)

        # Download the LFW dataset using kagglehub
        print_blue(f"Downloading dataset {KAGGLE_DATASET}...")

        # Download the dataset using the correct kagglehub API
        # The previous code used kagglehub.api.dataset_download_files which is incorrect
        path = kagglehub.dataset_download(KAGGLE_DATASET, force_download=True)

        print_green(f"Successfully downloaded LFW dataset to {path}")

        # Find the LFW directory in the downloaded content
        # Since kagglehub manages the path itself, we'll use the returned path
        lfw_dir = path

        if os.path.exists(lfw_dir):
            # Verify we have image files in the download
            image_count = 0
            for root, dirs, files in os.walk(lfw_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_count += 1
                        if image_count > 10:  # Just check a few to confirm it's working
                            break
                if image_count > 10:
                    break

            if image_count > 0:
                print_green(f"Found {image_count}+ images in the downloaded dataset")
                return (
                    lfw_dir,
                    True,
                )  # Return True as second value to indicate it's a kagglehub download
            else:
                print_warning("Downloaded directory exists but contains no images")
                # No need to remove it, we'll use the path as is

        # If we got here, we didn't find any images in the primary directory
        print_warning(
            "Looking for images in alternative directories within the download..."
        )
        for root, dirs, files in os.walk(lfw_dir):
            image_files = [
                f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if image_files:
                print_green(f"Found {len(image_files)} images in {root}")
                return (
                    root,
                    True,
                )  # Return True as second value to indicate it's a kagglehub download

        print_red("Could not find any face images in Kaggle download")
        return None, False

    except Exception as e:
        print_red(f"Error downloading via kagglehub: {str(e)}")
        return None, False


def download_file(url, destination):
    """Download a file with progress bar."""
    try:
        print_blue(f"Attempting to download from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Get file size
        total_size = int(response.headers.get("content-length", 0))

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Set up progress bar
        progress_bar = tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {os.path.basename(destination)}",
        )

        # Download file
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()
        return True

    except Exception as e:
        print_red(f"Error downloading {url}: {str(e)}")
        if os.path.exists(destination):
            os.remove(destination)
        return False


def try_download_with_alternatives(destination):
    """Try downloading from the primary URL and fall back to alternatives if needed."""
    # Try the primary URL first
    if download_file(DOWNLOAD_URL, destination):
        return True

    print_warning("Primary download failed. Trying alternative sources...")

    # Try alternative URLs
    for url in ALTERNATIVE_URLS:
        print_blue(f"Trying alternative URL: {url}")
        if download_file(url, destination):
            return True

    # If we get here, all downloads failed
    print_red(
        "All download attempts failed. Please check your internet connection or try again later."
    )
    return False


def extract_archive(archive_path, extract_path):
    """Extract a tar.gz archive."""
    try:
        print_blue(f"Extracting {os.path.basename(archive_path)}...")

        # Create extraction directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)

        # Extract the archive
        with tarfile.open(archive_path) as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting files"):
                tar.extract(member, path=extract_path)

        print_green(f"Successfully extracted to {extract_path}")
        return True

    except Exception as e:
        print_red(f"Error extracting {archive_path}: {str(e)}")
        return False


def copy_images_to_negative_path(lfw_dir, negative_path, max_images=None):
    """Copy LFW images to the negative examples directory."""
    try:
        print_blue(
            f"Copying LFW images to negative examples directory ({negative_path})..."
        )

        # Create negative path if it doesn't exist
        os.makedirs(negative_path, exist_ok=True)

        # Count number of images copied
        image_count = 0
        skipped_count = 0

        # Walk through the LFW directory
        for root, dirs, files in os.walk(lfw_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    source_path = os.path.join(root, file)
                    # Create a unique destination filename to avoid conflicts
                    dest_filename = f"lfw_{os.path.basename(root)}_{file}"
                    dest_path = os.path.join(negative_path, dest_filename)

                    # Copy the file
                    try:
                        shutil.copy2(source_path, dest_path)
                        image_count += 1

                        # Print progress periodically
                        if image_count % 500 == 0:
                            print_blue(f"Copied {image_count} images so far...")

                        # Stop if we've reached the maximum number of images
                        if max_images and image_count >= max_images:
                            print_blue(
                                f"Reached maximum number of images ({max_images})"
                            )
                            break

                    except Exception as e:
                        print_red(f"Error copying {source_path}: {str(e)}")
                        skipped_count += 1

            # Stop if we've reached the maximum number of images
            if max_images and image_count >= max_images:
                break

        print_green(f"Successfully copied {image_count} images to {negative_path}")
        if skipped_count > 0:
            print_red(f"Skipped {skipped_count} images due to errors")

        return image_count

    except Exception as e:
        print_red(f"Error copying images: {str(e)}")
        return 0


def main():
    """Main function to download LFW dataset and prepare it for negative examples."""
    print_blue("Starting LFW dataset download and preparation")
    start_time = time.time()

    # Load configuration
    config = load_config()

    if not config:
        print_red("Failed to load configuration")
        return 1

    # Check for negative path in config
    if "neg_path" not in config:
        print_red("Missing 'neg_path' in configuration")
        return 1

    negative_path = config["neg_path"]

    # Create temporary directory for downloads
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_lfw")
    os.makedirs(temp_dir, exist_ok=True)

    # First try using kagglehub (preferred method)
    kaggle_result = download_via_kagglehub(temp_dir)
    kaggle_lfw_dir = kaggle_result[0]
    is_kaggle_download = kaggle_result[1] if len(kaggle_result) > 1 else False

    if kaggle_lfw_dir:
        # Successfully downloaded via kagglehub
        print_green("Successfully downloaded LFW dataset via kagglehub")

        # Copy images to negative path
        image_count = copy_images_to_negative_path(kaggle_lfw_dir, negative_path)

        if image_count > 0:
            print_green(
                f"Successfully copied {image_count} images from kagglehub download"
            )

            # Clean up temporary files
            print_blue("Cleaning up downloaded files...")
            try:
                # Clean up kaggle files if possible
                if is_kaggle_download:
                    # For kagglehub downloads, we can try to clear cache
                    # But this might not be necessary as kagglehub manages its own cache
                    try:
                        import kagglehub

                        # Try to use the experimental clear_cache feature if available
                        if hasattr(kagglehub, "clear_cache"):
                            kagglehub.clear_cache()
                            print_green("Cleared kagglehub cache")
                    except Exception as e:
                        print_warning(f"Could not clear kagglehub cache: {str(e)}")

                # Delete the temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print_green("Temporary download directory removed")

            except Exception as e:
                print_red(f"Error during cleanup: {str(e)}")

            # Calculate execution time
            execution_time = time.time() - start_time
            minutes, seconds = divmod(execution_time, 60)

            print_green(
                f"LFW dataset preparation completed in {int(minutes)}m {int(seconds)}s"
            )
            print_green(f"Added {image_count} negative examples to {negative_path}")

            return 0
        else:
            print_warning(
                "Failed to copy images from kagglehub download. Trying alternative methods..."
            )
    else:
        print_warning("kagglehub download failed. Trying alternative methods...")

    # Fall back to traditional download methods if kagglehub fails
    # Download LFW dataset via direct URLs
    archive_path = os.path.join(temp_dir, "lfw.tgz")

    # Check if the existing archive file is valid
    if os.path.exists(archive_path):
        try:
            # Try opening the file to verify it's a valid tarfile
            with tarfile.open(archive_path) as test_tar:
                # Just try to read the first member to verify the file is valid
                members = test_tar.getmembers()
                if len(members) > 0:
                    print_blue(f"Existing archive at {archive_path} is valid")
                    download_success = True
                else:
                    print_warning("Existing archive appears empty, will re-download")
                    os.remove(archive_path)
                    download_success = False
        except Exception as e:
            print_warning(f"Existing archive is corrupt: {str(e)}")
            print_blue("Removing corrupt archive and downloading fresh copy")
            os.remove(archive_path)
            download_success = False
    else:
        download_success = False

    if not download_success:
        print_blue(f"Downloading LFW dataset using direct URLs...")
        download_success = try_download_with_alternatives(archive_path)
        if not download_success:
            print_red("Failed to download LFW dataset from all sources")
            if os.path.exists(negative_path) and os.listdir(negative_path):
                # If the negative path already has some images, we can continue
                print_warning(f"Using existing negative examples in {negative_path}")
                print_warning(
                    "Model quality may be affected by limited negative examples"
                )
                print_green(
                    f"Found {len(os.listdir(negative_path))} existing negative examples"
                )

                # Clean up temp directory
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        print_green("Temporary download directory removed")
                except Exception as e:
                    print_red(f"Error removing temporary files: {str(e)}")

                return 0  # Return success to continue the pipeline
            else:
                # Create minimal negative examples
                print_warning("Creating minimal synthetic negative examples...")
                create_synthetic_negative_examples(negative_path)

                # Clean up temp directory
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        print_green("Temporary download directory removed")
                except Exception as e:
                    print_red(f"Error removing temporary files: {str(e)}")

                return 0  # Return success to continue the pipeline

    # Continue with extraction if download was successful
    if download_success:
        # Extract the archive
        extract_path = os.path.join(temp_dir, "extracted")
        if not extract_archive(archive_path, extract_path):
            print_red("Failed to extract LFW dataset")
            # Clean up failed extraction
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print_green(
                        "Temporary download directory removed after failed extraction"
                    )
            except Exception as e:
                print_red(f"Error removing temporary files: {str(e)}")
            return 1

        # Find the LFW directory
        lfw_dir = None
        for item in os.listdir(extract_path):
            if item == "lfw" or item == "lfw-funneled":
                lfw_dir = os.path.join(extract_path, item)
                break

        if not lfw_dir:
            print_red("Could not find LFW directory in extracted files")
            # Clean up failed process
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print_green(
                        "Temporary download directory removed after failed directory lookup"
                    )
            except Exception as e:
                print_red(f"Error removing temporary files: {str(e)}")
            return 1

        # Copy images to negative path
        image_count = copy_images_to_negative_path(lfw_dir, negative_path)

        if image_count == 0:
            print_red("No images were copied to the negative examples directory")
            # Clean up failed process
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print_green(
                        "Temporary download directory removed after failed image copy"
                    )
            except Exception as e:
                print_red(f"Error removing temporary files: {str(e)}")
            return 1

        # Clean up all temporary files
        print_blue("Cleaning up all downloaded files...")
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print_green("All temporary files and downloads removed")
        except Exception as e:
            print_red(f"Error removing temporary files: {str(e)}")

        # Calculate execution time
        execution_time = time.time() - start_time
        minutes, seconds = divmod(execution_time, 60)

        print_green(
            f"LFW dataset preparation completed in {int(minutes)}m {int(seconds)}s"
        )
        print_green(f"Added {image_count} negative examples to {negative_path}")

    return 0


def create_synthetic_negative_examples(output_dir, num_examples=100):
    """Create synthetic face images to use as minimal negative examples."""
    try:
        import numpy as np
        from PIL import Image

        print_blue(f"Creating {num_examples} synthetic negative examples...")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create random face-like images
        image_size = 128
        created_count = 0

        for i in range(num_examples):
            # Create a random noise image
            # Using a normal distribution for more face-like appearance
            img_array = (
                np.random.normal(loc=128, scale=40, size=(image_size, image_size, 3))
                .clip(0, 255)
                .astype(np.uint8)
            )

            # Add some structure (oval face shape)
            y, x = np.ogrid[:image_size, :image_size]
            center = image_size / 2
            mask = (
                (x - center) ** 2 / (center * 0.8) ** 2
                + (y - center) ** 2 / (center * 1.0) ** 2
            ) <= 1

            # Apply mask to make it more face-like
            for c in range(3):
                img_array[:, :, c] = img_array[:, :, c] * mask

            # Add some facial feature suggestions
            # Eyes
            eye_y = int(image_size * 0.4)
            left_eye_x = int(image_size * 0.3)
            right_eye_x = int(image_size * 0.7)
            eye_size = int(image_size * 0.05)

            # Darker regions for eyes
            img_array[
                eye_y - eye_size : eye_y + eye_size,
                left_eye_x - eye_size : left_eye_x + eye_size,
            ] = (
                img_array[
                    eye_y - eye_size : eye_y + eye_size,
                    left_eye_x - eye_size : left_eye_x + eye_size,
                ]
                * 0.7
            )

            img_array[
                eye_y - eye_size : eye_y + eye_size,
                right_eye_x - eye_size : right_eye_x + eye_size,
            ] = (
                img_array[
                    eye_y - eye_size : eye_y + eye_size,
                    right_eye_x - eye_size : right_eye_x + eye_size,
                ]
                * 0.7
            )

            # Mouth
            mouth_y = int(image_size * 0.7)
            mouth_width = int(image_size * 0.4)
            mouth_height = int(image_size * 0.05)

            img_array[
                mouth_y - mouth_height : mouth_y + mouth_height,
                int(center - mouth_width / 2) : int(center + mouth_width / 2),
            ] = (
                img_array[
                    mouth_y - mouth_height : mouth_y + mouth_height,
                    int(center - mouth_width / 2) : int(center + mouth_width / 2),
                ]
                * 0.8
            )

            # Convert to PIL Image
            img = Image.fromarray(img_array)

            # Save image
            filename = os.path.join(output_dir, f"synthetic_face_{i+1:03d}.jpg")
            img.save(filename, quality=90)
            created_count += 1

            # Print progress periodically
            if (i + 1) % 10 == 0:
                print_blue(f"Created {i+1} synthetic examples...")

        print_green(f"Successfully created {created_count} synthetic negative examples")
        return created_count

    except Exception as e:
        print_red(f"Error creating synthetic examples: {str(e)}")

        # Create even simpler examples if PIL or numpy fails
        try:
            print_warning("Attempting to create very basic examples...")
            basic_count = create_basic_negative_examples(output_dir, num_examples=20)
            return basic_count
        except:
            print_red("Failed to create even basic negative examples")
            return 0


def create_basic_negative_examples(output_dir, num_examples=20):
    """Create very basic gradient images as a last resort for negative examples."""
    import os
    from PIL import Image, ImageDraw

    os.makedirs(output_dir, exist_ok=True)
    created = 0

    for i in range(num_examples):
        # Create a new image with a gradient background
        img = Image.new("RGB", (128, 128), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Draw a colored rectangle with gradient
        color1 = (i * 10 % 255, 100 + i * 5 % 155, 150 + i * 7 % 105)
        color2 = (200 - i * 10 % 200, 50 + i * 3 % 200, 100 + i * 9 % 155)

        for y in range(128):
            # Simple gradient
            r = int(color1[0] + (color2[0] - color1[0]) * y / 128)
            g = int(color1[1] + (color2[1] - color1[1]) * y / 128)
            b = int(color1[2] + (color2[2] - color1[2]) * y / 128)
            draw.line([(0, y), (128, y)], fill=(r, g, b))

        # Add a simple oval to suggest a face
        draw.ellipse([20, 20, 108, 108], outline=(100, 100, 100))

        # Add simple eyes and mouth
        draw.ellipse([40, 45, 50, 55], fill=(50, 50, 50))
        draw.ellipse([78, 45, 88, 55], fill=(50, 50, 50))
        draw.arc([40, 65, 88, 95], start=0, end=180, fill=(50, 50, 50), width=2)

        filename = os.path.join(output_dir, f"basic_face_{i+1:02d}.jpg")
        img.save(filename)
        created += 1

    print_green(f"Created {created} basic negative examples")
    return created


if __name__ == "__main__":
    sys.exit(main())
