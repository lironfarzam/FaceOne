#!/usr/bin/env python3

import os
import sys
import requests
import shutil
from tqdm import tqdm
import argparse
import subprocess


def download_file(url, local_path):
    """
    Download a file from a URL to a local path with a progress bar.

    Args:
        url (str): The URL to download from
        local_path (str): The local path to save the file to
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        # Get the file size
        response = requests.head(url, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        file_size = int(response.headers.get("content-length", 0))

        # Download the file with a progress bar
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            print(
                f"Downloading {os.path.basename(local_path)} ({file_size / 1024 / 1024:.2f} MB)"
            )
            with open(local_path, "wb") as f, tqdm(
                desc=local_path,
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {str(e)}")
        # Remove partially downloaded file if it exists
        if os.path.exists(local_path):
            os.remove(local_path)
        return False


def download_liveportrait_models():
    """Download the LivePortrait models from multiple possible sources."""

    # Define model files and their potential sources
    model_files = {
        # LivePortrait models
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/landmark.onnx": [
            "https://huggingface.co/lfarz/FaceOne/resolve/main/liveportrait/landmark.onnx",
            "https://github.com/lironfarzam/FaceOne-models/raw/main/liveportrait/landmark.onnx",
        ],
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth": [
            "https://huggingface.co/lfarz/FaceOne/resolve/main/liveportrait/base_models/appearance_feature_extractor.pth",
            "https://github.com/lironfarzam/FaceOne-models/raw/main/liveportrait/base_models/appearance_feature_extractor.pth",
        ],
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/motion_extractor.pth": [
            "https://huggingface.co/lfarz/FaceOne/resolve/main/liveportrait/base_models/motion_extractor.pth",
            "https://github.com/lironfarzam/FaceOne-models/raw/main/liveportrait/base_models/motion_extractor.pth",
        ],
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/spade_generator.pth": [
            "https://huggingface.co/lfarz/FaceOne/resolve/main/liveportrait/base_models/spade_generator.pth",
            "https://github.com/lironfarzam/FaceOne-models/raw/main/liveportrait/base_models/spade_generator.pth",
        ],
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/warping_module.pth": [
            "https://huggingface.co/lfarz/FaceOne/resolve/main/liveportrait/base_models/warping_module.pth",
            "https://github.com/lironfarzam/FaceOne-models/raw/main/liveportrait/base_models/warping_module.pth",
        ],
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth": [
            "https://huggingface.co/lfarz/FaceOne/resolve/main/liveportrait/retargeting_models/stitching_retargeting_module.pth",
            "https://github.com/lironfarzam/FaceOne-models/raw/main/liveportrait/retargeting_models/stitching_retargeting_module.pth",
        ],
        # Insightface models
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx": [
            "https://huggingface.co/lfarz/FaceOne/resolve/main/insightface/models/buffalo_l/2d106det.onnx",
            "https://github.com/lironfarzam/FaceOne-models/raw/main/insightface/models/buffalo_l/2d106det.onnx",
        ],
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx": [
            "https://huggingface.co/lfarz/FaceOne/resolve/main/insightface/models/buffalo_l/det_10g.onnx",
            "https://github.com/lironfarzam/FaceOne-models/raw/main/insightface/models/buffalo_l/det_10g.onnx",
        ],
    }

    success_count = 0
    failed_files = []

    # Check which files are already present and have a reasonable size
    for file_path in list(model_files.keys()):
        if os.path.exists(file_path) and os.path.getsize(file_path) > 10000:  # >10KB
            print(f"✅ {file_path} already exists, skipping download")
            success_count += 1
            model_files.pop(file_path)  # Remove from files to download

    # Try to download missing files
    for file_path, urls in model_files.items():
        file_downloaded = False
        for url in urls:
            if download_file(url, file_path):
                print(f"✅ Successfully downloaded {file_path}")
                success_count += 1
                file_downloaded = True
                break
            else:
                print(
                    f"Failed to download {file_path} from {url}, trying next source if available"
                )

        if not file_downloaded:
            failed_files.append(file_path)
            print(f"Failed to download {file_path} from any source.")

    # Return the result
    if failed_files:
        print(f"\nThe following model files are still missing or invalid:")
        for file in failed_files:
            print(f"  - {file}")

        print(f"\nYou have the following options:")
        print(f"1. Provide the path to your original LivePortrait repository:")
        print(f"   ./download_models.py --original-repo /path/to/liveportrait")
        print(f"2. Clone the LivePortrait repository and extract the models:")
        print(f"   ./download_models.py --clone-repo")
        print(f"3. Download the models directly from the FaceOne repository:")
        print(f"   git clone https://github.com/lironfarzam/FaceOne.git original_repo")
        print(f"   ./download_models.py --original-repo original_repo")

        return False
    else:
        print("\n✅ All model files have been downloaded successfully!")
        return True


def extract_from_original_repo(original_repo_path, target_path):
    """Extract model files from an original LivePortrait repository."""
    source_path_mappings = {
        # LivePortrait models
        "pretrained_weights/landmark.onnx": "Live_portrait/LivePortrait/pretrained_weights/liveportrait/landmark.onnx",
        "pretrained_weights/base_models/appearance_feature_extractor.pth": "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        "pretrained_weights/base_models/motion_extractor.pth": "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/motion_extractor.pth",
        "pretrained_weights/base_models/spade_generator.pth": "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/spade_generator.pth",
        "pretrained_weights/base_models/warping_module.pth": "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/warping_module.pth",
        "pretrained_weights/retargeting_models/stitching_retargeting_module.pth": "Live_portrait/LivePortrait/pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
        # Insightface models - these paths might need adjustment based on the original repo structure
        "pretrained_weights/insightface/models/buffalo_l/2d106det.onnx": "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx",
        "pretrained_weights/insightface/models/buffalo_l/det_10g.onnx": "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx",
    }

    # Alternative paths that might be found in the repository
    alt_path_mappings = {
        "LivePortrait/pretrained_weights/landmark.onnx": "Live_portrait/LivePortrait/pretrained_weights/liveportrait/landmark.onnx",
        "LivePortrait/pretrained_weights/liveportrait/landmark.onnx": "Live_portrait/LivePortrait/pretrained_weights/liveportrait/landmark.onnx",
        # Add more alternatives as needed
    }

    success_count = 0
    failed_files = []

    # Combine the mappings into one for easier searching
    all_mappings = {**source_path_mappings, **alt_path_mappings}

    for source_rel_path, target_rel_path in all_mappings.items():
        source_abs_path = os.path.join(original_repo_path, source_rel_path)

        # Skip if the target file already exists
        if os.path.exists(target_rel_path) and os.path.getsize(target_rel_path) > 10000:
            print(f"✅ {target_rel_path} already exists, skipping")
            success_count += 1
            continue

        # Try to copy the file
        if os.path.exists(source_abs_path) and os.path.getsize(source_abs_path) > 10000:
            os.makedirs(os.path.dirname(target_rel_path), exist_ok=True)
            try:
                shutil.copy2(source_abs_path, target_rel_path)
                print(f"✅ Copied {source_rel_path} to {target_rel_path}")
                success_count += 1
            except Exception as e:
                print(f"❌ Failed to copy {source_rel_path}: {str(e)}")
                failed_files.append(target_rel_path)
        else:
            # Only add to failed files if it's in the primary mappings
            if source_rel_path in source_path_mappings.keys():
                failed_files.append(target_rel_path)

    # Return the result
    if failed_files:
        print(f"\nThe following model files could not be copied:")
        for file in failed_files:
            print(f"  - {file}")
        return False
    else:
        print("\n✅ All model files have been copied successfully!")
        return True


def clone_liveportrait_repo():
    """Clone the LivePortrait repo and extract model files."""
    temp_dir = "liveportrait_temp"

    # Clean up any existing temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Clone the repo
    try:
        print(f"Cloning LivePortrait repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/TalkUHulk/liveportrait.git", temp_dir],
            check=True,
        )

        # Extract model files
        result = extract_from_original_repo(temp_dir, ".")

        # Clean up
        shutil.rmtree(temp_dir)

        return result
    except Exception as e:
        print(f"❌ Failed to clone repository: {str(e)}")
        # Clean up if directory exists
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False


def check_model_files():
    """Check if all model files exist and have a reasonable size."""
    model_files = [
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/landmark.onnx",
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/motion_extractor.pth",
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/spade_generator.pth",
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/warping_module.pth",
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx",
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx",
    ]

    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 10000:  # <10KB
            missing_files.append(file_path)

    if missing_files:
        print(f"Found {len(missing_files)} missing or invalid model files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("All model files exist and appear to be valid.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Download or copy LivePortrait model files"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check if model files exist"
    )
    parser.add_argument(
        "--original-repo", type=str, help="Path to original LivePortrait repository"
    )
    parser.add_argument(
        "--clone-repo",
        action="store_true",
        help="Clone LivePortrait repository and extract model files",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force download even if files exist"
    )

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs(
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models",
        exist_ok=True,
    )
    os.makedirs(
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/retargeting_models",
        exist_ok=True,
    )
    os.makedirs(
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l",
        exist_ok=True,
    )

    # Just check if model files exist
    if args.check:
        sys.exit(0 if check_model_files() else 1)

    # Extract from original repo if provided
    if args.original_repo:
        if extract_from_original_repo(args.original_repo, "."):
            sys.exit(0)
        else:
            print("Falling back to downloading models...")

    # Clone repo if requested
    if args.clone_repo:
        if clone_liveportrait_repo():
            sys.exit(0)
        else:
            print("Falling back to downloading models...")

    # Otherwise download the models
    if download_liveportrait_models():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
