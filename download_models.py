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

    # Get the file size
    response = requests.head(url)
    file_size = int(response.headers.get("content-length", 0))

    # Download the file with a progress bar
    with requests.get(url, stream=True) as r:
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

    print(f"Downloaded {local_path}")


def download_liveportrait_models():
    """Download the LivePortrait model files"""
    # Try multiple potential sources

    models = {
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/landmark.onnx": [
            "https://huggingface.co/TalkUHulk/liveportrait/resolve/main/landmark.onnx",
            "https://github.com/TalkUHulk/liveportrait/raw/main/pretrained_weights/liveportrait/landmark.onnx",
        ],
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx": [
            "https://github.com/deepinsight/insightface/raw/master/detection/scrfd/onnx/scrfd_2.5g_kps.onnx"
        ],
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx": [
            "https://github.com/deepinsight/insightface/raw/master/detection/scrfd/onnx/scrfd_10g_kps.onnx"
        ],
    }

    for local_path, urls in models.items():
        if (
            not os.path.exists(local_path) or os.path.getsize(local_path) < 1000
        ):  # If file doesn't exist or is very small
            success = False
            for url in urls:
                try:
                    download_file(url, local_path)
                    success = True
                    break
                except Exception as e:
                    print(f"Error downloading {local_path} from {url}: {e}")

            if not success:
                print(f"Failed to download {local_path} from any source.")


def extract_from_original_repo(original_repo_path, target_path):
    """
    Extract model files from the original repository.

    Args:
        original_repo_path (str): Path to the original repository
        target_path (str): Path to copy the model file to
    """
    source_path = os.path.join(original_repo_path, target_path)
    if os.path.exists(source_path):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Copy the file
        print(f"Copying {source_path} to {target_path}")
        shutil.copy2(source_path, target_path)
        print(f"Copied {target_path}")
        return True
    else:
        print(f"File not found in original repository: {source_path}")
        return False


def clone_liveportrait_repo():
    """Clone the LivePortrait repository"""
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/TalkUHulk/liveportrait.git",
                "liveportrait_original",
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning LivePortrait repository: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download LivePortrait model files")
    parser.add_argument(
        "--check", action="store_true", help="Check if model files exist and are valid"
    )
    parser.add_argument(
        "--original-repo", type=str, help="Path to the original LivePortrait repository"
    )
    parser.add_argument(
        "--clone-repo",
        action="store_true",
        help="Clone the original LivePortrait repository",
    )
    args = parser.parse_args()

    missing_files = []
    model_paths = [
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/landmark.onnx",
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx",
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx",
    ]

    # Check for missing files
    for model_path in model_paths:
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
            missing_files.append(model_path)

    if args.check:
        if missing_files:
            print("The following model files are missing or invalid:")
            for f in missing_files:
                print(f"  - {f}")
            print("Run this script without --check to download them.")
            return False
        else:
            print("All model files exist and appear to be valid.")
            return True
    elif args.clone_repo:
        # Clone the LivePortrait repository
        if clone_liveportrait_repo():
            # Extract model files from the cloned repository
            for model_path in missing_files:
                extract_from_original_repo("liveportrait_original", model_path)
        else:
            print("Failed to clone the LivePortrait repository.")
            return False
    elif args.original_repo:
        # Extract model files from the original repository
        for model_path in missing_files:
            extract_from_original_repo(args.original_repo, model_path)
    else:
        # Download the model files
        download_liveportrait_models()

    # Check if we still have missing files
    still_missing = []
    for model_path in model_paths:
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
            still_missing.append(model_path)

    if still_missing:
        print("\nThe following model files are still missing or invalid:")
        for f in still_missing:
            print(f"  - {f}")
        print("\nYou have the following options:")
        print("1. Provide the path to your original LivePortrait repository:")
        print("   ./download_models.py --original-repo /path/to/liveportrait")
        print("2. Clone the LivePortrait repository and extract the models:")
        print("   ./download_models.py --clone-repo")
        print("3. Download the models directly from the FaceOne repository:")
        print("   git clone https://github.com/lironfarzam/FaceOne.git original_repo")
        print("   ./download_models.py --original-repo original_repo")
        return False
    else:
        print("\nAll model files have been successfully downloaded or extracted.")
        return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
