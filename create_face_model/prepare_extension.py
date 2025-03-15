#!/usr/bin/env python3
"""
FaceOne: Chrome Extension Preparation
=====================================

This script prepares the trained model and embeddings for use in the Chrome extension.
It performs the following tasks:
1. Copies the trained model to the Chrome extension directory
2. Copies the face embeddings to the Chrome extension directory
3. Ensures all files are in the correct format for browser use

Author: Liron Farzam
"""

import os
import sys
import json
import shutil
import time
from datetime import datetime

# Add parent directory to path to import cool_utils
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


def ensure_directory(directory):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
    return os.path.exists(directory)


def copy_model_files(source_dir, dest_dir):
    """Copy model files to the Chrome extension directory."""
    print_blue(f"Copying model files from {source_dir} to {dest_dir}")

    if not os.path.exists(source_dir):
        print_red(f"Source directory {source_dir} does not exist")
        return False

    # Ensure destination directory exists
    ensure_directory(dest_dir)

    # Check if source is a directory containing model files
    if os.path.isdir(source_dir):
        tfjs_dir = os.path.join(source_dir, "tfjs_graph_model")
        if os.path.exists(tfjs_dir):
            source_dir = tfjs_dir
        else:
            tfjs_dir = os.path.join(source_dir, "tfjs_layers_model")
            if os.path.exists(tfjs_dir):
                source_dir = tfjs_dir
            else:
                # Try all possible subdirectories
                for subdir in os.listdir(source_dir):
                    subdir_path = os.path.join(source_dir, subdir)
                    if os.path.isdir(subdir_path) and "model.json" in os.listdir(
                        subdir_path
                    ):
                        source_dir = subdir_path
                        break

    # Check if we found a valid model directory
    if not os.path.exists(os.path.join(source_dir, "model.json")):
        print_red(f"Could not find model.json in {source_dir} or its subdirectories")

        # Create minimal model files if not found
        print_blue("Creating minimal model files as placeholder")
        model_json = {
            "format": "graph-model",
            "generatedBy": "FaceOne",
            "convertedBy": "FaceOne",
            "modelTopology": {},
            "weightsManifest": [{"paths": ["group1-shard1of1.bin"], "weights": []}],
        }

        # Write model.json
        with open(os.path.join(dest_dir, "model.json"), "w") as f:
            json.dump(model_json, f, indent=2)

        # Create empty weights file
        with open(os.path.join(dest_dir, "group1-shard1of1.bin"), "wb") as f:
            f.write(b"")

        print_green("Created minimal model files")
        return False

    # Copy model files
    try:
        # Clear destination directory
        if os.path.exists(dest_dir):
            for item in os.listdir(dest_dir):
                item_path = os.path.join(dest_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        # Copy all files from source to destination
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            dest_item = os.path.join(dest_dir, item)

            if os.path.isfile(source_item):
                shutil.copy2(source_item, dest_item)
            elif os.path.isdir(source_item):
                shutil.copytree(source_item, dest_item)

        print_green(f"Successfully copied model files to {dest_dir}")
        return True

    except Exception as e:
        print_red(f"Error copying model files: {e}")
        return False


def copy_embeddings(source_file, dest_file):
    """Copy embeddings to the Chrome extension directory."""
    print_blue(f"Copying embeddings from {source_file} to {dest_file}")

    if not os.path.exists(source_file):
        print_red(f"Source file {source_file} does not exist")

        # Create minimal embeddings file if not found
        print_blue("Creating minimal embeddings file as placeholder")

        # Create placeholder embeddings (512-dimensional zero vectors)
        placeholder_embeddings = []
        for _ in range(5):  # 5 placeholder embeddings
            embedding = [0.0] * 512
            placeholder_embeddings.append(embedding)

        # Ensure destination directory exists
        ensure_directory(os.path.dirname(dest_file))

        # Write to destination
        with open(dest_file, "w") as f:
            json.dump(placeholder_embeddings, f)

        print_green("Created minimal embeddings file")
        return False

    try:
        # Ensure destination directory exists
        ensure_directory(os.path.dirname(dest_file))

        # Copy embeddings file
        shutil.copy2(source_file, dest_file)

        print_green(f"Successfully copied embeddings to {dest_file}")
        return True

    except Exception as e:
        print_red(f"Error copying embeddings: {e}")
        return False


def update_extension_version():
    """Update the extension version in manifest.json."""
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "chrome_extensions",
        "manifest.json",
    )

    if not os.path.exists(manifest_path):
        print_red(f"Manifest file not found: {manifest_path}")
        return False

    # Skip version updating as requested
    print_blue("Skipping version update as requested")
    return True

    # # Previous implementation commented out
    # """
    # try:
    #     with open(manifest_path, "r") as f:
    #         manifest = json.load(f)

    #     # Update version with current date
    #     current_date = datetime.now().strftime("%y.%m.%d")
    #     if "version" in manifest:
    #         version_parts = manifest["version"].split(".")
    #         if len(version_parts) >= 3:
    #             # Keep major.minor but update patch with date
    #             manifest["version"] = (
    #                 f"{version_parts[0]}.{version_parts[1]}.{current_date}"
    #             )
    #         else:
    #             manifest["version"] = f"1.0.{current_date}"
    #     else:
    #         manifest["version"] = f"1.0.{current_date}"

    #     # Write updated manifest
    #     with open(manifest_path, "w") as f:
    #         json.dump(manifest, f, indent=2)

    #     print_green(f"Updated extension version to {manifest['version']}")
    #     return True

    # except Exception as e:
    #     print_red(f"Error updating extension version: {e}")
    #     return False
    # """


def main():
    """Main function to prepare the Chrome extension."""
    print_blue("Starting Chrome extension preparation")

    # Load configuration
    config = load_config()

    if not config:
        print_red("Failed to load configuration")
        return 1

    # Check required configuration parameters
    required_params = [
        "path_for_model",
        "chrome_extension_model_path",
        "path_for_positives_embeddings",
        "chrome_extension_embeddings_path",
    ]

    missing_params = [param for param in required_params if param not in config]

    if missing_params:
        print_red(
            f"Missing required configuration parameters: {', '.join(missing_params)}"
        )
        return 1

    # Prepare model and embeddings
    success = True

    # Copy model files
    model_success = copy_model_files(
        config["path_for_model"], config["chrome_extension_model_path"]
    )
    if not model_success:
        success = False

    # Copy embeddings
    embeddings_success = copy_embeddings(
        config["path_for_positives_embeddings"],
        config["chrome_extension_embeddings_path"],
    )
    if not embeddings_success:
        success = False

    # Update extension version
    version_success = update_extension_version()
    if not version_success:
        success = False

    if success:
        print_green("Chrome extension preparation completed successfully")
        return 0
    else:
        print_red("Chrome extension preparation completed with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
