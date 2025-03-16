#!/usr/bin/env python3
"""
FaceOne: Main Pipeline Execution
================================

This script orchestrates the complete FaceOne pipeline:
1. Download face images from Facebook profiles
2. Process and extract faces from the images
3. Generate live portraits using the processed faces
4. Download LFW dataset for negative examples
5. Create a face recognition model
6. Prepare files for the Chrome extension

Run this script to execute the entire workflow in sequence.

Author: Liron Farzam
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(message):
    """Print a formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD} {message} {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_step(step_num, total_steps, message):
    """Print a formatted step message."""
    print(
        f"\n{Colors.BLUE}{Colors.BOLD}[Step {step_num}/{total_steps}] {message}{Colors.ENDC}\n"
    )


def print_success(message):
    """Print a success message."""
    print(f"{Colors.GREEN}{message}{Colors.ENDC}")


def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.WARNING}{message}{Colors.ENDC}")


def print_error(message):
    """Print an error message."""
    print(f"{Colors.RED}{Colors.BOLD}ERROR: {message}{Colors.ENDC}")


def run_script(script_path, description, step_num, total_steps):
    """Run a Python script and handle any errors."""
    print_step(step_num, total_steps, description)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True,
            # Uncomment to capture output instead of showing it live
            # capture_output=True
        )

        elapsed_time = time.time() - start_time
        print_success(
            f"✓ {description} completed successfully in {elapsed_time:.2f} seconds."
        )
        return True

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print_error(
            f"Script failed after {elapsed_time:.2f} seconds with return code {e.returncode}"
        )
        if hasattr(e, "output") and e.output:
            print_error(f"Output: {e.output}")
        return False

    except Exception as e:
        elapsed_time = time.time() - start_time
        print_error(
            f"An unexpected error occurred after {elapsed_time:.2f} seconds: {str(e)}"
        )
        return False


def check_prerequisites():
    """Check if all necessary prerequisites are met."""
    print_header("Checking Prerequisites")

    # Check if config.json exists
    if not os.path.exists("config.json"):
        print_error(
            "config.json not found. Please create this file with your configuration settings."
        )
        print_warning("See the example config in the documentation.")
        return False

    # Load and validate config
    try:
        with open("config.json", "r") as f:
            config = json.load(f)

        # Check for required configuration parameters
        required_params = [
            "profile_url",
            "download_photo_folder",
            "path_for_model",
            "chrome_extension_model_path",
            "path_for_positives_embeddings",
            "chrome_extension_embeddings_path",
        ]

        missing_params = [param for param in required_params if param not in config]

        if missing_params:
            print_error(
                f"Missing required configuration parameters: {', '.join(missing_params)}"
            )
            return False

        print_success("✓ Configuration file validated.")

    except json.JSONDecodeError:
        print_error("config.json is not valid JSON. Please check its format.")
        return False
    except Exception as e:
        print_error(f"Error checking configuration: {str(e)}")
        return False

    # Check for required directories, create if missing
    directories = [
        "Facebook_profile_handling",
        "create_face_model",
        "chrome_extensions",
        "Live_portrait",
    ]

    for directory in directories:
        if not os.path.exists(directory):
            print_error(f"Required directory {directory} not found.")
            return False

    print_success("✓ All required directories exist.")

    # Check for required script files
    required_scripts = [
        "Facebook_profile_handling/download_images.py",
        "Facebook_profile_handling/face_processing.py",
        "Live_portrait/live_portrait_generator.py",
        "create_face_model/download_lfw.py",
        "create_face_model/create_model.py",
        "create_face_model/prepare_extension.py",
    ]

    for script in required_scripts:
        if not os.path.exists(script):
            print_error(f"Required script {script} not found.")
            return False

    print_success("✓ All required scripts exist.")

    # Check if necessary Python packages are installed by reading requirements.txt
    try:
        import pkg_resources
        import re

        # Read requirements.txt
        if not os.path.exists("requirements.txt"):
            print_error("requirements.txt not found.")
            return False

        with open("requirements.txt", "r") as f:
            requirements = f.readlines()

        # Package name mapping for special cases
        package_mapping = {
            "opencv-python": "cv2",
            "pillow": "PIL",
            "scikit-learn": "sklearn",
            "scikit-image": "skimage",
            "pyyaml": "yaml",
            "ffmpeg-python": "ffmpeg",
        }

        # Parse requirements, skipping comments and empty lines
        packages_to_check = []
        for line in requirements:
            line = line.strip()
            # Skip comments, empty lines, and special requirements (like --extra-index-url)
            if (
                not line
                or line.startswith("#")
                or line.startswith("-r")
                or line.startswith("--")
            ):
                continue

            # Extract package name (remove version specifiers)
            package_name = re.split(r"[<>=~]", line)[0].strip()
            if package_name:
                packages_to_check.append(package_name)

        # Check each package
        missing_packages = []
        for package in packages_to_check:
            try:
                # Handle special cases with different import names
                if package.lower() in package_mapping:
                    # For packages with different import names, try to import the module
                    module_name = package_mapping[package.lower()]
                    try:
                        __import__(module_name)
                    except ImportError:
                        missing_packages.append(package)
                else:
                    # Use pkg_resources for standard packages
                    pkg_resources.get_distribution(package)
            except (pkg_resources.DistributionNotFound, ImportError):
                missing_packages.append(package)
            except Exception as e:
                print_warning(f"Warning checking {package}: {str(e)}")

        if missing_packages:
            print_error(
                f"Missing required Python packages: {', '.join(missing_packages)}"
            )
            print_warning(
                "Please install all dependencies: pip install -r requirements.txt"
            )
            return False

        print_success(
            f"✓ All {len(packages_to_check)} dependencies from requirements.txt are installed."
        )

    except Exception as e:
        print_error(f"Error checking dependencies: {str(e)}")
        print_warning(
            "Please install all dependencies: pip install -r requirements.txt"
        )
        return False

    return True


def setup_directories(config):
    """Set up necessary directories based on config."""
    os.makedirs(config["download_photo_folder"], exist_ok=True)

    # Ensure model directories exist
    model_path = config["path_for_model"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    chrome_ext_path = config["chrome_extension_model_path"]
    os.makedirs(os.path.dirname(chrome_ext_path), exist_ok=True)

    # Set up live portrait directories if configured
    live_portrait_dirs = [
        "input_source_folder",
        "input_video_folder",
        "output_video_folder",
        "positives_folder",
        "anchors_folder",
    ]

    for dir_key in live_portrait_dirs:
        if dir_key in config:
            os.makedirs(config[dir_key], exist_ok=True)
            print_success(f"✓ Created live portrait directory: {dir_key}")

    print_success("✓ Directories set up successfully.")


def main():
    """Main function to run the entire FaceOne pipeline."""
    parser = argparse.ArgumentParser(description="Run the FaceOne pipeline")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip the image download step"
    )
    parser.add_argument(
        "--skip-processing", action="store_true", help="Skip the face processing step"
    )
    parser.add_argument(
        "--skip-portraits",
        action="store_true",
        help="Skip the live portrait generation step",
    )
    parser.add_argument(
        "--skip-lfw", action="store_true", help="Skip the LFW dataset download step"
    )
    parser.add_argument(
        "--skip-model", action="store_true", help="Skip the model creation step"
    )
    parser.add_argument(
        "--chrome-only",
        action="store_true",
        help="Only prepare the Chrome extension files",
    )
    args = parser.parse_args()

    print_header("FaceOne Pipeline Execution")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not check_prerequisites():
        return

    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    # Setup directories
    setup_directories(config)

    total_steps = 6 - sum(
        [
            args.skip_download,
            args.skip_processing,
            args.skip_portraits,
            args.skip_lfw,
            args.skip_model,
            args.chrome_only,
        ]
    )
    current_step = 1
    pipeline_success = True

    start_time = time.time()

    # 1. Download images
    if not args.skip_download and not args.chrome_only:
        if not run_script(
            "Facebook_profile_handling/download_images.py",
            "Downloading images from Facebook profiles",
            current_step,
            total_steps,
        ):
            print_error("Image download failed. Pipeline aborted.")
            return
        current_step += 1

    # 2. Process images
    if not args.skip_processing and not args.chrome_only:
        if not run_script(
            "Facebook_profile_handling/face_processing.py",
            "Processing and extracting faces from images",
            current_step,
            total_steps,
        ):
            print_error("Face processing failed. Pipeline aborted.")
            return
        current_step += 1

    # 3. Generate Live Portraits
    if not args.skip_portraits and not args.chrome_only:
        if not run_script(
            "Live_portrait/live_portrait_generator.py",
            "Generating live portraits from processed faces",
            current_step,
            total_steps,
        ):
            print_warning(
                "Live portrait generation had issues. Pipeline will continue but portraits may not be available."
            )
        current_step += 1

    # 4. Download LFW dataset for negative examples
    if not args.skip_lfw and not args.chrome_only:
        lfw_result = run_script(
            "create_face_model/download_lfw.py",
            "Downloading LFW dataset for negative examples",
            current_step,
            total_steps,
        )
        if not lfw_result:
            print_warning(
                "LFW dataset download had issues, but the pipeline will continue with available negative examples."
            )
            print_warning(
                "The model will use synthetic or existing negative examples, which may impact quality."
            )
        current_step += 1

    # 5. Create model
    if not args.skip_model and not args.chrome_only:
        if not run_script(
            "create_face_model/create_model.py",
            "Creating and training face recognition model",
            current_step,
            total_steps,
        ):
            print_error("Model creation failed. Pipeline aborted.")
            return
        current_step += 1

    # 6. Prepare Chrome extension files
    if not run_script(
        "create_face_model/prepare_extension.py",
        "Preparing files for Chrome extension",
        current_step,
        total_steps,
    ):
        print_warning(
            "Chrome extension preparation encountered issues, but may still be usable."
        )
        pipeline_success = False

    # Calculate total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print_header("Pipeline Summary")

    if pipeline_success:
        print_success(
            f"✓ FaceOne pipeline completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )
        print_success(f"✓ Model saved to: {config['path_for_model']}")
        print_success(
            f"✓ Chrome extension files ready at: {config['chrome_extension_model_path']}"
        )

        print("\nNext steps:")
        print("1. Open Chrome and go to chrome://extensions")
        print("2. Enable Developer Mode (toggle in top-right)")
        print("3. Click 'Load unpacked' and select the chrome_extensions folder")
        print("4. The FaceOne icon should appear in your toolbar")
    else:
        print_warning("Pipeline completed with warnings or errors.")
        print("Please check the logs above for details and try to resolve the issues.")


if __name__ == "__main__":
    main()
