import os
import subprocess
import cv2
import json
import sys
import time
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint

# Add parent directory to path to import cool_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cool_utils import load_config, print_green, print_red, print_blue


def detect_gpu():
    """
    Detect available GPU hardware and return information about it.

    Returns:
        dict: Information about available GPU hardware
    """
    gpu_info = {
        "has_gpu": False,
        "type": None,
        "name": None,
        "platform": platform.system(),
    }

    # Check for CUDA (NVIDIA) GPU
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["type"] = "cuda"
            gpu_info["count"] = torch.cuda.device_count()
            gpu_info["name"] = torch.cuda.get_device_name(0)
            return gpu_info
    except (ImportError, Exception):
        pass

    # Check for MPS (Apple Silicon) GPU
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["type"] = "mps"
            gpu_info["name"] = "Apple Silicon GPU"
            return gpu_info
    except (ImportError, Exception):
        pass

    # Check for ROCm (AMD) GPU
    try:
        import torch

        if hasattr(torch, "hip") and torch.hip.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["type"] = "rocm"
            gpu_info["count"] = torch.hip.device_count()
            gpu_info["name"] = "AMD GPU"
            return gpu_info
    except (ImportError, Exception):
        pass

    return gpu_info


def resolve_path(path, base_dir=None):
    """
    Resolve a path that might be relative to the base directory.

    Args:
        path (str): The path to resolve.
        base_dir (str): The base directory to resolve relative paths from.

    Returns:
        str: The resolved absolute path.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if path.startswith("./"):
        # Remove the leading './' and join with base_dir
        return os.path.join(base_dir, path[2:])
    elif os.path.isabs(path):
        return path
    else:
        return os.path.join(base_dir, path)


def generate_video_with_liveportrait(
    image_path: str, video_path: str, output_folder: str, config: dict
) -> tuple:
    """
    Generate a face-swapped video using LivePortrait.

    Args:
        image_path (str): Path to the source portrait image.
        video_path (str): Path to the driving video.
        output_folder (str): Directory to save the output.
        config (dict): Configuration dictionary with LivePortrait settings.

    Returns:
        tuple: (success, image_name, video_name, output_file, error_message)
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_folder, f"{video_name}_{image_name}_output.mp4")

    # Build command with parameters from config
    command = [
        "python",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "LivePortrait/inference.py"
        ),
        "--source",
        image_path,
        "--driving",
        video_path,
        "--output-dir",
        output_folder,
        "--animation-region",
        config.get("animation_region", "all"),
        "--audio-priority",
        config.get("audio_priority", "driving"),
    ]

    # Add optional flags based on config
    if config.get("do_pasteback", True):
        command.append("--flag-pasteback")
    if config.get("do_crop", True):
        command.append("--flag-do-crop")
    if config.get("force_cpu", False):
        command.append("--flag-force-cpu")

    # Set environment variables for GPU support
    env = os.environ.copy()

    # Apple Silicon MPS support
    if (
        config.get("enable_mps", True)
        and platform.system() == "Darwin"
        and platform.processor() == "arm"
    ):
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # CUDA memory settings
    if config.get("cuda_memory_fraction") and not config.get("force_cpu", False):
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow memory growth

    # Half precision for faster GPU processing
    if config.get("use_half_precision", True) and not config.get("force_cpu", False):
        command.append("--flag-use-half-precision")

    print_blue(f"Processing: {video_name} with {image_name}")

    try:
        # Run the command and show output directly (like the original script)
        subprocess.run(command, check=True, env=env)
        print_green(f"Generated video saved to {output_file}")
        return (True, image_name, video_name, output_file, None)
    except subprocess.CalledProcessError as e:
        error_message = str(e)
        print_red(f"Error during video generation: {error_message}")
        return (False, image_name, video_name, output_file, error_message)


def process_video_image_pair(args):
    """
    Process a single video-image pair for parallel execution.

    Args:
        args: Tuple containing (image_path, video_path, output_folder, config)

    Returns:
        Result of generate_video_with_liveportrait
    """
    image_path, video_path, output_folder, config = args
    return generate_video_with_liveportrait(
        image_path, video_path, output_folder, config
    )


def process_folders_parallel(
    input_folder: str,
    video_folder: str,
    output_folder: str,
    config: dict,
    max_workers: int = None,
) -> list:
    """
    Process all images and videos to create face-swapped animations in parallel.

    Args:
        input_folder (str): Folder containing input images.
        video_folder (str): Folder containing driving videos.
        output_folder (str): Folder to save output videos.
        config (dict): Configuration dictionary with LivePortrait settings.
        max_workers (int, optional): Maximum number of parallel workers. Defaults to CPU count.

    Returns:
        list: List of generated video paths
    """
    # Ensure directories exist
    if not os.path.exists(input_folder):
        print_red(f"Input folder not found: {input_folder}")
        return []

    if not os.path.exists(video_folder):
        print_red(f"Video folder not found: {video_folder}")
        return []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print_green(f"Created output folder: {output_folder}")

    images = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ]

    if not images:
        print_red(f"No images found in {input_folder}")
        return []

    videos = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith(("mp4", "avi", "mov"))
    ]

    if not videos:
        print_red(f"No videos found in {video_folder}")
        return []

    # Calculate total combinations
    total_combinations = len(videos) * len(images)
    print_blue(
        f"Found {len(images)} images and {len(videos)} videos - {total_combinations} combinations to process"
    )

    # Prepare all combinations for parallel processing
    combinations = []
    for video_path in videos:
        for image_path in images:
            combinations.append((image_path, video_path, output_folder, config))

    # Determine number of workers
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), total_combinations)

    print_blue(f"Using {max_workers} parallel workers")
    print_blue("Starting video generation process...")

    # Setup progress tracking
    successful_videos = []
    failed_videos = []

    # Process combinations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_video_image_pair, combo) for combo in combinations
        ]

        # Process results as they complete
        for future in as_completed(futures):
            try:
                success, image_name, video_name, output_file, error = future.result()

                if success:
                    successful_videos.append(output_file)
                else:
                    failed_videos.append((image_name, video_name, error))
            except Exception as e:
                print_red(f"Unexpected error: {str(e)}")

    # Report final results
    print_blue("Video generation process completed")
    print_green(f"Successfully generated {len(successful_videos)} videos")
    if failed_videos:
        print_red(f"Failed to generate {len(failed_videos)} videos")

    return successful_videos


def extract_frames_from_video(
    video_path: str, positives_folder: str, anchors_folder: str, frame_interval: int = 5
):
    """
    Extract frames from a video and save them to positives and anchors folders.

    Args:
        video_path (str): Path to the video file.
        positives_folder (str): Directory to save positive frames.
        anchors_folder (str): Directory to save anchor frames.
        frame_interval (int): Interval between frames to extract (default: 5).
    """
    if not os.path.exists(positives_folder):
        os.makedirs(positives_folder)
    if not os.path.exists(anchors_folder):
        os.makedirs(anchors_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting frames from {video_path}, total frames: {frame_count}")

    frame_index = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frames at specified intervals
        if frame_index % frame_interval == 0:
            # Save frames to respective folders
            positive_frame_path = os.path.join(
                positives_folder,
                f"{os.path.basename(video_path)}_frame_{frame_index}.jpg",
            )
            anchor_frame_path = os.path.join(
                anchors_folder,
                f"{os.path.basename(video_path)}_frame_{frame_index}.jpg",
            )
            cv2.imwrite(positive_frame_path, frame)
            cv2.imwrite(anchor_frame_path, frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    print_green(f"Frames saved for video: {video_path} (Total: {saved_count} frames)")
    return (video_path, saved_count)


def extract_frames_worker(args):
    """
    Worker function for parallel frame extraction.

    Args:
        args: Tuple containing (video_path, positives_folder, anchors_folder, frame_interval)

    Returns:
        Tuple of (video_path, saved_count)
    """
    video_path, positives_folder, anchors_folder, frame_interval = args

    # Use the extract_frames_from_video function directly
    return extract_frames_from_video(
        video_path, positives_folder, anchors_folder, frame_interval
    )


def process_output_videos_parallel(
    output_folder: str,
    positives_folder: str,
    anchors_folder: str,
    frame_interval: int = 5,
    max_workers: int = None,
):
    """
    Process output videos in parallel to extract frames.

    Args:
        output_folder (str): Folder containing output videos.
        positives_folder (str): Folder to save positive frames.
        anchors_folder (str): Folder to save anchor frames.
        frame_interval (int): Interval between frames to extract (default: 5).
        max_workers (int, optional): Maximum number of parallel workers. Defaults to CPU count.
    """
    if not os.path.exists(output_folder):
        print_red(f"Output folder not found: {output_folder}")
        return

    videos = [
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder)
        if f.lower().endswith(("mp4", "avi", "mov")) and "_concat" not in f
    ]

    if not videos:
        print_red(f"No videos found in {output_folder}")
        return

    # Determine number of workers
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(videos))

    print_blue(f"Processing {len(videos)} output videos for frame extraction")
    print_blue(f"Using {max_workers} parallel workers")

    # Prepare arguments for parallel processing
    args_list = [
        (video_path, positives_folder, anchors_folder, frame_interval)
        for video_path in videos
    ]

    # Process videos in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(extract_frames_worker, args) for args in args_list]

        # Process results as they complete
        for future in as_completed(futures):
            # Results are handled in the extract_frames_worker function
            pass

    print_green(f"Completed frame extraction from {len(videos)} videos")


def main(config_path: str = None, max_workers: int = None, force_cpu: bool = None):
    """
    Main function to run the LivePortrait generator.

    Args:
        config_path (str): Path to the configuration file.
        max_workers (int, optional): Maximum number of parallel workers. Defaults to CPU count.
        force_cpu (bool, optional): Whether to force CPU usage. Overrides config setting.
    """
    # Find the config file
    if config_path is None:
        # Try different possible locations for the config file
        possible_paths = [
            "config.json",  # Current directory
            "../config.json",  # Parent directory
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config.json",
            ),  # Project root
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

        if config_path is None:
            print_red("Could not find config.json in any of the expected locations.")
            return

    print_blue(f"Using config file: {config_path}")

    # Load configuration
    config = load_config(config_path)

    # Override force_cpu if specified
    if force_cpu is not None:
        config["force_cpu"] = force_cpu

    # Get number of workers from config or use default
    if max_workers is None:
        max_workers = config.get("num_of_workers", multiprocessing.cpu_count())

    # Detect GPU
    gpu_info = detect_gpu()

    # Setup environment for GPU/CPU
    if config.get("force_cpu", False):
        print_blue("Forcing CPU usage as requested")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
    elif gpu_info["has_gpu"]:
        print_green(f"GPU detected: {gpu_info['name']} ({gpu_info['type']})")

        # Apple Silicon specific settings
        if gpu_info["type"] == "mps" and config.get("enable_mps", True):
            print_blue("Setting up Apple Silicon GPU (MPS) support")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # CUDA specific settings
        elif gpu_info["type"] == "cuda":
            print_blue("Setting up CUDA GPU support")
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

            # Set memory fraction if specified
            if config.get("cuda_memory_fraction"):
                fraction = float(config.get("cuda_memory_fraction"))
                print_blue(f"Limiting GPU memory usage to {fraction*100}%")
    else:
        print_blue("No GPU detected, using CPU")

    # Resolve paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_source_folder = resolve_path(config["input_source_folder"], base_dir)
    input_video_folder = resolve_path(config["input_video_folder"], base_dir)
    output_video_folder = resolve_path(config["output_video_folder"], base_dir)
    positives_folder = resolve_path(config["positives_folder"], base_dir)
    anchors_folder = resolve_path(config["anchors_folder"], base_dir)

    print_blue("Starting Live Portrait Generator")
    print("input_source_folder: ", input_source_folder)
    print("input_video_folder: ", input_video_folder)
    print("output_video_folder: ", output_video_folder)
    print("positives_folder: ", positives_folder)
    print("anchors_folder: ", anchors_folder)
    print("force_cpu: ", config.get("force_cpu", False))
    print("parallel workers: ", max_workers)

    if (
        config.get("use_half_precision", True)
        and not config.get("force_cpu", False)
        and gpu_info["has_gpu"]
    ):
        print("using half precision: True (faster GPU processing)")

    # Record start time
    start_time = time.time()

    # Process folders for video generation in parallel
    generated_videos = process_folders_parallel(
        input_source_folder,
        input_video_folder,
        output_video_folder,
        config,
        max_workers,
    )

    # Post-process output videos in parallel
    if generated_videos:
        process_output_videos_parallel(
            output_video_folder,
            positives_folder,
            anchors_folder,
            config.get("frame_interval", 5),
            max_workers,
        )

    # Calculate and display total execution time
    execution_time = time.time() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print_green(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Live Portrait Generator")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")

    args = parser.parse_args()

    # Determine force_cpu value
    force_cpu = None
    if args.cpu:
        force_cpu = True
    elif args.gpu:
        force_cpu = False

    # For backward compatibility
    config_path = args.config
    if config_path is None and len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        config_path = sys.argv[1]

    max_workers = args.workers
    if max_workers is None and len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        try:
            max_workers = int(sys.argv[2])
        except ValueError:
            print_red(f"Invalid number of workers: {sys.argv[2]}. Using default.")

    main(config_path, max_workers, force_cpu)
