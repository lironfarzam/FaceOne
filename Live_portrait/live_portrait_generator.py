import os
import subprocess
import cv2
import json
import sys

# Add parent directory to path to import cool_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cool_utils import load_config, print_green, print_red, print_blue


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
) -> None:
    """
    Generate a face-swapped video using LivePortrait.

    Args:
        image_path (str): Path to the source portrait image.
        video_path (str): Path to the driving video.
        output_folder (str): Directory to save the output.
        config (dict): Configuration dictionary with LivePortrait settings.
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
    if config.get("use_cpu", False):
        command.append("--flag-force-cpu")

    try:
        subprocess.run(command, check=True)
        print_green(f"Generated video saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print_red(f"Error during video generation: {e}")


def process_folders(
    input_folder: str, video_folder: str, output_folder: str, config: dict
) -> None:
    """
    Process all images and videos to create face-swapped animations.

    Args:
        input_folder (str): Folder containing input images.
        video_folder (str): Folder containing driving videos.
        output_folder (str): Folder to save output videos.
        config (dict): Configuration dictionary with LivePortrait settings.
    """
    # Ensure directories exist
    if not os.path.exists(input_folder):
        print_red(f"Input folder not found: {input_folder}")
        return

    if not os.path.exists(video_folder):
        print_red(f"Video folder not found: {video_folder}")
        return

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
        return

    videos = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith(("mp4", "avi", "mov"))
    ]

    if not videos:
        print_red(f"No videos found in {video_folder}")
        return

    print_blue(f"Found {len(images)} images and {len(videos)} videos")

    for video_path in videos:
        for image_path in images:
            generate_video_with_liveportrait(
                image_path, video_path, output_folder, config
            )


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
    print_green(f"Extracted {saved_count} frames from video: {video_path}")


def process_output_videos(
    output_folder: str,
    positives_folder: str,
    anchors_folder: str,
    frame_interval: int = 5,
):
    """
    Go through the output folder, process videos that do not contain '_concat',
    extract frames, and save them to positives and anchors folders.

    Args:
        output_folder (str): Folder containing output videos.
        positives_folder (str): Folder to save positive frames.
        anchors_folder (str): Folder to save anchor frames.
        frame_interval (int): Interval between frames to extract (default: 5).
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

    print_blue(f"Processing {len(videos)} output videos for frame extraction")

    for video_path in videos:
        extract_frames_from_video(
            video_path, positives_folder, anchors_folder, frame_interval
        )


def main(config_path: str = None):
    """
    Main function to run the LivePortrait generator.

    Args:
        config_path (str): Path to the configuration file.
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

    # Setup environment if CPU usage is required
    if config.get("use_cpu", False):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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
    print("use_cpu: ", config["use_cpu"])

    # Process folders for video generation
    process_folders(
        input_source_folder, input_video_folder, output_video_folder, config
    )

    # Post-process output videos
    process_output_videos(
        output_video_folder,
        positives_folder,
        anchors_folder,
        config.get("frame_interval", 5),
    )


if __name__ == "__main__":
    # Allow specifying a different config file as a command-line argument
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    main(config_path)
