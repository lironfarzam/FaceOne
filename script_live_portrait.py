import os
import subprocess
import json
import cv2


def load_config(config_path: str = "0-FaceOne/config.json") -> dict:
    """Load configuration from JSON file.

    Args:
        config_path (str, optional): Path to the configuration file. Defaults to "config.json".

    Returns:
        dict: Configuration dictionary.
    """

    with open(config_path, "r") as f:
        return json.load(f)


def generate_video_with_liveportrait(
    image_path: str, video_path: str, output_folder: str
) -> None:
    """
    Generate a face-swapped video using LivePortrait.
    Args:
        image_path (str): Path to the source portrait image.
        video_path (str): Path to the driving video.
        output_folder (str): Directory to save the output.

    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_folder, f"{video_name}_{image_name}_output.mp4")

    command = [
        "python",
        "./Live_portrait/LivePortrait/inference.py",
        "--source",
        image_path,
        "--driving",
        video_path,
        "--output-dir",
        output_folder,
        "--flag-pasteback",
        "--flag-do-crop",
        "--animation-region",
        "all",
        "--audio-priority",
        "driving",
        # "--flag-force-cpu",
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Generated video saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video generation: {e}")


def process_folders(input_folder: str, video_folder: str, output_folder: str) -> None:
    """
    Process all images and videos to create face-swapped animations.
    Args:
        input_folder (str): Folder containing input images.
        video_folder (str): Folder containing driving videos.
        output_folder (str): Folder to save output videos.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ]
    videos = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith(("mp4", "avi", "mov"))
    ]

    for video_path in videos:
        for image_path in images:
            generate_video_with_liveportrait(image_path, video_path, output_folder)


def extract_frames_from_video(
    video_path: str, positives_folder: str, anchors_folder: str
):
    """
    Extract frames from a video and save them to positives and anchors folders.
    Args:
        video_path (str): Path to the video file.
        positives_folder (str): Directory to save positive frames.
        anchors_folder (str): Directory to save anchor frames.
    """
    if not os.path.exists(positives_folder):
        os.makedirs(positives_folder)
    if not os.path.exists(anchors_folder):
        os.makedirs(anchors_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting frames from {video_path}, total frames: {frame_count}")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frames to respective folders
        positive_frame_path = os.path.join(
            positives_folder, f"{os.path.basename(video_path)}_frame_{frame_index}.jpg"
        )
        anchor_frame_path = os.path.join(
            anchors_folder, f"{os.path.basename(video_path)}_frame_{frame_index}.jpg"
        )
        cv2.imwrite(positive_frame_path, frame)
        cv2.imwrite(anchor_frame_path, frame)
        frame_index += 1

    cap.release()
    print(f"Frames saved for video: {video_path}")


def process_output_videos(
    output_folder: str, positives_folder: str, anchors_folder: str
):
    """
    Go through the output folder, process videos that do not contain '_concat',
    extract frames, and save them to positives and anchors folders.
    Args:
        output_folder (str): Folder containing output videos.
        positives_folder (str): Folder to save positive frames.
        anchors_folder (str): Folder to save anchor frames.
    """
    videos = [
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder)
        if f.lower().endswith(("mp4", "avi", "mov")) and "_concat" not in f
    ]

    for video_path in videos:
        extract_frames_from_video(video_path, positives_folder, anchors_folder)


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Setup environment if CPU usage is required
    if config.get("use_cpu", False):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Process folders for video generation
    process_folders(
        config["input_source_folder"],
        config["input_video_folder"],
        config["output_video_folder"],
    )

    # Post-process output videos
    process_output_videos(
        config["output_video_folder"],
        config["positives_folder"],
        config["anchors_folder"],
    )
