import os
import json
import shutil
import numpy as np
from deepface import DeepFace


def get_embedding(image_path, model_name="Facenet512"):
    """Get the embedding vector for the given image."""
    embedding = DeepFace.represent(
        img_path=image_path, model_name=model_name, enforce_detection=False
    )
    return embedding[0]["embedding"]


def save_embedding(image_path, output_path):
    """Save the embedding vector to a JSON file."""
    embedding = get_embedding(image_path)
    with open(output_path, "w") as f:
        json.dump(embedding, f)
    print(f"Embedding saved to {output_path}")


def main():
    image_dir = "../imgs/test_imgs"
    output_dir = "./embeddings_py"
    temp_dir = "./temp_images"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    for image_name in os.listdir(image_dir):
        if image_name.endswith(".jpeg"):
            image_path = os.path.join(image_dir, image_name)
            temp_image_path = os.path.join(temp_dir, f"temp_{image_name}")
            shutil.copy(image_path, temp_image_path)
            output_path = os.path.join(output_dir, f"{image_name}.json")
            save_embedding(temp_image_path, output_path)
            os.remove(temp_image_path)


if __name__ == "__main__":
    main()
