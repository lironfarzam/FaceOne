import os
import json
import numpy as np


def load_embedding(filepath):
    with open(filepath, "r") as f:
        return np.array(json.load(f))


def compare_embeddings(embedding_dir_py, embedding_dir_js):
    py_files = sorted([f for f in os.listdir(embedding_dir_py) if f.endswith(".json")])
    js_files = sorted([f for f in os.listdir(embedding_dir_js) if f.endswith(".json")])

    for py_file, js_file in zip(py_files, js_files):
        embedding_py = load_embedding(os.path.join(embedding_dir_py, py_file))
        embedding_js = load_embedding(os.path.join(embedding_dir_js, js_file))

        if np.allclose(embedding_py, embedding_js, atol=1e-5):
            print(f"{py_file} and {js_file} are similar.")
        else:
            print(f"{py_file} and {js_file} are different.")


def main():
    embedding_dir_py = "./embeddings_py"
    embedding_dir_js = "./embeddings_js"
    compare_embeddings(embedding_dir_py, embedding_dir_js)


if __name__ == "__main__":
    main()
