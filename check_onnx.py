#!/usr/bin/env python3

import os
import sys
import onnx


def check_onnx_model(model_path):
    """Check if an ONNX model is valid."""
    try:
        # Load the ONNX model
        print(f"Loading ONNX model from {model_path}")

        if not os.path.exists(model_path):
            print(f"ERROR: File does not exist: {model_path}")
            return False

        # Get file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
        print(f"File size: {file_size:.2f} MB")

        # Try to load the model
        model = onnx.load(model_path)

        # Check the model
        try:
            onnx.checker.check_model(model)
            print("The model is valid!")
            return True
        except onnx.checker.ValidationError as e:
            print(f"The model is invalid: {e}")
            return False
    except Exception as e:
        print(f"Error checking the model: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_onnx.py <path_to_onnx_model>")
        sys.exit(1)

    model_path = sys.argv[1]
    result = check_onnx_model(model_path)

    if not result:
        sys.exit(1)  # Return non-zero exit code if model is invalid
