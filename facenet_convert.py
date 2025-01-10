import os
import tensorflow as tf
from deepface import DeepFace
import tensorflowjs as tfjs


def load_and_wrap_facenet512(model_name="Facenet512"):
    """
    Load the specified DeepFace model and wrap it as a TensorFlow Keras model.

    Parameters:
    - model_name (str): The name of the DeepFace model to load (default: "Facenet512").

    Returns:
    - model: A TensorFlow Keras model instance.
    """
    print(f"Loading DeepFace model: {model_name}...")
    deepface_model = DeepFace.build_model(model_name)

    # Check if the model has a `model` attribute (common in DeepFace models)
    if hasattr(deepface_model, "model"):
        tf_model = deepface_model.model  # Extract the Keras model
    else:
        raise TypeError(
            f"The model returned by DeepFace.build_model is not callable or lacks a 'model' attribute."
        )

    # Wrap the extracted model into a TensorFlow Keras model
    input_layer = tf.keras.Input(shape=(160, 160, 3))  # Assuming 160x160 RGB images
    embeddings = tf_model(input_layer)  # Pass input through the extracted model
    model = tf.keras.Model(inputs=input_layer, outputs=embeddings)

    print(f"{model_name} model loaded and wrapped successfully.")
    return model


def save_and_convert_deepface_model(model_name="Facenet512", base_path="./model"):
    """
    Save the pre-trained DeepFace model in two formats:
      1) tfjs_layers_model (from the single-file `.keras`)
      2) tfjs_graph_model (from the SavedModel directory)

    Parameters:
    - model_name (str): Name of the DeepFace model (e.g., "Facenet512").
    - base_path (str): Base directory to save the model.
    """
    print("-" * 50)
    # Load and wrap the DeepFace model
    model = load_and_wrap_facenet512(model_name=model_name)

    # ---------------------------------------------------------
    # Save as Keras single-file model -> tfjs_layers_model
    # ---------------------------------------------------------
    print("-" * 50)
    keras_filepath = os.path.join(base_path, "keras", f"{model_name}_model.keras")
    layers_output_path = os.path.join(base_path, f"{model_name}_tfjs_layers_model")

    # Save the model in Keras single-file format
    os.makedirs(os.path.dirname(keras_filepath), exist_ok=True)
    tf.keras.models.save_model(model, keras_filepath)
    print(f"Keras model saved at: {keras_filepath}")

    # Convert to tfjs_layers_model
    print(f"Converting {model_name} to tfjs_layers_model...")
    tfjs.converters.save_keras_model(model, layers_output_path)
    print("tfjs_layers_model created successfully.")

    # ---------------------------------------------------------
    # Save as TensorFlow SavedModel -> tfjs_graph_model
    # ---------------------------------------------------------
    print("-" * 50)
    saved_model_dir = os.path.join(base_path, "saved_model", f"{model_name}_model")
    graph_output_path = os.path.join(base_path, f"{model_name}_tfjs_graph_model")

    # Save the model in SavedModel format
    os.makedirs(saved_model_dir, exist_ok=True)
    tf.saved_model.save(model, saved_model_dir)
    print(f"Model saved in SavedModel format at: {saved_model_dir}")

    # Convert to tfjs_graph_model
    print(f"Converting {model_name} to tfjs_graph_model...")
    conversion_command = f"""
        tensorflowjs_converter \
        --input_format=tf_saved_model \
        --saved_model_tags=serve \
        --output_format=tfjs_graph_model \
        "{saved_model_dir}" \
        "{graph_output_path}"
    """
    os.system(conversion_command)
    print("tfjs_graph_model created successfully.")
    print("-" * 50)


# Example Usage
if __name__ == "__main__":
    save_and_convert_deepface_model(
        model_name="Facenet512", base_path="./model/FaceNet/"
    )
