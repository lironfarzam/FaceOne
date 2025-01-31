# get the facenet512 model and print the model summary
import tensorflow as tf
from deepface import DeepFace

model = DeepFace.build_model("Facenet512")
keras_model = model.model
keras_model.summary()
