const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");
const path = require("path");
const sharp = require("sharp");

console.log("Node.js script started.");

class L1DistanceLayer extends tf.layers.Layer {
  call(inputs) {
    const [input1, input2] = inputs;
    return tf.abs(tf.sub(input1, input2));
  }

  static get className() {
    return "L1DistanceLayer";
  }
}
tf.serialization.registerClass(L1DistanceLayer);

const MODEL_TYPE = "tfjs_graph_model"; // Change to "tfjs_layers_model" or "tfjs_graph_model"

let MODEL_PATH;
if (MODEL_TYPE === "tfjs_graph_model") {
  MODEL_PATH = path.resolve(__dirname, "../model/myModel/tfjs_graph_model/model.json");
} else if (MODEL_TYPE === "tfjs_layers_model") {
  MODEL_PATH = path.resolve(__dirname, "../model/myModel/tfjs_layers_model/model.json");
} else {
  throw new Error("Invalid MODEL_TYPE specified.");
}

console.log("Loading model from:", MODEL_PATH);

async function loadModel(modelPath, modelType) {
  try {
    const content = fs.readFileSync(modelPath, "utf8");
    console.log("model.json size:", content.length, "chars");

    console.log("Loading model...");
    let model;
    if (modelType === "tfjs_graph_model") {
      model = await tf.loadGraphModel("file://" + modelPath);
    } else if (modelType === "tfjs_layers_model") {
      model = await tf.loadLayersModel("file://" + modelPath);
    } else {
      throw new Error("Invalid MODEL_TYPE specified.");
    }
    console.log("Model loaded successfully.");
    return model;
  } catch (error) {
    console.error("Error loading model:", error);
  }
}

async function preprocessImage(imagePath) {
  try {
    const buffer = await sharp(imagePath).resize(160, 160).toBuffer();
    const tensor = tf.tidy(() => {
      const decodedImage = tf.node.decodeImage(buffer, 3);
      return decodedImage.expandDims(0).toFloat().div(255.0); // Normalized tensor
    });

    return tensor;
  } catch (error) {
    console.error("Error preprocessing image:", error);
  }
}

async function loadImageAndGetEmbedding(model, imagePath) {
  try {
    const tensor = await preprocessImage(imagePath);
    if (!tensor) {
      console.log("Failed to preprocess image.");
      return null;
    }

    // Get embedding using the loaded model
    const embedding = model.predict(tensor);
    tensor.dispose(); // Free memory

    return embedding;
  } catch (error) {
    console.error("Error during embedding generation:", error);
    return null;
  }
}

async function loadConfig(configPath) {
  try {
    const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
    console.log("Configuration loaded:", config);
    return config;
  } catch (error) {
    console.error("Error loading configuration:", error);
  }
}

async function main() {
  // Load configuration
  const config = await loadConfig(path.resolve(__dirname, "../config.json"));

  // Load trained model
  const model = await loadModel(MODEL_PATH, MODEL_TYPE);
  if (!model) {
    console.error("Model loading failed.");
    return;
  }

  // Load Facenet512 model
  let deepFaceNetModel;
  try {
    const FACENET_MODEL_PATH = path.resolve(
      __dirname,
      "../model/FaceNet/Facenet512_tfjs_graph_model/model.json"
    );
    deepFaceNetModel = await tf.loadGraphModel("file://" + FACENET_MODEL_PATH);
    console.log("Facenet512 model loaded successfully.");
  } catch (error) {
    console.error("Error loading Facenet512 model:", error);
    return;
  }

  // Process images and compute embeddings
  const imagePath_A = path.resolve(__dirname, "../imgs/test_imgs/A.jpeg");
  const imagePath_B = path.resolve(__dirname, "../imgs/test_imgs/B.jpeg");

  console.log("Processing image:", imagePath_A);
  const embedding_A = await loadImageAndGetEmbedding(deepFaceNetModel, imagePath_A);
  if (embedding_A) {
    console.log("Embedding vector:", embedding_A.arraySync());
  } else {
    console.log("No face detected in the image.");
  }

  console.log("Processing image:", imagePath_B);
  const embedding_B = await loadImageAndGetEmbedding(deepFaceNetModel, imagePath_B);
  if (embedding_B) {
    console.log("Embedding vector:", embedding_B.arraySync());
  } else {
    console.log("No face detected in the image.");
  }

  // use the model to predict the similarity between the embeddings
  const distance = model.predict([embedding_A, embedding_B]);

  // Free memory
  embedding_A.dispose();
  embedding_B.dispose();
  model.dispose();
  deepFaceNetModel.dispose();

  



  console.log("Distance between embeddings:", distance.arraySync());
}

main().catch((err) => console.error(err));
