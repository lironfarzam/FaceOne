const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");
const cv = require("@techstark/opencv-js"); // OpenCV.js for Node.js
const path = require("path");
const sharp = require("sharp");


async function preprocessImage(imagePath) {
  const buffer = await sharp(imagePath).resize(160, 160).toBuffer();
  const tensor = tf.tidy(() => {
    const decodedImage = tf.node.decodeImage(buffer, 3);
    return decodedImage.expandDims(0).toFloat().div(255.0); // Normalized tensor
  });
  return tensor;
}

async function loadModel(modelPath) {
  return await tf.loadGraphModel("file://" + modelPath);
}

async function getEmbedding(model, imagePath) {
  const tensor = await preprocessImage(imagePath);
  const embedding = model.predict(tensor);
  tensor.dispose(); // Free memory
  return embedding.arraySync()[0];
}

async function saveEmbedding(imagePath, outputPath, model) {
  const embedding = await getEmbedding(model, imagePath);
  fs.writeFileSync(outputPath, JSON.stringify(embedding));
  console.log(`Embedding saved to ${outputPath}`);
}

async function main() {
  const imageDir = "../imgs/test_imgs";
  const outputDir = "./embeddings_js";
  const modelPath = path.resolve(__dirname, "./model/FaceNet/Facenet512_tfjs_graph_model/model.json");

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
  }

  const model = await loadModel(modelPath);

  const imageFiles = fs.readdirSync(imageDir).filter(file => file.endsWith(".jpeg"));
  for (const imageName of imageFiles) {
    const imagePath = path.join(imageDir, imageName);
    const outputPath = path.join(outputDir, `${imageName}.json`);
    await saveEmbedding(imagePath, outputPath, model);
  }
}

main().catch(err => console.error(err));
