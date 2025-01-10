let faceNetModel, myModel, embeddings;

async function loadModelsAndEmbeddings() {
  console.log("Loading face-api models");
  const modelPromises = [
    faceapi.nets.ssdMobilenetv1.loadFromUri('models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('models'),
  ];
  await Promise.all(modelPromises);
  console.log("Face-api models loaded");

  const faceNetModelPath = "./models/Facenet512_tfjs_graph_model/model.json";
  console.log("Loading TensorFlow faceNetModel from:", faceNetModelPath);
  faceNetModel = await tf.loadGraphModel(faceNetModelPath);
  console.log("TensorFlow faceNetModel loaded");

  const myModelPath = "./models/myModel_tfjs_graph_model/model.json";
  console.log("Loading TensorFlow myModel from:", myModelPath);
  myModel = await tf.loadGraphModel(myModelPath);
  console.log("TensorFlow myModel loaded");

  const embeddingsPath = "./models/100_positive_embeddings.json";
  console.log("Loading embeddings from:", embeddingsPath);
  const response = await fetch(embeddingsPath);
  embeddings = await response.json();
  console.log("Embeddings loaded");
}

document.getElementById("load-image").addEventListener("click", async () => {
  console.log("Load image button clicked");
  const imageUrl = document.getElementById("image-url").value;
  if (!imageUrl) return alert("Please enter a valid image URL!");

  console.log("Image URL:", imageUrl);

  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const facesContainer = document.getElementById("faces");

  facesContainer.innerHTML = ""; // Clear previous faces

  const img = new Image();
  img.crossOrigin = "anonymous"; // Enable CORS
  img.src = imageUrl;

  img.onload = async () => {
    console.log("Image loaded successfully");
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    // Detect faces with SSD MobileNet v1
    const detectionsSSD = await faceapi.detectAllFaces(img, new faceapi.SsdMobilenetv1Options()).withFaceLandmarks();
    console.log("SSD MobileNet Detections:", detectionsSSD);

    // Process detections with SSD MobileNet v1
    const facePromisesSSD = detectionsSSD.map(detection => processFaceDetection(detection, img, ctx, facesContainer, "green"));
    await Promise.all(facePromisesSSD);
  };

  img.onerror = () => alert("Failed to load image. Please check the URL.");
});

async function processFaceDetection(detection, img, ctx, facesContainer, color) {
  const { x, y, width, height } = detection.detection.box;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, width, height);

  // Calculate square dimensions
  const size = Math.max(width, height);
  const offsetX = (size - width) / 2;
  const offsetY = (size - height) / 2;

  // Create face crop
  const faceCanvas = document.createElement("canvas");
  faceCanvas.width = 160;
  faceCanvas.height = 160;
  const faceCtx = faceCanvas.getContext("2d");
  faceCtx.drawImage(
    img,
    x - offsetX, y - offsetY, size, size, // Source coordinates
    0, 0, 160, 160  // Destination coordinates (enlarged to 160x160)
  );
  const faceDiv = document.createElement("div");
  faceDiv.classList.add("face");
  faceDiv.appendChild(faceCanvas);
  faceDiv.appendChild(document.createElement("br"));
  faceDiv.appendChild(document.createTextNode(`Size: 160x160`));
  facesContainer.appendChild(faceDiv);

  console.log("Face cropped and added to container");

  const embedding = await getEmbedding(faceNetModel, faceCanvas);
  // console.log(`Embedding for face: ${JSON.stringify(embedding)}`);

  const matchCount = await compareEmbeddings(embedding, embeddings);
  console.log(`Number of matches: ${matchCount} out of 100`);
  faceDiv.appendChild(document.createElement("br"));
  faceDiv.appendChild(document.createTextNode(`Matches: ${matchCount} out of 100`));
}

async function getEmbedding(faceNetModel, faceCanvas) {
  console.log("Generating embedding for face");
  const tensor = tf.tidy(() => {
    const imageData = faceCanvas.getContext("2d").getImageData(0, 0, faceCanvas.width, faceCanvas.height);
    const decodedImage = tf.browser.fromPixels(imageData);
    return decodedImage.expandDims(0).toFloat().div(255.0); // Normalized tensor
  });
  const embedding = faceNetModel.predict(tensor);
  tensor.dispose(); // Free memory
  console.log("Embedding generated");
  return embedding.arraySync()[0];
}

async function compareEmbeddings(embedding, embeddings) {
  console.log("Comparing embeddings");
  let matchCount = 0;
  for (const savedEmbedding of embeddings) {
    const inputTensor = tf.tensor([embedding]);
    const savedTensor = tf.tensor([savedEmbedding]);
    const prediction = myModel.predict([inputTensor, savedTensor]);
    console.log("Prediction:", prediction.dataSync());
    const result = prediction.dataSync()[0];
    if (result >= 0.5) {
      matchCount++;
    }
    inputTensor.dispose();
    savedTensor.dispose();
    prediction.dispose();
  }
  console.log("Comparison complete");
  return matchCount;
}

function clearCanvas() {
  console.log("Clearing canvas and faces container");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const facesContainer = document.getElementById("faces");
  facesContainer.innerHTML = "";
}

// Load models and embeddings on page load
window.onload = loadModelsAndEmbeddings;
