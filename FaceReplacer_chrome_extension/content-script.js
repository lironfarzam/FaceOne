// content-script.js

console.log("🚀 Content script starting...");

// Wait for TF to be ready before initializing
window.addEventListener('tfReady', async () => {
  try {
    console.log("✅ TensorFlow is ready");
    console.log("🔧 Current TF backend:", tf.getBackend());
    console.log("🧠 Starting model loading sequence...");
    
    await loadAllModels();
    
    console.log("🔍 Starting image detection system...");
    observeAndProcessImages();
    
  } catch (err) {
    console.error("❌ Initialization error:", err);
  }
});

// ------------------------------------------------------
// 2) Load face-api models, FaceNet, myModel, embeddings
// ------------------------------------------------------
let faceNetModel, myModel, embeddings;
async function loadAllModels() {
  console.log("📚 Loading face-api models...");
  try {
    await faceapi.nets.ssdMobilenetv1.loadFromUri(chrome.runtime.getURL("models"));
    console.log("✅ SSD MobileNet loaded");
    
    await faceapi.nets.faceLandmark68Net.loadFromUri(chrome.runtime.getURL("models"));
    console.log("✅ Face Landmarks model loaded");

    // Load FaceNet (Facenet512)
    const faceNetModelPath = chrome.runtime.getURL("models/Facenet512_tfjs_graph_model/model.json");
    console.log("📥 Loading FaceNet from:", faceNetModelPath);
    faceNetModel = await tf.loadGraphModel(faceNetModelPath);
    console.log("✅ FaceNet model loaded");

    // Load myModel
    const myModelPath = chrome.runtime.getURL("models/myModel_tfjs_graph_model/model.json");
    console.log("📥 Loading comparison model from:", myModelPath);
    myModel = await tf.loadGraphModel(myModelPath);
    console.log("✅ Comparison model loaded");

    // Load 100 known embeddings
    const embeddingsPath = chrome.runtime.getURL("models/100_positive_embeddings.json");
    console.log("📥 Loading embeddings from:", embeddingsPath);
    const response = await fetch(embeddingsPath);
    embeddings = await response.json();
    console.log("✅ Embeddings loaded:", embeddings.length, "vectors");
    
    console.log("🎯 All models loaded successfully!");
  } catch (err) {
    console.error("❌ Error loading models:", err);
    throw err;
  }
}

// ------------------------------------------------------
// 3) Observe existing + new images, run face detection
// ------------------------------------------------------
function observeAndProcessImages() {
  console.log("👀 Starting image observer...");

  // Process all existing <img> and <image>
  const imgTags = document.querySelectorAll("img");
  console.log("🖼️ Found", imgTags.length, "existing <img> tags");
  imgTags.forEach(img => processImageElement(img));

  const svgImages = document.querySelectorAll("image");
  console.log("🖼️ Found", svgImages.length, "existing SVG images");
  svgImages.forEach(img => processImageElement(img));

  // Observe new additions in DOM
  const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
      if (mutation.type === "childList") {
        mutation.addedNodes.forEach(node => {
          // If node is an element with potential images
          if (node.nodeType === 1) {
            if (node.tagName === "IMG" || node.tagName === "IMAGE") {
              processImageElement(node);
            } else {
              // Check if this element has nested <img>/<image>
              const nestedImgs = node.querySelectorAll?.("img") || [];
              nestedImgs.forEach(img => processImageElement(img));

              const nestedSvgImages = node.querySelectorAll?.("image") || [];
              nestedSvgImages.forEach(img => processImageElement(img));
            }
          }
        });
      }
    });
  });

  observer.observe(document, {
    childList: true,
    subtree: true
  });

  console.log("✅ Image observer initialized");
}

// ------------------------------------------------------
// 4) For each <img> or <image>, run face detection
// ------------------------------------------------------
async function processImageElement(element) {
  try {
    console.log("🔍 Processing image:", getElementSrc(element));

    // Wait for the element to load if it's <img>
    if (element.tagName === "IMG") {
      console.log("⌛ Waiting for image to load...");
      await waitForImgLoad(element);
      console.log("✅ Image loaded");
    }
    // If it's <image> (SVG), there's no real onload. We'll just attempt detection.

    console.log("🔍 Running face detection...");
    const detections = await faceapi
      .detectAllFaces(element, new faceapi.SsdMobilenetv1Options())
      .withFaceLandmarks();

    if (!detections || detections.length === 0) {
      console.log("ℹ️ No faces found in:", getElementSrc(element));
      return;
    }

    console.log(`✨ Found ${detections.length} faces in:`, getElementSrc(element));
    // For each face, do embedding + compare
    for (let i = 0; i < detections.length; i++) {
      console.log(`👤 Processing face #${i + 1}...`);
      const matchCount = await handleFaceDetection(detections[i], element);
      console.log(`✅ Face #${i+1}: ${matchCount}/100 matches (threshold ≥ 0.5)`);
    }
  } catch (err) {
    console.error("❌ Error processing image:", getElementSrc(element), err);
  }
}

// ------------------------------------------------------
// 5) Wait for an <img> to fully load
// ------------------------------------------------------
function waitForImgLoad(img) {
  return new Promise(resolve => {
    if (img.complete && img.naturalWidth > 0) {
      resolve();
    } else {
      img.onload = () => resolve();
      img.onerror = () => resolve();
    }
  });
}

// ------------------------------------------------------
// 6) Crop the face from detection, get embedding, compare
// ------------------------------------------------------
async function handleFaceDetection(detection, element) {
  console.log("🎯 Extracting face region...");
  const { x, y, width, height } = detection.detection.box;
  const size = Math.max(width, height);
  const offsetX = (size - width) / 2;
  const offsetY = (size - height) / 2;

  // Offscreen canvas
  const faceCanvas = document.createElement("canvas");
  faceCanvas.width = 160;
  faceCanvas.height = 160;
  const faceCtx = faceCanvas.getContext("2d");

  // Attempt drawImage
  faceCtx.drawImage(
    element,
    x - offsetX, y - offsetY, size, size,
    0, 0, 160, 160
  );

  console.log("🧬 Getting face embedding...");
  // Get embedding
  const embedding = await getEmbedding(faceCanvas);

  console.log("🔍 Comparing with known embeddings...");
  // Compare
  const matchCount = await compareEmbeddings(embedding, embeddings);
  return matchCount;
}

// ------------------------------------------------------
// 7) Convert 160x160 face canvas to embedding
// ------------------------------------------------------
async function getEmbedding(faceCanvas) {
  const tensor = tf.tidy(() => {
    const imageData = faceCanvas
      .getContext("2d")
      .getImageData(0, 0, faceCanvas.width, faceCanvas.height);
    const decodedImage = tf.browser.fromPixels(imageData);
    return decodedImage.expandDims(0).toFloat().div(255.0);
  });

  const embeddingTensor = faceNetModel.predict(tensor);
  tensor.dispose();

  const embedding = embeddingTensor.arraySync()[0];
  embeddingTensor.dispose();
  return embedding;
}

// ------------------------------------------------------
// 8) Compare with 100 embeddings using myModel
// ------------------------------------------------------
async function compareEmbeddings(embedding, embeddingsArray) {
  let matchCount = 0;
  for (const savedEmbedding of embeddingsArray) {
    const inputTensor = tf.tensor([embedding]);
    const savedTensor = tf.tensor([savedEmbedding]);

    const prediction = myModel.predict([inputTensor, savedTensor]);
    const result = prediction.dataSync()[0]; // e.g. 0.0..1.0

    if (result >= 0.5) {
      matchCount++;
    }

    inputTensor.dispose();
    savedTensor.dispose();
    prediction.dispose();
  }
  return matchCount;
}

// ------------------------------------------------------
// Helper to get <img>.src or <image>'s xlink:href
// ------------------------------------------------------
function getElementSrc(el) {
  if (el.tagName === "IMG") {
    return el.src;
  } else if (el.tagName === "IMAGE") {
    return el.getAttribute("xlink:href");
  }
  return "(unknown element)";
}
