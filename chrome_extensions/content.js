/**
 * @fileoverview Content script for Face API detection Chrome extension.
 * This script provides real-time face detection on web images using Face API.js.
 * @author Liron Farzam
 * @version 1.0.0
 */

/**
 * @typedef {Object} State
 * @property {boolean} modelsLoaded - Indicates if ML models are loaded
 * @property {number} modelLoadAttempts - Number of attempts to load models
 * @property {number} MAX_LOAD_ATTEMPTS - Maximum number of load attempts
 * @property {boolean} isProcessing - Flag to prevent concurrent processing
 * @property {Set<string>} processedImages - Set of processed image URLs
 */

/** @type {State} */
const state = {
  modelsLoaded: false,
  modelLoadAttempts: 0,
  MAX_LOAD_ATTEMPTS: 3,
  isProcessing: false,
  processedImages: new Set()
};

/**
 * Configuration options for Face API detection
 * Optimized for high-accuracy face detection across various scenarios
 * @type {Object}
 */
const FACE_API_DETECTION_OPTIONS = {
  scoreThreshold: 0.1,    // Much lower threshold to catch partial and rotated faces
  inputSize: 1200,         // Even larger input size for better detection of all face sizes
  scaleFactor: 0.99,       // More gradual scaling for better detection at all sizes
  maxNumBoxes: 200,        // Double the max number of detection boxes
  minConfidence: 0.2,      // Very low confidence threshold to catch extreme angles
  iouThreshold: 0.3,       // Lower IOU threshold to detect overlapping faces
  useTinyModel: false,     // Use full model for better accuracy
  minFaceSize: 10         // Detect even very small faces
};

/**
 * Initializes the extension context and ensures Chrome runtime is available
 * @returns {Promise<void>} Resolves when extension context is ready
 */
async function initializeExtensionContext() {
  console.log('initializeExtensionContext()');

  return new Promise((resolve) => {
    if (chrome.runtime && chrome.runtime.id) {
      resolve();
    } else {
      const checkContext = setInterval(() => {
        if (chrome.runtime && chrome.runtime.id) {
          clearInterval(checkContext);
          resolve();
        }
      }, 100);
    }
  });
}

/**
 * Loads the Face API detection models required for the extension
 * @returns {Promise<void>} Resolves when Face API models are loaded
 * @throws {Error} If Face API models fail to load after maximum attempts
 */
async function loadFaceApiModels() {
  console.log('loadFaceApiModels()');
  if (state.modelsLoaded) return;
  
  try {
    await initializeExtensionContext();
    
    state.modelLoadAttempts++;
    const modelPath = chrome.runtime.getURL('models');
    
    // Load only the SSD MobileNet model for optimal face detection
    await faceapi.nets.ssdMobilenetv1.loadFromUri(modelPath);
    
    state.modelsLoaded = true;
    console.log('Face detection model loaded successfully');
  } catch (error) {
    console.error('Error loading face detection model:', error);
    
    if (error.message.includes('Extension context invalidated')) {
      state.modelsLoaded = false;
      if (state.modelLoadAttempts < state.MAX_LOAD_ATTEMPTS) {
        console.log(`Retrying model load, attempt ${state.modelLoadAttempts} of ${state.MAX_LOAD_ATTEMPTS}`);
        await new Promise(resolve => setTimeout(resolve, 1000));
        return loadFaceApiModels();
      }
    }
    throw error;
  }
}

/**
 * Creates a CORS-compatible image object
 * @param {HTMLImageElement} originalImage - The original image element
 * @returns {Promise<HTMLImageElement>} A CORS-enabled image
 */
async function createCORSImage(originalImage) {
  console.log('createCORSImage()');
  return new Promise((resolve, reject) => {
    const corsImage = new Image();
    corsImage.crossOrigin = 'anonymous';
    
    corsImage.onload = () => resolve(corsImage);
    corsImage.onerror = () => {
      console.warn('CORS image load failed, falling back to original image');
      resolve(originalImage);
    };
    
    if (originalImage.src) {
      corsImage.src = originalImage.src;
    } else {
      reject(new Error('No image source found'));
    }
  });
}

/**
 * Detects faces using Face API and draws detection overlays
 * @param {HTMLImageElement} img - The image element to process
 * @returns {Promise<void>}
 */
async function detectFacesWithFaceApi(img) {
  console.log('detectFacesWithFaceApi()');
  if (state.isProcessing) return;
  state.isProcessing = true;
  
  try {
    await loadFaceApiModels();
    
    // Remove existing canvas
    const existingCanvas = document.querySelector('.face-detection-canvas');
    if (existingCanvas) {
      existingCanvas.remove();
    }
    
    // Create wrapper
    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    wrapper.style.display = 'inline-block';
    wrapper.style.width = img.width + 'px';
    wrapper.style.height = img.height + 'px';
    img.parentElement.insertBefore(wrapper, img);
    wrapper.appendChild(img);
    
    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.className = 'face-detection-canvas';
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.pointerEvents = 'none';
    
    // Set canvas dimensions to match displayed image size
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.style.width = img.width + 'px';
    canvas.style.height = img.height + 'px';
    
    const processImage = await createCORSImage(img);
    
    // Calculate scale factors
    const displayToNaturalRatioX = img.naturalWidth / img.width;
    const displayToNaturalRatioY = img.naturalHeight / img.height;
    
    // Detect faces
    const detections = await faceapi.detectAllFaces(
      processImage,
      new faceapi.SsdMobilenetv1Options(FACE_API_DETECTION_OPTIONS)
    );
    
    // Update console messages
    if (detections.length === 0) {
      console.log('Face API: No faces detected');
      return;
    }
    
    console.log(`Face API: Detected ${detections.length} faces`);
    
    // Draw detections
    const ctx = canvas.getContext('2d');
    
    detections.forEach((detection, index) => {
      const box = detection.box;
      
      // Scale coordinates from natural size to display size
      const scaledBox = {
        x: box.x / displayToNaturalRatioX,
        y: box.y / displayToNaturalRatioY,
        width: box.width / displayToNaturalRatioX,
        height: box.height / displayToNaturalRatioY
      };
      
      // Draw face box
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        scaledBox.x,
        scaledBox.y,
        scaledBox.width,
        scaledBox.height
      );
      
      // Draw corner indicators
      const cornerSize = Math.min(scaledBox.width, scaledBox.height) * 0.2; // Relative corner size
      ctx.lineWidth = 2;
      
      // Draw corners
      const corners = [
        // Top-left
        [scaledBox.x, scaledBox.y, scaledBox.x + cornerSize, scaledBox.y],
        [scaledBox.x, scaledBox.y, scaledBox.x, scaledBox.y + cornerSize],
        // Top-right
        [scaledBox.x + scaledBox.width - cornerSize, scaledBox.y, scaledBox.x + scaledBox.width, scaledBox.y],
        [scaledBox.x + scaledBox.width, scaledBox.y, scaledBox.x + scaledBox.width, scaledBox.y + cornerSize],
        // Bottom-left
        [scaledBox.x, scaledBox.y + scaledBox.height - cornerSize, scaledBox.x, scaledBox.y + scaledBox.height],
        [scaledBox.x, scaledBox.y + scaledBox.height, scaledBox.x + cornerSize, scaledBox.y + scaledBox.height],
        // Bottom-right
        [scaledBox.x + scaledBox.width - cornerSize, scaledBox.y + scaledBox.height, scaledBox.x + scaledBox.width, scaledBox.y + scaledBox.height],
        [scaledBox.x + scaledBox.width, scaledBox.y + scaledBox.height - cornerSize, scaledBox.x + scaledBox.width, scaledBox.y + scaledBox.height]
      ];
      
      corners.forEach(([x1, y1, x2, y2]) => {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      });
      
      // Add face number with scaled font size
      const fontSize = Math.max(12, Math.min(scaledBox.width, scaledBox.height) * 0.2);
      ctx.font = `${fontSize}px Arial`;
      const text = `Face ${index + 1}`;
      const textWidth = ctx.measureText(text).width;
      
      // Draw text background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(scaledBox.x, scaledBox.y - fontSize - 8, textWidth + 6, fontSize + 4);
      
      // Draw text
      ctx.fillStyle = 'white';
      ctx.fillText(text, scaledBox.x + 3, scaledBox.y - 6);
    });
    
    wrapper.appendChild(canvas);
  } catch (error) {
    console.error('Face API detection error:', error);
    if (error.message.includes('Extension context invalidated')) {
      state.modelsLoaded = false;
      state.modelLoadAttempts = 0;
    }
  } finally {
    state.isProcessing = false;
  }
}

/**
 * Removes duplicate face detections based on box overlap
 * @param {Array<Object>} detections - Array of face detections
 * @returns {Array<Object>} Filtered array without duplicates
 */
function removeDuplicateDetections(detections) {
  console.log('removeDuplicateDetections()');
  return detections.reduce((unique, detection) => {
    const isDuplicate = unique.some(existing => {
      const boxOverlap = getIntersectionOverUnion(existing.box, detection.box);
      return boxOverlap > 0.45; // Adjust this threshold as needed
    });
    
    if (!isDuplicate) {
      unique.push(detection);
    }
    return unique;
  }, []);
}

/**
 * Calculates Intersection over Union for two bounding boxes
 * @param {Object} box1 - First bounding box
 * @param {Object} box2 - Second bounding box
 * @returns {number} IoU value between 0 and 1
 */
function getIntersectionOverUnion(box1, box2) {
  console.log('getIntersectionOverUnion()');
  const intersection = {
    x: Math.max(box1.x, box2.x),
    y: Math.max(box1.y, box2.y),
    width: Math.min(box1.x + box1.width, box2.x + box2.width) - Math.max(box1.x, box2.x),
    height: Math.min(box1.y + box1.height, box2.y + box2.height) - Math.max(box1.y, box2.y)
  };
  
  if (intersection.width <= 0 || intersection.height <= 0) return 0;
  
  const intersectionArea = intersection.width * intersection.height;
  const box1Area = box1.width * box1.height;
  const box2Area = box2.width * box2.height;
  
  return intersectionArea / (box1Area + box2Area - intersectionArea);
}

/**
 * Prevents text selection when clicking on images
 * @param {Event} e - The event object
 */
function preventTextSelection(e) {
  console.log('preventTextSelection()');
  if (e.target.tagName === 'IMG') {
    e.preventDefault();
    window.getSelection().removeAllRanges();
  }
}

// Event listeners
document.addEventListener('mousedown', preventTextSelection);
document.addEventListener('selectstart', preventTextSelection);

// Modified click handler
let clickTimeout;
document.addEventListener('click', (e) => {
  if (e.target.tagName === 'IMG') {
    e.preventDefault();
    window.getSelection().removeAllRanges();
    
    clearTimeout(clickTimeout);
    clickTimeout = setTimeout(() => {
      // Force reprocess on click even if already processed
      state.processedImages.delete(e.target.src);
      detectFacesWithFaceApi(e.target).catch(error => {
        console.error('Face API click handler error:', error);
      });
    }, 100);
  }
}, { passive: false });

/**
 * Checks if an image is valid for processing
 * @param {HTMLImageElement} img - The image to validate
 * @returns {boolean} True if the image is valid for processing
 */
function isValidImage(img) {
  console.log('isValidImage()');
  return (
    img.width > 50 && // Ignore tiny images
    img.height > 50 &&
    !state.processedImages.has(img.src) &&
    !img.closest('.face-detection-wrapper') // Avoid processing already processed images
  );
}

/**
 * Processes a visible image for face detection
 * @param {HTMLImageElement} img - The image to process
 */
function handleVisibleImage(img) {
  console.log('handleVisibleImage()');
  if (!isValidImage(img)) return;
  
  state.processedImages.add(img.src);
  detectFacesWithFaceApi(img).catch(error => {
    console.error('Face API auto detection error:', error);
    state.processedImages.delete(img.src); // Allow retry on error
  });
}

/**
 * Starts observing images on the page for face detection
 */
function observeImages() {
  console.log('observeImages()');
  const images = document.getElementsByTagName('IMG');
  Array.from(images).forEach(img => {
    if (img.complete) {
      handleVisibleImage(img);
    }
    imageObserver.observe(img);
  });
}

// Create intersection observer
/** @type {IntersectionObserver} */
const imageObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting && entry.target.tagName === 'IMG') {
      handleVisibleImage(entry.target);
    }
  });
}, {
  root: null,
  rootMargin: '50px', // Start loading slightly before images become visible
  threshold: 0.1 // Trigger when at least 10% of the image is visible
});

// Observe new images added to the page
/** @type {MutationObserver} */
const documentObserver = new MutationObserver((mutations) => {
  mutations.forEach(mutation => {
    mutation.addedNodes.forEach(node => {
      if (node.nodeName === 'IMG') {
        if (node.complete) {
          handleVisibleImage(node);
        }
        imageObserver.observe(node);
      }
      
      // Check for images within added nodes
      const images = node.getElementsByTagName && node.getElementsByTagName('IMG');
      if (images) {
        Array.from(images).forEach(img => {
          if (img.complete) {
            handleVisibleImage(img);
          }
          imageObserver.observe(img);
        });
      }
    });
  });
});

// Start observing the document for added images
documentObserver.observe(document.body, {
  childList: true,
  subtree: true
});

// Initialize on page load
window.addEventListener('load', () => {
  // Clear any existing selection
  window.getSelection().removeAllRanges();
  
  initializeExtensionContext()
    .then(() => {
      loadFaceApiModels().then(() => {
        // Start processing existing images
        observeImages();
      });
    })
    .catch(error => console.error('Initialization error:', error));
});