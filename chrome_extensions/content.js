/**
 * @fileoverview Content script for Face API detection Chrome extension.
 * This script provides real-time face detection on web images using Face API.js.
 * @author Liron Farzam
 * @version 1.0.0
 */

/**
 * @typedef {Object} State
 * @property {boolean} modelsLoaded - Indicates if ML models are loaded
 * @property {boolean} faceNetLoaded - Indicates if FaceNet model is loaded
 * @property {number} modelLoadAttempts - Number of attempts to load models
 * @property {number} MAX_LOAD_ATTEMPTS - Maximum number of load attempts
 * @property {boolean} isProcessing - Flag to prevent concurrent processing
 * @property {Set<string>} processedImages - Set of processed image URLs
 */

// Update flagShowFrameonImage to use Chrome storage
let flagShowFrameonImage = {
  frameProsessedImage: true,    // Controls green frame around processed images
  frameFaceDetected: true,      // Controls red frame around face detected
  addLabel: true,               // Controls face number label
  autoProcessImages: true,      // Controls automatic processing of all images
  minimumImageSize: 100         // Increased minimum size for better reliability
};

// Load settings from Chrome storage
chrome.storage.sync.get({
  // Default values
  frameProsessedImage: true,
  frameFaceDetected: true,
  addLabel: true,
  autoProcessImages: true,
  minimumImageSize: 100
}, (items) => {
  flagShowFrameonImage = items;
});

// Listen for settings updates
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'UPDATE_SETTINGS') {
    const oldSettings = {...flagShowFrameonImage};
    flagShowFrameonImage = message.settings;
    
    // Clear frames if any visual settings were turned off
    if (
      (!flagShowFrameonImage.frameProsessedImage && oldSettings.frameProsessedImage) ||
      (!flagShowFrameonImage.frameFaceDetected && oldSettings.frameFaceDetected) ||
      (!flagShowFrameonImage.addLabel && oldSettings.addLabel)
    ) {
      clearAllFrames();
    }
    
    // If auto-processing is enabled, reprocess visible images
    if (flagShowFrameonImage.autoProcessImages) {
      observeElements();
    }
  }
});

/** @type {State} */
const state = {
  modelsLoaded: false,
  faceNetLoaded: false,
  modelLoadAttempts: 0,
  MAX_LOAD_ATTEMPTS: 3,
  isProcessing: false,
  processedImages: new Set(),
  processingQueue: [], // Add queue for parallel processing
  maxParallelProcessing: 3 // Maximum number of parallel processes
};

let faceNetModel = null;

// Add sandbox iframe management
let sandboxFrame = null;

function createSandboxFrame() {
  if (sandboxFrame) return;
  
  sandboxFrame = document.createElement('iframe');
  sandboxFrame.src = chrome.runtime.getURL('sandbox.html');
  sandboxFrame.style.display = 'none';
  document.body.appendChild(sandboxFrame);
}

// Add cache for processed images
const imageCache = {
  embeddings: new Map(),
  maxSize: 100,  // Maximum number of cached embeddings
  
  add(src, embedding) {
    if (this.embeddings.size >= this.maxSize) {
      const firstKey = this.embeddings.keys().next().value;
      this.embeddings.delete(firstKey);
    }
    this.embeddings.set(src, embedding);
  },
  
  get(src) {
    return this.embeddings.get(src);
  },
  
  has(src) {
    return this.embeddings.has(src);
  }
};

// Optimize queue processing with batching and prioritization
const processingQueue = {
  items: [],
  processing: false,
  batchSize: 3,  // Process 3 images at a time
  
  add(element, priority = false) {
    const item = { element, priority };
    if (priority) {
      this.items.unshift(item);
    } else {
      this.items.push(item);
    }
    this.process();
  },
  
  async process() {
    if (this.processing || this.items.length === 0) return;
    
    this.processing = true;
    while (this.items.length > 0) {
      const batch = this.items.splice(0, this.batchSize);
      const promises = batch.map(async ({ element }) => {
        try {
          const src = element.tagName === 'IMG' ? element.src : element.getAttribute('xlink:href');
          
          // Check cache first
          if (imageCache.has(src)) {
            const embedding = imageCache.get(src);
            // Handle cached embedding (e.g., display visualization)
            return;
          }
          
          await detectFacesWithFaceApi(element);
        } catch (error) {
          console.error('Processing error:', error);
          const src = element.tagName === 'IMG' ? element.src : element.getAttribute('xlink:href');
          state.processedImages.delete(src);
        }
      });
      
      await Promise.all(promises);
    }
    this.processing = false;
  }
};

/**
 * Configuration options for Face API detection
 * Optimized for high-accuracy face detection across various scenarios
 * @type {Object}
 */
const FACE_API_DETECTION_OPTIONS = {
  scoreThreshold: 0.3,     // Increased threshold for more reliable detections
  inputSize: 320,          // Standard size for face-api.js
  scaleFactor: 0.8,        // Better balance of speed and accuracy
  maxNumBoxes: 100,        // Reasonable limit for most use cases
  minConfidence: 0.3,      // Higher confidence for more reliable detections
  iouThreshold: 0.5,       // Standard IOU threshold
  useTinyModel: false,     // Use full model for better accuracy
  minFaceSize: 20         // Minimum face size in pixels
};

/**
 * Initializes the extension context and ensures Chrome runtime is available
 * @returns {Promise<void>} Resolves when extension context is ready
 */
async function initializeExtensionContext() {

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
 * Loads the FaceNet model for face embedding generation
 * @returns {Promise<void>} Resolves when FaceNet model is loaded
 */
async function loadFaceNetModel() {
  if (state.faceNetLoaded) return;
  
  try {
    createSandboxFrame();
    
    // Wait for sandbox to be ready
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const modelPath = chrome.runtime.getURL('models/FaceNet/Facenet512_tfjs_graph_model/model.json');
    // console.log('Requesting FaceNet model load from sandbox, path:', modelPath);
    
    return new Promise((resolve, reject) => {
      const handleMessage = (event) => {
        if (event.data.type === 'MODEL_LOADED') {
          window.removeEventListener('message', handleMessage);
          if (event.data.success) {
            // console.log('FaceNet model loaded successfully with info:', event.data.modelInfo);
            state.faceNetLoaded = true;
            resolve();
          } else {
            console.error('FaceNet model loading failed:', event.data.error);
            reject(new Error(event.data.error));
          }
        }
      };
      
      window.addEventListener('message', handleMessage);
      sandboxFrame.contentWindow.postMessage({ 
        type: 'LOAD_MODEL',
        modelPath: modelPath
      }, '*');
    });
  } catch (error) {
    console.error('Error loading FaceNet model:', error);
    throw error;
  }
}

/**
 * Loads all required models for face detection and embedding
 * @returns {Promise<void>} Resolves when all models are loaded
 */
async function loadFaceApiModels() {
  if (state.modelsLoaded && state.faceNetLoaded) return;
  
  try {
    await initializeExtensionContext();
    
    state.modelLoadAttempts++;
    const modelPath = chrome.runtime.getURL('models');
    
    // Load face detection model
    // console.log('Loading Face API model...');
    await faceapi.nets.ssdMobilenetv1.loadFromUri(modelPath);
    state.modelsLoaded = true;
    // console.log('Face detection model loaded successfully');
    
    // Load FaceNet model
    // console.log('Loading FaceNet model...');
    await loadFaceNetModel();
    // console.log('All models loaded successfully');
    
  } catch (error) {
    console.error('Error loading models:', error);
    
    if (error.message.includes('Extension context invalidated')) {
      state.modelsLoaded = false;
      state.faceNetLoaded = false;
      if (state.modelLoadAttempts < state.MAX_LOAD_ATTEMPTS) {
        // console.log(`Retrying model load, attempt ${state.modelLoadAttempts} of ${state.MAX_LOAD_ATTEMPTS}`);
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
  return new Promise((resolve, reject) => {
    const corsImage = new Image();
    corsImage.crossOrigin = 'anonymous';
    
    corsImage.onload = () => resolve(corsImage);
    corsImage.onerror = () => {
      // console.warn('CORS image load failed, falling back to original image');
      resolve(originalImage);
    };
    
    if (originalImage.src) {
      corsImage.src = originalImage.src;
    } else {
      reject(new Error('No image source found'));
    }
  });
}

// Add new function to create a clean scaled image
async function createScaledImage(img) {
  return new Promise((resolve, reject) => {
    const scaledImg = new Image();
    scaledImg.crossOrigin = 'anonymous';
    
    scaledImg.onload = () => {
      // Calculate scale factor (make image at least 400px in smallest dimension)
      const targetSize = 400;
      const scale = Math.max(
        targetSize / scaledImg.naturalWidth,
        targetSize / scaledImg.naturalHeight
      );
      
      scaledImg.scaleFactor = scale;
      resolve(scaledImg);
    };
    
    scaledImg.onerror = () => {
      // If CORS fails, try without it
      const fallbackImg = new Image();
      fallbackImg.onload = () => {
        fallbackImg.scaleFactor = 1;
        resolve(fallbackImg);
      };
      fallbackImg.onerror = reject;
      fallbackImg.src = img.src;
    };
    
    // Try to load with CORS first
    scaledImg.src = img.src;
  });
}

async function extractFaceRegion(img, detection) {
  try {
    // Create a temporary canvas for the full image
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = img.width;
    tempCanvas.height = img.height;

    // Create a new image with crossOrigin attribute
    const corsImage = new Image();
    corsImage.crossOrigin = 'anonymous';
    
    await new Promise((resolve, reject) => {
      corsImage.onload = resolve;
      corsImage.onerror = () => {
        // console.warn('CORS image load failed, attempting without CORS');
        reject(new Error('CORS load failed'));
      };
      corsImage.src = img.src;
    }).catch(() => {
      // If CORS fails, try to use the original image
      // Note: This might still fail for cross-origin images
      return new Promise((resolve) => {
        corsImage.onload = resolve;
        corsImage.removeAttribute('crossOrigin');
        // Try to force a reload without CORS
        corsImage.src = img.src + (img.src.includes('?') ? '&' : '?') + 'nocors=' + Date.now();
      });
    });

    // Draw the CORS-enabled image to the temporary canvas
    tempCtx.drawImage(corsImage, 0, 0, img.width, img.height);
    
    // Create a canvas for the face region
    const faceCanvas = document.createElement('canvas');
    const ctx = faceCanvas.getContext('2d');
    
    // Calculate the square region (use the larger of width/height)
    const size = Math.max(detection.box.width, detection.box.height);
    const centerX = detection.box.x + detection.box.width / 2;
    const centerY = detection.box.y + detection.box.height / 2;
    
    // Add padding to the face region (20% on each side)
    const padding = size * 0.2;
    const paddedSize = size + (padding * 2);
    
    // Set canvas size to the padded square dimensions
    faceCanvas.width = paddedSize;
    faceCanvas.height = paddedSize;
    
    // Draw the face region onto the canvas
    ctx.drawImage(
      tempCanvas,
      centerX - paddedSize/2,  // source x
      centerY - paddedSize/2,  // source y
      paddedSize,             // source width
      paddedSize,             // source height
      0,                      // dest x
      0,                      // dest y
      paddedSize,             // dest width
      paddedSize              // dest height
    );
    
    // Resize to 160x160 (FaceNet input size)
    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = 160;
    resizedCanvas.height = 160;
    const resizedCtx = resizedCanvas.getContext('2d');
    
    // Use better quality image scaling
    resizedCtx.imageSmoothingEnabled = true;
    resizedCtx.imageSmoothingQuality = 'high';
    resizedCtx.drawImage(faceCanvas, 0, 0, 160, 160);
    
    // Clean up
    tempCanvas.remove();
    faceCanvas.remove();
    
    return resizedCanvas;
  } catch (error) {
    // console.error('Error in extractFaceRegion:', error);
    throw error;
  }
}

// Optimize face detection with progressive loading
async function detectFacesWithFaceApi(img) {
  try {
    await loadFaceApiModels();
    
    const src = img.tagName === 'IMG' ? img.src : img.getAttribute('xlink:href');
    
    // Show loading indicator
    const wrapper = createWrapper(img);
    if (flagShowFrameonImage.addLabel) {
      addLoadingIndicator(wrapper);
    }
    
    // Scale and process image
    const scaledImg = await createScaledImage(img);
    const detections = await faceapi.detectAllFaces(
      scaledImg,
      new faceapi.SsdMobilenetv1Options({
        ...FACE_API_DETECTION_OPTIONS,
        scoreThreshold: 0.4  // Slightly higher threshold for better accuracy
      })
    );
    
    // Scale back detections
    if (scaledImg.scaleFactor !== 1) {
      scaleDetections(detections, scaledImg.scaleFactor);
    }
    
    // Process faces and generate embeddings
    for (const detection of detections) {
      const faceCanvas = await extractFaceRegion(img, detection);
      try {
        const embedding = await generateEmbedding(faceCanvas);
        imageCache.add(src, embedding);
      } catch (error) {
        console.error('Embedding generation error:', error);
      }
    }
    
    // Update visualization
    updateVisualization(wrapper, img, detections);
    
  } catch (error) {
    console.error('Face detection error:', error);
    throw error;
  }
}

// Helper functions for improved visualization
function createWrapper(img) {
  const wrapper = document.createElement('div');
  wrapper.className = 'face-detection-wrapper';
  wrapper.style.position = 'relative';
  wrapper.style.display = 'inline-block';
  wrapper.style.width = img.width + 'px';
  wrapper.style.height = img.height + 'px';
  
  if (flagShowFrameonImage.frameProsessedImage) {
    wrapper.style.border = '3px solid #00ff00';
    wrapper.style.boxSizing = 'border-box';
    wrapper.style.padding = '2px';
  }
  
  img.parentElement.insertBefore(wrapper, img);
  wrapper.appendChild(img);
  return wrapper;
}

function addLoadingIndicator(wrapper) {
  const indicator = document.createElement('div');
  indicator.className = 'processing-indicator';
  indicator.style.position = 'absolute';
  indicator.style.top = '5px';
  indicator.style.right = '5px';
  indicator.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
  indicator.style.color = 'white';
  indicator.style.padding = '2px 5px';
  indicator.style.borderRadius = '3px';
  indicator.style.fontSize = '12px';
  indicator.textContent = 'Processing...';
  wrapper.appendChild(indicator);
}

function updateVisualization(wrapper, img, detections) {
  // Remove existing canvas and indicators
  const existingCanvas = wrapper.querySelector('.face-detection-canvas');
  const processingIndicator = wrapper.querySelector('.processing-indicator');
  if (existingCanvas) existingCanvas.remove();
  if (processingIndicator) processingIndicator.remove();
  
  if (detections.length === 0) {
    if (flagShowFrameonImage.addLabel) {
      addResultIndicator(wrapper, 'No Faces Detected');
    }
    return;
  }
  
  // Create and setup canvas
  const canvas = createDetectionCanvas(img);
  if (flagShowFrameonImage.frameFaceDetected) {
    drawDetections(canvas, detections, img);
  }
  
  if (flagShowFrameonImage.addLabel) {
    addResultIndicator(wrapper, `${detections.length} Face(s) Detected`);
  }
  
  wrapper.appendChild(canvas);
}

/**
 * Removes duplicate face detections based on box overlap
 * @param {Array<Object>} detections - Array of face detections
 * @returns {Array<Object>} Filtered array without duplicates
 */
function removeDuplicateDetections(detections) {
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
 * Checks if an element is valid for processing
 * @param {Element} element - The element to validate
 * @returns {boolean} True if the element is valid for processing
 */
function isValidElement(element) {
  const isImg = element.tagName === 'IMG';
  const isSvgImage = element.tagName === 'image';
  
  if (!isImg && !isSvgImage) return false;
  
  const src = isImg ? element.src : element.getAttribute('xlink:href');
  const width = element.width || element.clientWidth || parseInt(element.getAttribute('width')) || 0;
  const height = element.height || element.clientHeight || parseInt(element.getAttribute('height')) || 0;
  
  const hasValidSize = (width >= flagShowFrameonImage.minimumImageSize && 
                       height >= flagShowFrameonImage.minimumImageSize);
  
  return (
    src && 
    hasValidSize &&
    !state.processedImages.has(src) &&
    !element.closest('.face-detection-wrapper')
  );
}

/**
 * Processes a visible element for face detection
 * @param {Element} element - The element to process
 */
function handleVisibleElement(element) {
  if (!isValidElement(element) || !flagShowFrameonImage.autoProcessImages) return;
  
  const src = element.tagName === 'IMG' ? element.src : element.getAttribute('xlink:href');
  if (!state.processedImages.has(src)) {
    state.processedImages.add(src);
    processingQueue.add(element);
  }
}

/**
 * Starts observing elements on the page for face detection
 */
function observeElements() {
  const elements = [
    ...Array.from(document.getElementsByTagName('img')),
    ...Array.from(document.getElementsByTagName('image')),
    ...Array.from(document.querySelectorAll('image[xlink\\:href^="https://"]'))
  ];
  
  // Additional selector for Facebook-style images
  const fbImages = document.querySelectorAll('image[preserveAspectRatio="xMidYMid slice"][xlink\\:href^="https://"]');
  elements.push(...Array.from(fbImages));
  
  // console.log(`Found ${elements.length} images to process`);
  
  elements.forEach(element => {
    if (element.complete || element.tagName === 'image') {
      handleVisibleElement(element);
    }
    imageObserver.observe(element);
  });
}

// Create intersection observer
/** @type {IntersectionObserver} */
const imageObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting && (entry.target.tagName === 'IMG' || entry.target.tagName === 'image')) {
      handleVisibleElement(entry.target);
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
    if (mutation.type === 'childList') {
      mutation.addedNodes.forEach(node => {
        // Immediately process any new image nodes
        if (node.tagName === 'IMG' || node.tagName === 'image') {
          // console.log('New image detected:', node);
          // Process immediately without waiting for intersection
          handleVisibleElement(node);
          // Also observe for future changes
          imageObserver.observe(node);
        }
        
        // Check for images within added nodes
        if (node.querySelectorAll) {
          ['IMG', 'image'].forEach(tagName => {
            const elements = node.querySelectorAll(tagName);
            elements.forEach(element => {
              // console.log('New nested image detected:', element);
              // Process immediately
              handleVisibleElement(element);
              // Also observe for future changes
              imageObserver.observe(element);
            });
          });
        }
      });
    }
    // Also check for attribute changes that might affect images
    else if (mutation.type === 'attributes') {
      const target = mutation.target;
      if ((target.tagName === 'IMG' || target.tagName === 'image') && 
          (mutation.attributeName === 'src' || mutation.attributeName === 'xlink:href')) {
        // console.log('Image source changed:', target);
        // Re-process when src changes
        state.processedImages.delete(target.src || target.getAttribute('xlink:href'));
        handleVisibleElement(target);
      }
    }
  });
});

// Start observing with enhanced options
documentObserver.observe(document.body, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeFilter: ['src', 'xlink:href'] // Only watch relevant attributes
});

// Initialize on page load
window.addEventListener('load', () => {
  // Clear any existing selection
  window.getSelection().removeAllRanges();
  
  // console.log('Starting automatic image processing...');
  initializeExtensionContext()
    .then(() => {
      loadFaceApiModels().then(() => {
        if (flagShowFrameonImage.autoProcessImages) {
          observeElements();
          // console.log('Automatic image processing enabled');
        }
      });
    })
    .catch(error => console.error('Initialization error:', error));
});

// Remove or comment out the click handler if you want only automatic processing
// document.addEventListener('click', (e) => { ... });

/**
 * Generates embedding for a face image using FaceNet
 * @param {HTMLCanvasElement} faceCanvas - Canvas containing the face image
 * @returns {Promise<Float32Array>} Face embedding vector
 */
async function generateEmbedding(faceCanvas) {
  if (!state.faceNetLoaded) {
    throw new Error('FaceNet model not loaded');
  }

  const imageData = faceCanvas.getContext("2d").getImageData(0, 0, faceCanvas.width, faceCanvas.height);
  
  return new Promise((resolve, reject) => {
    const handleMessage = (event) => {
      if (event.data.type === 'EMBEDDING_GENERATED') {
        window.removeEventListener('message', handleMessage);
        if (event.data.success) {
          resolve(new Float32Array(event.data.embedding));
        } else {
          reject(new Error(event.data.error));
        }
      }
    };
    
    window.addEventListener('message', handleMessage);
    sandboxFrame.contentWindow.postMessage({
      type: 'GENERATE_EMBEDDING',
      imageData: Array.from(imageData.data)
    }, '*');
  });
}

// Add this function to clear all frames and restore original images
function clearAllFrames() {
  // Find all face detection wrappers
  const wrappers = document.querySelectorAll('.face-detection-wrapper');
  
  wrappers.forEach(wrapper => {
    // Get the original image
    const img = wrapper.querySelector('img, image');
    if (img) {
      // Remove the wrapper and insert the original image back
      wrapper.parentNode.insertBefore(img, wrapper);
      wrapper.remove();
    }
  });
  
  // Clear the processed images set to allow reprocessing if needed
  state.processedImages.clear();
}

/**
 * Scales back detection boxes based on the image scale factor
 * @param {Array<Object>} detections - Array of face detections
 * @param {number} scaleFactor - Scale factor to apply
 */
function scaleDetections(detections, scaleFactor) {
  detections.forEach(detection => {
    detection.box.x /= scaleFactor;
    detection.box.y /= scaleFactor;
    detection.box.width /= scaleFactor;
    detection.box.height /= scaleFactor;
  });
}

/**
 * Creates a canvas for drawing face detections
 * @param {HTMLImageElement} img - The source image
 * @returns {HTMLCanvasElement} The detection canvas
 */
function createDetectionCanvas(img) {
  const canvas = document.createElement('canvas');
  canvas.className = 'face-detection-canvas';
  canvas.style.position = 'absolute';
  canvas.style.top = '0';
  canvas.style.left = '0';
  canvas.style.pointerEvents = 'none';
  
  canvas.width = img.width;
  canvas.height = img.height;
  canvas.style.width = img.width + 'px';
  canvas.style.height = img.height + 'px';
  
  return canvas;
}

/**
 * Adds a result indicator to the wrapper
 * @param {HTMLElement} wrapper - The wrapper element
 * @param {string} text - The text to display
 */
function addResultIndicator(wrapper, text) {
  const indicator = document.createElement('div');
  indicator.className = 'result-indicator';
  indicator.style.position = 'absolute';
  indicator.style.top = '5px';
  indicator.style.right = '5px';
  indicator.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
  indicator.style.color = 'white';
  indicator.style.padding = '2px 5px';
  indicator.style.borderRadius = '3px';
  indicator.style.fontSize = '12px';
  indicator.textContent = text;
  wrapper.appendChild(indicator);
}

/**
 * Draws face detections on the canvas
 * @param {HTMLCanvasElement} canvas - The canvas to draw on
 * @param {Array<Object>} detections - Array of face detections
 * @param {HTMLImageElement} img - The source image
 */
function drawDetections(canvas, detections, img) {
  const ctx = canvas.getContext('2d');
  const displayToNaturalRatioX = img.naturalWidth / img.width;
  const displayToNaturalRatioY = img.naturalHeight / img.height;
  
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
    
    // Draw corners
    const cornerSize = Math.min(scaledBox.width, scaledBox.height) * 0.2;
    ctx.lineWidth = 2;
    
    // Define corners
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
    
    // Draw corners
    corners.forEach(([x1, y1, x2, y2]) => {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });
    
    // Add face number if enabled
    if (flagShowFrameonImage.addLabel) {
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
    }
  });
}