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

const flagShowFrameonImage = {
  frameProsessedImage: true,    // Controls green frame around processed images
  frameFaceDetected: true,      // Controls red frame around face detected
  autoProcessImages: true       // Controls automatic processing of all images
}

/** @type {State} */
const state = {
  modelsLoaded: false,
  modelLoadAttempts: 0,
  MAX_LOAD_ATTEMPTS: 3,
  isProcessing: false,
  processedImages: new Set(),
  processingQueue: [], // Add queue for parallel processing
  maxParallelProcessing: 3 // Maximum number of parallel processes
};

// Add queue processor function
async function processQueue() {
  if (state.processingQueue.length === 0) return;
  
  const batch = state.processingQueue.splice(0, state.maxParallelProcessing);
  const promises = batch.map(element => {
    return detectFacesWithFaceApi(element).catch(error => {
      console.error('Face API processing error:', error);
      const src = element.tagName === 'IMG' ? element.src : element.getAttribute('xlink:href');
      state.processedImages.delete(src);
    });
  });
  
  await Promise.all(promises);
  if (state.processingQueue.length > 0) {
    processQueue(); // Process next batch
  }
}

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
  try {
    await loadFaceApiModels();
    
    const imgSrc = img.tagName === 'IMG' ? img.src : img.getAttribute('xlink:href');
    // console.log('Processing image:', imgSrc);

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
    
    // Apply green frame only if flagShowFrameonImage.frameProsessedImage is true
    if (flagShowFrameonImage.frameProsessedImage) {
      wrapper.style.border = '3px solid #00ff00';
      wrapper.style.boxSizing = 'border-box';
      wrapper.style.padding = '2px';
      
      // Add processed indicator
      const indicator = document.createElement('div');
      indicator.style.position = 'absolute';
      indicator.style.top = '5px';
      indicator.style.right = '5px';
      indicator.style.backgroundColor = 'rgba(0, 255, 0, 0.7)';
      indicator.style.color = 'white';
      indicator.style.padding = '2px 5px';
      indicator.style.borderRadius = '3px';
      indicator.style.fontSize = '12px';
      indicator.textContent = 'Processed';
      wrapper.appendChild(indicator);
    }
    
    wrapper.className = 'face-detection-wrapper';
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
    
    // Update console messages with image source
    if (detections.length === 0) {
      console.log('Face API: No faces detected in:', imgSrc);
      return;
    }
    
    console.log(`Face API: Detected ${detections.length} faces in:`, imgSrc);
    
    // Draw detections only if frameFaceDetected is true
    if (flagShowFrameonImage.frameFaceDetected) {
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
        
        // Draw corners and text only if frameFaceDetected is true
        if (flagShowFrameonImage.frameFaceDetected) {
          // Draw corner indicators
          const cornerSize = Math.min(scaledBox.width, scaledBox.height) * 0.2;
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
        }
      });
    }
    
    // Update indicator if faces are detected
    if (detections.length > 0 && wrapper.querySelector('div')) {
      wrapper.querySelector('div').textContent = `${detections.length} Face(s) Detected`;
    }

    wrapper.appendChild(canvas);
  } catch (error) {
    console.error('Face API detection error:', error);
    throw error; // Propagate error for queue handling
  }
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
  const width = element.width || element.clientWidth;
  const height = element.height || element.clientHeight;
  
  return (
    width > 50 &&
    height > 50 &&
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
    state.processingQueue.push(element);
    
    // Start processing if not already running
    if (state.processingQueue.length === 1) {
      processQueue();
    }
  }
}

/**
 * Starts observing elements on the page for face detection
 */
function observeElements() {
  const elements = [
    ...Array.from(document.getElementsByTagName('img')),
    ...Array.from(document.getElementsByTagName('image'))
  ];
  
  console.log(`Found ${elements.length} images to process`);
  
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