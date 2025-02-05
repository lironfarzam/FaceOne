/**
 * @fileoverview Content script for FaceOne Chrome extension.
 * Provides real-time face detection and embedding generation for web images.
 * Uses FaceAPI.js for detection and FaceNet for embedding generation.
 * @author Liron Farzam
 * @version 1.0.0
 */

//=============================================================================
// Configuration and State Management
//=============================================================================

/**
 * @typedef {Object} State
 * @property {boolean} modelsLoaded - Indicates if ML models are loaded
 * @property {boolean} faceNetLoaded - Indicates if FaceNet model is loaded
 * @property {number} modelLoadAttempts - Number of attempts to load models
 * @property {number} MAX_LOAD_ATTEMPTS - Maximum number of load attempts
 * @property {boolean} isProcessing - Flag to prevent concurrent processing
 * @property {Set<string>} processedImages - Set of processed image URLs
 */

/** @type {State} */
const state = {
    modelsLoaded: false,
    faceNetLoaded: false,
    modelLoadAttempts: 0,
    MAX_LOAD_ATTEMPTS: 3,
    isProcessing: false,
    processingQueue: [],
    maxParallelProcessing: 3
};

/**
 * Extension settings with default values
 * @type {Object}
 */
let flagShowFrameonImage = {
    frameProsessedImage: true,
    frameFaceDetected: true,
    addLabel: true,
    autoProcessImages: true,
    minimumImageSize: 100
};

/**
 * Model loading status tracking
 * @type {Object}
 */
const modelStatus = {
    faceApi: {
        loaded: false,
        loading: false,
        error: null
    },
    faceNet: {
        loaded: false,
        loading: false,
        error: null
    }
};

//=============================================================================
// Model Management
//=============================================================================

let sandboxFrame = null;
let faceNetModel = null;

/**
 * Creates sandbox iframe for TensorFlow operations
 */
function createSandboxFrame() {
    if (sandboxFrame) return;
    sandboxFrame = document.createElement('iframe');
    sandboxFrame.src = chrome.runtime.getURL('sandbox.html');
    sandboxFrame.style.display = 'none';
    document.body.appendChild(sandboxFrame);
}

/**
 * Loads the FaceNet model in sandbox
 */
async function loadFaceNetModel() {
    // Add a loading lock with timeout
    const LOAD_TIMEOUT = 30000; // 30 seconds timeout
    
    if (modelStatus.faceNet.loaded) return;
    
    if (modelStatus.faceNet.loading) {
        // Wait for existing load to complete
        const startTime = Date.now();
        while (modelStatus.faceNet.loading && (Date.now() - startTime) < LOAD_TIMEOUT) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        if (modelStatus.faceNet.loaded) return;
        if (modelStatus.faceNet.loading) {
            modelStatus.faceNet.loading = false;
            throw new Error('FaceNet model load timeout');
        }
    }
    
    try {
        modelStatus.faceNet.loading = true;
        modelStatus.faceNet.error = null;
        
        createSandboxFrame();
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const modelPath = chrome.runtime.getURL('models/FaceNet/Facenet512_tfjs_graph_model/model.json');
        
        return new Promise((resolve, reject) => {
            const handleMessage = (event) => {
                if (event.data.type === 'MODEL_LOADED') {
                    window.removeEventListener('message', handleMessage);
                    if (event.data.success) {
                        modelStatus.faceNet.loaded = true;
                        state.faceNetLoaded = true;
                        resolve();
                    } else {
                        const error = new Error(event.data.error || 'FaceNet model loading failed');
                        modelStatus.faceNet.error = error;
                        reject(error);
                    }
                }
            };
            
            window.addEventListener('message', handleMessage);
            sandboxFrame.contentWindow.postMessage({ 
                type: 'LOAD_MODEL',
                modelPath: modelPath
            }, '*');
            
            // Add timeout for message response
            setTimeout(() => {
                window.removeEventListener('message', handleMessage);
                reject(new Error('Model load response timeout'));
            }, LOAD_TIMEOUT);
        });
    } catch (error) {
        modelStatus.faceNet.error = error;
        console.error('Error loading FaceNet model:', error);
        throw error;
    } finally {
        modelStatus.faceNet.loading = false;
    }
}

/**
 * Loads all required models with retry mechanism
 */
async function loadFaceApiModels() {
    if (modelStatus.faceApi.loaded && modelStatus.faceNet.loaded) return;
    
    try {
        await initializeExtensionContext();
        
        state.modelLoadAttempts++;
        const modelPath = chrome.runtime.getURL('models');
        
        if (!modelStatus.faceApi.loaded) {
            modelStatus.faceApi.loading = true;
            modelStatus.faceApi.error = null;
            try {
                await retryModelLoad(async () => {
                    await faceapi.nets.ssdMobilenetv1.loadFromUri(modelPath);
                    modelStatus.faceApi.loaded = true;
                    state.modelsLoaded = true;
                });
            } catch (error) {
                modelStatus.faceApi.error = error;
                throw error;
            } finally {
                modelStatus.faceApi.loading = false;
            }
        }
        
        if (!modelStatus.faceNet.loaded) {
            await retryModelLoad(() => loadFaceNetModel());
        }
        
    } catch (error) {
        console.error('Error loading models:', error);
        
        if (error.message.includes('Extension context invalidated')) {
            clearModelStatus();
            
            if (state.modelLoadAttempts < state.MAX_LOAD_ATTEMPTS) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                return loadFaceApiModels();
            }
        }
        throw error;
    }
}

//=============================================================================
// Image Processing
//=============================================================================

/**
 * Configuration options for Face API detection
 */
const FACE_API_DETECTION_OPTIONS = {
    scoreThreshold: 0.3,
    inputSize: 320,
    scaleFactor: 0.8,
    maxNumBoxes: 100,
    minConfidence: 0.3,
    iouThreshold: 0.5,
    useTinyModel: false,
    minFaceSize: 20
};

// Replace all image tracking with a single system
const imageTracker = {
    images: new Map(), // Map<string, ImageInfo>
    maxSize: 1000,
    
    add(src, info = {}) {
        if (this.images.size >= this.maxSize) {
            // Remove oldest entry
            const oldestKey = Array.from(this.images.keys())[0];
            this.images.delete(oldestKey);
        }
        
        this.images.set(src, {
            isProcessed: false,
            hasBeenTested: false,
            canDelete: false,
            embedding: null,
            timestamp: Date.now(),
            ...info
        });
    },
    
    markProcessed(src, success = true, embedding = null) {
        const info = this.images.get(src) || {};
        this.images.set(src, {
            ...info,
            isProcessed: success,
            hasBeenTested: true,
            embedding: embedding,
            timestamp: Date.now()
        });
    },
    
    shouldProcess(src) {
        const info = this.images.get(src);
        return !info || (!info.hasBeenTested && !info.isProcessed);
    },
    
    getEmbedding(src) {
        const info = this.images.get(src);
        return info ? info.embedding : null;
    },
    
    has(src) {
        return this.images.has(src);
    },
    
    clear() {
        this.images.clear();
    }
};

/**
 * Processes an image for face detection
 */
async function detectFacesWithFaceApi(img) {
    try {
        // Ensure models are loaded before processing
        await ensureModelsLoaded();
        
        const src = img.tagName === 'IMG' ? img.src : img.getAttribute('xlink:href');
        const wrapper = createWrapper(img);
        
        if (flagShowFrameonImage.addLabel) {
            addLoadingIndicator(wrapper);
        }
        
        const scaledImg = await createScaledImage(img);
        const detections = await faceapi.detectAllFaces(
            scaledImg,
            new faceapi.SsdMobilenetv1Options({
                ...FACE_API_DETECTION_OPTIONS,
                scoreThreshold: 0.4
            })
        );
        
        if (scaledImg.scaleFactor !== 1) {
            scaleDetections(detections, scaledImg.scaleFactor);
        }
        
        for (const detection of detections) {
            const faceCanvas = await extractFaceRegion(img, detection);
            try {
                const embedding = await generateEmbedding(faceCanvas);
                imageTracker.markProcessed(src, true, embedding);
            } catch (error) {
                console.error('Embedding generation error:', error);
                imageTracker.markProcessed(src, false);
            }
        }
        
        updateVisualization(wrapper, img, detections);
        
    } catch (error) {
        console.error('Face detection error:', error);
        throw error;
    }
}

//=============================================================================
// UI Management
//=============================================================================

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
          if (imageTracker.has(src)) {
            const embedding = imageTracker.getEmbedding(src);
            // Handle cached embedding (e.g., display visualization)
            return;
          }
          
          await detectFacesWithFaceApi(element);
        } catch (error) {
          console.error('Processing error:', error);
          const src = element.tagName === 'IMG' ? element.src : element.getAttribute('xlink:href');
          imageTracker.markProcessed(src, false);
        }
      });
      
      await Promise.all(promises);
    }
    this.processing = false;
  }
};

// Helper functions for improved visualization
function createWrapper(img) {
  // Store original styles if not already stored
  if (!img.getAttribute('data-original-style')) {
    img.setAttribute('data-original-style', img.style.cssText);
  }
  
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
// document.addEventListener('mousedown', preventTextSelection);
// document.addEventListener('selectstart', preventTextSelection);

// Modified click handler
let clickTimeout;
document.addEventListener('click', (e) => {
    if (e.target.tagName === 'IMG') {
        e.preventDefault();
        window.getSelection().removeAllRanges();
        
        clearTimeout(clickTimeout);
        clickTimeout = setTimeout(() => {
            const src = e.target.src;
            imageTracker.add(src, { hasBeenTested: false, isProcessed: false });
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
        !imageTracker.has(src) &&
        !element.closest('.face-detection-wrapper')
    );
}

// Add a function to ensure models are loaded
async function ensureModelsLoaded() {
    if (!modelStatus.faceApi.loaded || !modelStatus.faceNet.loaded) {
        console.log('Models not loaded, loading now...');
        await loadFaceApiModels();
        
        // Additional verification
        if (!modelStatus.faceApi.loaded || !modelStatus.faceNet.loaded) {
            throw new Error('Failed to load models after attempt');
        }
        console.log('Models loaded successfully');
    }
}

// Modify handleVisibleElement to await model loading
async function handleVisibleElement(element) {
    if (!element || !flagShowFrameonImage.autoProcessImages) return;
    
    try {
        // Ensure models are loaded before processing
        await ensureModelsLoaded();
        
        if (!isValidElement(element)) return;
        
        const src = element.tagName === 'IMG' ? element.src : element.getAttribute('xlink:href');
        if (!src) return;
        
        const info = imageTracker.images.get(src);
        const needsProcessing = !info || !info.isProcessed;
        
        if (needsProcessing) {
            console.log('Processing image:', src);
            if (!info) {
                imageTracker.add(src);
            }
            await detectFacesWithFaceApi(element);
        } else if (info && !info.isProcessed) {
            // Reapply visualization if needed
            const wrapper = createWrapper(element);
            if (info.detections) {
                updateVisualization(wrapper, element, info.detections);
            }
        }
    } catch (error) {
        console.error('Error in handleVisibleElement:', error);
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
        imageTracker.markProcessed(target.src || target.getAttribute('xlink:href'), false);
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
window.addEventListener('load', async () => {
    console.log('Initializing FaceOne extension...');
    
    try {
        await initializeExtensionContext();
        console.log('Extension context initialized');
        
        await loadFaceApiModels();
        console.log('Models loaded, starting observation');
        
        // Start observers first
        startObservers();
        
        // Process existing images if auto-processing is enabled
        if (flagShowFrameonImage.autoProcessImages) {
            console.log('Auto-processing enabled, scanning existing images');
            processExistingImages();
        }
    } catch (error) {
        console.error('Initialization error:', error);
    }
});

// Add new function to process existing images
async function processExistingImages() {
    const elements = [
        ...Array.from(document.getElementsByTagName('img')),
        ...Array.from(document.getElementsByTagName('image')),
        ...Array.from(document.querySelectorAll('image[xlink\\:href^="https://"]')),
        ...Array.from(document.querySelectorAll('image[preserveAspectRatio="xMidYMid slice"][xlink\\:href^="https://"]'))
    ];
    
    console.log(`Found ${elements.length} images to process`);
    
    // Process images in batches to avoid overwhelming the system
    const batchSize = 3;
    for (let i = 0; i < elements.length; i += batchSize) {
        const batch = elements.slice(i, i + batchSize);
        await Promise.all(batch.map(async element => {
            if (element.complete || element.tagName === 'image') {
                try {
                    await handleVisibleElement(element);
                } catch (error) {
                    console.error('Error processing image:', error);
                }
            }
        }));
    }
}

// Modify handleVisibleElement to be async and more robust
async function handleVisibleElement(element) {
    if (!element || !flagShowFrameonImage.autoProcessImages) return;
    
    try {
        if (!isValidElement(element)) return;
        
        const src = element.tagName === 'IMG' ? element.src : element.getAttribute('xlink:href');
        if (!src) return;
        
        const info = imageTracker.images.get(src);
        const needsProcessing = !info || !info.isProcessed;
        
        if (needsProcessing) {
            console.log('Processing image:', src);
            if (!info) {
                imageTracker.add(src);
            }
            await detectFacesWithFaceApi(element);
        } else if (info && !info.isProcessed) {
            // Reapply visualization if needed
            const wrapper = createWrapper(element);
            if (info.detections) {
                updateVisualization(wrapper, element, info.detections);
            }
        }
    } catch (error) {
        console.error('Error in handleVisibleElement:', error);
    }
}

// Modify startObservers to be more efficient
function startObservers() {
    stopObservers();
    
    // Configure intersection observer
    imageObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['src', 'xlink:href']
    });
    
    // Configure mutation observer
    documentObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['src', 'xlink:href']
    });
    
    console.log('Observers started');
}

function stopObservers() {
    imageObserver.disconnect();
    documentObserver.disconnect();
}

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

// Modify clearAllFrames to be more selective
function clearAllFrames() {
    console.log('Clearing frames and visualizations');
    
    const wrappers = document.querySelectorAll('.face-detection-wrapper');
    wrappers.forEach(wrapper => {
        const img = wrapper.querySelector('img, image');
        if (img) {
            img.style.cssText = img.getAttribute('data-original-style') || '';
            wrapper.parentNode.insertBefore(img, wrapper);
            wrapper.remove();
        }
    });
    
    // Only clear visual elements, not the processing status
    document.querySelectorAll('.face-detection-canvas, .processing-indicator, .result-indicator')
        .forEach(el => el.remove());
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

// Update clearProcessedImagesCache function
function clearProcessedImagesCache() {
    imageTracker.clear();
    clearModelStatus();
    if (sandboxFrame && sandboxFrame.contentWindow) {
        sandboxFrame.contentWindow.postMessage({ type: 'CLEAR_CACHE' }, '*');
    }
}

// Modify clearModelStatus to be more thorough
function clearModelStatus() {
    // Clear model status
    modelStatus.faceApi = {
        loaded: false,
        loading: false,
        error: null
    };
    modelStatus.faceNet = {
        loaded: false,
        loading: false,
        error: null
    };
    
    // Clear state
    state.modelsLoaded = false;
    state.faceNetLoaded = false;
    state.modelLoadAttempts = 0;
    
    // Clear sandbox if it exists
    if (sandboxFrame && sandboxFrame.contentWindow) {
        try {
            sandboxFrame.contentWindow.postMessage({ type: 'CLEANUP' }, '*');
        } catch (e) {
            console.warn('Error during sandbox cleanup:', e);
        }
    }
}

// Add retry mechanism for model loading
async function retryModelLoad(loadFunction, maxAttempts = 3) {
    let lastError;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await loadFunction();
        } catch (error) {
            console.warn(`Load attempt ${attempt} failed:`, error);
            lastError = error;
            if (attempt < maxAttempts) {
                await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
            }
        }
    }
    throw lastError;
}

// Update message handler to properly handle auto-processing changes
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Received message:', message);
    
    if (message.type === 'UPDATE_SETTINGS') {
        console.log('Updating settings:', message.settings);
        const oldSettings = {...flagShowFrameonImage};
        flagShowFrameonImage = message.settings;
        
        // Handle visual changes
        const visualSettingsChanged = (
            flagShowFrameonImage.frameProsessedImage !== oldSettings.frameProsessedImage ||
            flagShowFrameonImage.frameFaceDetected !== oldSettings.frameFaceDetected ||
            flagShowFrameonImage.addLabel !== oldSettings.addLabel
        );
        
        if (visualSettingsChanged) {
            console.log('Visual settings changed, reprocessing images');
            clearAllFrames();
            
            // Mark images for reprocessing but keep their embeddings
            imageTracker.images.forEach((info, src) => {
                info.isProcessed = false;  // This will trigger reprocessing
                // Keep hasBeenTested and embedding data
            });
            
            // Reprocess all visible images
            if (flagShowFrameonImage.autoProcessImages) {
                processExistingImages();
            }
        }
        
        // Handle auto-processing changes
        if (flagShowFrameonImage.autoProcessImages !== oldSettings.autoProcessImages) {
            if (flagShowFrameonImage.autoProcessImages) {
                console.log('Enabling auto-processing');
                startObservers();
                processExistingImages();
            } else {
                console.log('Disabling auto-processing');
                stopObservers();
                clearAllFrames();
            }
        }
        
        // Handle minimum size changes
        if (flagShowFrameonImage.minimumImageSize !== oldSettings.minimumImageSize) {
            console.log('Minimum size changed, reprocessing images');
            clearAllFrames();
            
            // Reset processing status but keep embeddings
            imageTracker.images.forEach((info, src) => {
                info.isProcessed = false;
            });
            
            if (flagShowFrameonImage.autoProcessImages) {
                processExistingImages();
            }
        }
        
        sendResponse({ success: true, settings: flagShowFrameonImage });
    } else if (message.type === 'REPROCESS_ALL') {
        console.log('Reprocessing all images');
        clearAllFrames();
        
        // Reset all processing status
        imageTracker.images.forEach((info, src) => {
            info.hasBeenTested = false;
            info.isProcessed = false;
            // Keep embeddings for reuse
        });
        
        // Restart observation and process all images
        startObservers();
        processExistingImages();
        
        sendResponse({ success: true });
    }
    
    return true;
});