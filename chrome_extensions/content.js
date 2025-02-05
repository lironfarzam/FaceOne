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
    },
    myModel: {  // Add new model status
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
 * Creates sandbox iframe for TensorFlow operations and waits for it to be ready
 */
async function createSandboxFrame() {
    if (sandboxFrame && sandboxFrame.contentWindow) return;

    // Cleanup any existing frame
    if (sandboxFrame) {
        try {
            document.body.removeChild(sandboxFrame);
        } catch (e) {
            console.warn('Error removing existing sandbox frame:', e);
        }
        sandboxFrame = null;
    }

    return new Promise((resolve, reject) => {
        try {
            console.log('Creating new sandbox frame...');
            sandboxFrame = document.createElement('iframe');
            sandboxFrame.src = chrome.runtime.getURL('sandbox.html');
            sandboxFrame.style.display = 'none';
            
            let frameLoadTimeout;
            let tfInitTimeout;
            
            const cleanup = () => {
                clearTimeout(frameLoadTimeout);
                clearTimeout(tfInitTimeout);
                sandboxFrame.removeEventListener('load', handleLoad);
                sandboxFrame.removeEventListener('error', handleError);
                window.removeEventListener('message', handleTfInit);
            };
            
            const handleTfInit = (event) => {
                if (event.data && event.data.type === 'TF_INITIALIZED') {
                    cleanup();
                    if (event.data.success) {
                        console.log('TensorFlow initialized successfully:', event.data.info);
                        resolve();
                    } else {
                        console.error('TF initialization failed:', event.data.error, 'Status:', event.data.status);
                        reject(new Error('TF initialization failed: ' + event.data.error));
                    }
                }
            };
            
            const handleLoad = () => {
                console.log('Sandbox frame loaded, waiting for TF initialization...');
                window.addEventListener('message', handleTfInit);
                
                tfInitTimeout = setTimeout(() => {
                    cleanup();
                    reject(new Error('TF initialization timeout'));
                }, 30000); // 30 second timeout for TF initialization
            };
            
            const handleError = (error) => {
                cleanup();
                reject(new Error('Sandbox frame failed to load: ' + error.message));
            };
            
            sandboxFrame.addEventListener('load', handleLoad);
            sandboxFrame.addEventListener('error', handleError);
            
            frameLoadTimeout = setTimeout(() => {
                cleanup();
                reject(new Error('Sandbox frame load timeout'));
            }, 10000);
            
            document.body.appendChild(sandboxFrame);
            
        } catch (error) {
            reject(new Error('Failed to create sandbox frame: ' + error.message));
        }
    });
}

/**
 * Loads the FaceNet model in sandbox
 */
async function loadFaceNetModel() {
    const modelPath = chrome.runtime.getURL('models/FaceNet/Facenet512_tfjs_graph_model/model.json');
    return new Promise((resolve, reject) => {
        const handleMessage = (event) => {
            if (event.data.type === 'MODEL_LOADED' && event.data.modelName === 'faceNet') {
                window.removeEventListener('message', handleMessage);
                if (event.data.success) {
                    if (event.data.modelInfo && event.data.modelInfo.warmedUp) {
                        modelStatus.faceNet.loaded = true;
                        console.log('FaceNet model loaded and warmed up successfully');
                        resolve();
                    } else {
                        reject(new Error('FaceNet model loaded but not warmed up'));
                    }
                } else {
                    reject(new Error(event.data.error || 'FaceNet model loading failed'));
                }
            }
        };
        
        window.addEventListener('message', handleMessage);
        sandboxFrame.contentWindow.postMessage({
            type: 'LOAD_MODEL',
            modelName: 'faceNet',
            modelPath: modelPath,
            waitForWarmup: true
        }, '*');
        
        setTimeout(() => {
            window.removeEventListener('message', handleMessage);
            reject(new Error('FaceNet model load timeout'));
        }, 30000);
    });
}

// Add loading lock
let isLoadingModels = false;

/**
 * Loads all required models with retry mechanism
 */
async function loadFaceApiModels() {
    // Check if models are already loaded
    if (modelStatus.faceApi.loaded && modelStatus.faceNet.loaded) {
        return;
    }

    // Ensure TensorFlow is ready before proceeding
    await ensureTensorFlowReady();

    // Prevent concurrent loading attempts
    if (isLoadingModels) {
        console.log('Model loading already in progress, waiting...');
        let waitStart = Date.now();
        while (isLoadingModels && (Date.now() - waitStart) < 30000) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        if (modelStatus.faceApi.loaded && modelStatus.faceNet.loaded) {
            return;
        }
        if (isLoadingModels) {
            throw new Error('Model loading timeout while waiting');
        }
    }

    isLoadingModels = true;
    
    try {
        // Ensure extension context is available
        if (!chrome.runtime || !chrome.runtime.id) {
            await initializeExtensionContext();
        }
        
        state.modelLoadAttempts++;
        
        // Create and wait for sandbox frame first
        if (!sandboxFrame || !sandboxFrame.contentWindow) {
            console.log('Creating sandbox frame...');
            await createSandboxFrame();
            // Additional wait to ensure frame is fully ready
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
        
        // Verify sandbox frame is properly initialized
        if (!sandboxFrame || !sandboxFrame.contentWindow) {
            throw new Error('Sandbox frame not properly initialized');
        }
        
        // Load FaceAPI first if not already loaded
        if (!modelStatus.faceApi.loaded && !modelStatus.faceApi.loading) {
            console.log('Loading FaceAPI model...');
            modelStatus.faceApi.loading = true;
            modelStatus.faceApi.error = null;
            
            try {
                const modelPath = chrome.runtime.getURL('models');
                await retryModelLoad(async () => {
                    try {
                        await faceapi.nets.ssdMobilenetv1.loadFromUri(modelPath);
                        modelStatus.faceApi.loaded = true;
                        console.log('FaceAPI model loaded successfully');
                        return true;
                    } catch (error) {
                        console.error('FaceAPI load attempt failed:', error);
                        return false;
                    }
                }, 3, 1000);
            } catch (error) {
                modelStatus.faceApi.error = error;
                console.error('Failed to load FaceAPI model:', error);
                throw error;
            } finally {
                modelStatus.faceApi.loading = false;
            }
        }
        
        // Verify FaceAPI loaded successfully before proceeding
        if (!modelStatus.faceApi.loaded) {
            throw new Error('FaceAPI model failed to load');
        }
        
        // Load FaceNet model if not already loaded
        if (!modelStatus.faceNet.loaded && !modelStatus.faceNet.loading) {
            console.log('Loading FaceNet model...');
            modelStatus.faceNet.loading = true;
            modelStatus.faceNet.error = null;
            
            try {
                await new Promise((resolve, reject) => {
                    const handleMessage = (event) => {
                        if (event.data.type === 'MODEL_LOADED' && event.data.modelName === 'faceNet') {
                            window.removeEventListener('message', handleMessage);
                            if (event.data.success) {
                                modelStatus.faceNet.loaded = true;
                                console.log('FaceNet model loaded successfully');
                                resolve();
                            } else {
                                reject(new Error(event.data.error || 'FaceNet model loading failed'));
                            }
                        }
                    };
                    
                    window.addEventListener('message', handleMessage);
                    const modelPath = chrome.runtime.getURL('models/FaceNet/Facenet512_tfjs_graph_model/model.json');
                    sandboxFrame.contentWindow.postMessage({
                        type: 'LOAD_MODEL',
                        modelName: 'faceNet',
                        modelPath: modelPath
                    }, '*');
                    
                    setTimeout(() => {
                        window.removeEventListener('message', handleMessage);
                        reject(new Error('FaceNet model load timeout'));
                    }, 30000);
                });
            } catch (error) {
                modelStatus.faceNet.error = error;
                console.error('Failed to load FaceNet model:', error);
                throw error;
            } finally {
                modelStatus.faceNet.loading = false;
            }
        }
        
        // Final verification of both models
        if (!modelStatus.faceApi.loaded || !modelStatus.faceNet.loaded) {
            const errors = [];
            if (!modelStatus.faceApi.loaded) errors.push('FaceAPI');
            if (!modelStatus.faceNet.loaded) errors.push('FaceNet');
            throw new Error(`Models not loaded: ${errors.join(', ')}`);
        }
        
        // Set state after both models are confirmed loaded
        state.modelsLoaded = true;
        console.log('All models loaded successfully');
        
        // Automatically start processing existing images
        if (flagShowFrameonImage.autoProcessImages) {
            console.log('Starting automatic image processing...');
            await processExistingImages();
            // Start observing for new images
            observeElements();
        }
        
    } catch (error) {
        console.error('Error loading models:', error);
        
        // Cleanup and retry
        if (error.message.includes('Extension context') || 
            error.message.includes('Sandbox frame') ||
            error.message.includes('Model verification failed')) {
            
            if (sandboxFrame) {
                try {
                    document.body.removeChild(sandboxFrame);
                } catch (e) {
                    console.warn('Error removing sandbox frame:', e);
                }
                sandboxFrame = null;
            }
            
            clearModelStatus();
            
            if (state.modelLoadAttempts < state.MAX_LOAD_ATTEMPTS) {
                console.log(`Retrying model load (attempt ${state.modelLoadAttempts}/${state.MAX_LOAD_ATTEMPTS})`);
                await new Promise(resolve => setTimeout(resolve, 2000));
                isLoadingModels = false;
                return loadFaceApiModels();
            }
        }
        throw error;
    } finally {
        isLoadingModels = false;
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

// Add cross-origin image handling function
async function createProxyImage(originalImg) {
    return new Promise((resolve, reject) => {
        // Create a blob URL from the image
        const createBlobUrl = async () => {
            try {
                const response = await fetch(originalImg.src, { mode: 'cors' });
                const blob = await response.blob();
                return URL.createObjectURL(blob);
            } catch (error) {
                console.warn('Failed to create blob URL:', error);
                return null;
            }
        };

        const img = new Image();
        img.crossOrigin = 'anonymous';

        img.onload = () => {
            resolve(img);
        };

        img.onerror = async () => {
            // If direct loading fails, try using a blob URL
            try {
                const blobUrl = await createBlobUrl();
                if (blobUrl) {
                    img.src = blobUrl;
                } else {
                    reject(new Error('Failed to load image'));
                }
            } catch (error) {
                reject(error);
            }
        };

        // First try loading directly with crossOrigin
        img.src = originalImg.src;
    });
}

// Modify detectFacesWithFaceApi function
async function detectFacesWithFaceApi(img) {
    try {
        // Ensure models are loaded before processing
        await ensureModelsLoaded();
        
        const src = img.tagName === 'IMG' ? img.src : img.getAttribute('xlink:href');
        const wrapper = createWrapper(img);
        
        if (flagShowFrameonImage.addLabel) {
            addLoadingIndicator(wrapper);
        }

        // Create a proxy image to handle cross-origin
        let proxyImg;
        try {
            proxyImg = await createProxyImage(img);
        } catch (error) {
            console.error('Failed to create proxy image:', error);
            throw new Error('Unable to process cross-origin image');
        }

        const scaledImg = await createScaledImage(proxyImg);
        
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
        
        const faceEmbeddings = [];
        for (const detection of detections) {
            const faceCanvas = await extractFaceRegion(proxyImg, detection);
            try {
                const embedding = await generateEmbedding(faceCanvas);
                faceEmbeddings.push(embedding);
                imageTracker.markProcessed(src, true, embedding);
                
                // If there are multiple faces, compute similarities between them
                if (faceEmbeddings.length > 1) {
                    const lastIndex = faceEmbeddings.length - 1;
                    for (let i = 0; i < lastIndex; i++) {
                        try {
                            const similarity = await computeFaceSimilarity(
                                faceEmbeddings[i],
                                faceEmbeddings[lastIndex]
                            );
                            console.log(`Similarity between face ${i + 1} and ${lastIndex + 1}: ${similarity}`);
                        } catch (error) {
                            console.error('Error computing similarity:', error);
                        }
                    }
                }
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

// Modified click handler with proper selection handling
let clickTimeout;
document.addEventListener('click', (e) => {
    if (e.target.tagName === 'IMG') {
        e.preventDefault();
        
        // Clear any existing selection safely
        try {
            const selection = window.getSelection();
            if (selection && selection.rangeCount > 0) {
                selection.removeAllRanges();
            }
        } catch (error) {
            // Ignore selection errors
            console.warn('Selection clear error:', error);
        }
        
        clearTimeout(clickTimeout);
        clickTimeout = setTimeout(() => {
            const src = e.target.src;
            if (!src) return;
            
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
        try {
            await loadFaceApiModels();
            
            // Double check model status
            if (!modelStatus.faceApi.loaded) {
                throw new Error('FaceAPI model failed to load');
            }
            if (!modelStatus.faceNet.loaded) {
                throw new Error('FaceNet model failed to load');
            }
            
            console.log('Models loaded successfully');
            state.modelsLoaded = true;
        } catch (error) {
            console.error('Error loading models:', error);
            clearModelStatus(); // Reset status on error
            throw error;
        }
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

// Add scaleDetections function
function scaleDetections(detections, scaleFactor) {
    return detections.map(detection => {
        const scaledBox = {
            x: detection.box.x / scaleFactor,
            y: detection.box.y / scaleFactor,
            width: detection.box.width / scaleFactor,
            height: detection.box.height / scaleFactor
        };
        return { ...detection, box: scaledBox };
    });
}

// Initialize documentObserver
let documentObserver = null;

// Modify observeElements function to initialize documentObserver if needed
function observeElements() {
    if (!state.modelsLoaded) {
        console.warn('Cannot start observation: models not loaded');
        return;
    }

    console.log('Starting image observation...');
    const elements = [
        ...Array.from(document.getElementsByTagName('img')),
        ...Array.from(document.getElementsByTagName('image')),
        ...Array.from(document.querySelectorAll('image[xlink\\:href^="https://"]')),
        ...Array.from(document.querySelectorAll('image[preserveAspectRatio="xMidYMid slice"][xlink\\:href^="https://"]'))
    ].filter(element => isValidElement(element));
    
    // Initialize imageObserver if needed
    if (!imageObserver) {
        console.warn('Image observer not initialized');
        return;
    }
    
    elements.forEach(element => {
        if (element.complete || element.tagName === 'image') {
            handleVisibleElement(element);
        }
        imageObserver.observe(element);
    });
    
    // Initialize documentObserver if not already initialized
    if (!documentObserver) {
        documentObserver = new MutationObserver((mutations) => {
            mutations.forEach(mutation => {
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach(node => {
                        // Immediately process any new image nodes
                        if ((node.tagName === 'IMG' || node.tagName === 'image') && isValidElement(node)) {
                            handleVisibleElement(node);
                            if (imageObserver) {
                                imageObserver.observe(node);
                            }
                        }
                        
                        // Check for images within added nodes
                        if (node.querySelectorAll) {
                            ['IMG', 'image'].forEach(tagName => {
                                const elements = Array.from(node.querySelectorAll(tagName))
                                    .filter(element => isValidElement(element));
                                
                                elements.forEach(element => {
                                    handleVisibleElement(element);
                                    if (imageObserver) {
                                        imageObserver.observe(element);
                                    }
                                });
                            });
                        }
                    });
                }
            });
        });
        
        documentObserver.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['src', 'xlink:href']
        });
    }
    
    console.log('Image observation started');
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

// Add function to start document observer
function startDocumentObserver() {
    if (documentObserver) return;
    
    documentObserver = new MutationObserver((mutations) => {
        mutations.forEach(mutation => {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach(node => {
                    // Immediately process any new image nodes
                    if ((node.tagName === 'IMG' || node.tagName === 'image') && isValidElement(node)) {
                        handleVisibleElement(node);
                        if (imageObserver) {
                            imageObserver.observe(node);
                        }
                    }
                    
                    // Check for images within added nodes
                    if (node.querySelectorAll) {
                        ['IMG', 'image'].forEach(tagName => {
                            const elements = Array.from(node.querySelectorAll(tagName))
                                .filter(element => isValidElement(element));
                            
                            elements.forEach(element => {
                                handleVisibleElement(element);
                                if (imageObserver) {
                                    imageObserver.observe(element);
                                }
                            });
                        });
                    }
                });
            }
        });
    });
    
    documentObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['src', 'xlink:href']
    });
}

// Add initialization helper functions
async function initializeExtensionContext() {
    return new Promise((resolve, reject) => {
        try {
            if (chrome.runtime && chrome.runtime.id) {
                resolve();
            } else {
                const checkContext = setInterval(() => {
                    if (chrome.runtime && chrome.runtime.id) {
                        clearInterval(checkContext);
                        resolve();
                    }
                }, 100);

                setTimeout(() => {
                    clearInterval(checkContext);
                    reject(new Error('Extension context initialization timeout'));
                }, 10000);
            }
        } catch (error) {
            reject(new Error('Failed to initialize extension context: ' + error.message));
        }
    });
}

// Modify initialization sequence
window.addEventListener('load', async () => {
    console.log('Initializing FaceOne extension...');
    
    try {
        // First ensure extension context is available
        await initializeExtensionContext();
        console.log('Extension context initialized');
        
        // Create and initialize sandbox frame with TensorFlow
        if (!sandboxFrame || !sandboxFrame.contentWindow) {
            await createSandboxFrame();
            console.log('Sandbox frame created');
            
            // Additional wait to ensure frame is fully ready
            await new Promise(resolve => setTimeout(resolve, 2000));
            console.log('Waiting period completed');
        }
        
        // Load models
        console.log('Starting model loading sequence...');
        await loadFaceApiModels();
        
        // Start processing if auto-processing is enabled
        if (flagShowFrameonImage.autoProcessImages) {
            console.log('Auto-processing enabled, starting image processing...');
            await processExistingImages();
            observeElements();
        }
        
    } catch (error) {
        console.error('Initialization error:', error);
        console.error('Model status at error:', JSON.stringify(modelStatus, null, 2));
        
        // Cleanup and retry
        if (error.message.includes('Extension context') || 
            error.message.includes('Sandbox frame') ||
            error.message.includes('TF initialization')) {
            
            if (sandboxFrame) {
                try {
                    document.body.removeChild(sandboxFrame);
                } catch (e) {
                    console.warn('Error removing sandbox frame:', e);
                }
                sandboxFrame = null;
            }
            
            clearModelStatus();
            
            // Retry initialization after a delay
            setTimeout(() => {
                console.log('Retrying initialization...');
                window.location.reload();
            }, 2000);
        }
    }
});

// Add function to ensure TensorFlow is ready
async function ensureTensorFlowReady() {
    if (!sandboxFrame || !sandboxFrame.contentWindow) {
        throw new Error('Sandbox frame not available');
    }

    const status = await checkTensorFlowStatus();
    if (!status.isInitialized || !status.tfBackendInitialized) {
        throw new Error('TensorFlow not properly initialized');
    }
}

// Modify processExistingImages to ensure models are ready
async function processExistingImages() {
    try {
        // Ensure models are loaded before starting
        if (!state.modelsLoaded) {
            console.log('Models not loaded, loading now...');
            await loadFaceApiModels();
        }

        console.log('Scanning page for images...');
        const elements = [
            ...Array.from(document.getElementsByTagName('img')),
            ...Array.from(document.getElementsByTagName('image')),
            ...Array.from(document.querySelectorAll('image[xlink\\:href^="https://"]')),
            ...Array.from(document.querySelectorAll('image[preserveAspectRatio="xMidYMid slice"][xlink\\:href^="https://"]'))
        ].filter(element => isValidElement(element));
        
        console.log(`Found ${elements.length} valid images to process`);
        
        // Process images in batches with delay between batches
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
            
            // Add small delay between batches to prevent overwhelming
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        console.log('Finished processing existing images');
    } catch (error) {
        console.error('Error processing existing images:', error);
    }
}

// Modify generateEmbedding to verify model readiness
async function generateEmbedding(faceCanvas) {
    return new Promise((resolve, reject) => {
        // First check model status
        const checkModelStatus = (event) => {
            if (event.data.type === 'MODEL_STATUS') {
                window.removeEventListener('message', checkModelStatus);
                if (event.data.isReady) {
                    proceedWithEmbedding();
                } else {
                    reject(new Error('FaceNet model not ready: ' + event.data.reason));
                }
            }
        };

        const proceedWithEmbedding = () => {
            const handleEmbedding = (event) => {
                if (event.data.type === 'EMBEDDING_GENERATED') {
                    window.removeEventListener('message', handleEmbedding);
                    if (event.data.success) {
                        resolve(new Float32Array(event.data.embedding));
                    } else {
                        reject(new Error(event.data.error));
                    }
                }
            };

            window.addEventListener('message', handleEmbedding);
            const imageData = faceCanvas.getContext("2d").getImageData(0, 0, faceCanvas.width, faceCanvas.height);
            
            sandboxFrame.contentWindow.postMessage({
                type: 'GENERATE_EMBEDDING',
                imageData: Array.from(imageData.data)
            }, '*');

            setTimeout(() => {
                window.removeEventListener('message', handleEmbedding);
                reject(new Error('Embedding generation timeout'));
            }, 30000);
        };

        // First check model status
        window.addEventListener('message', checkModelStatus);
        sandboxFrame.contentWindow.postMessage({
            type: 'CHECK_MODEL_STATUS',
            modelName: 'faceNet'
        }, '*');

        setTimeout(() => {
            window.removeEventListener('message', checkModelStatus);
            reject(new Error('Model status check timeout'));
        }, 5000);
    });
}

// Add function to load your new model
async function loadMyModel() {
    const modelPath = chrome.runtime.getURL('models/myModel/tfjs_graph_model/model.json');
    await loadTFModel('myModel', modelPath);
}

// Add function to run inference with your model
async function runModelInference(modelName, inputData, inputShape) {
    if (!modelStatus[modelName].loaded) {
        throw new Error(`${modelName} model not loaded`);
    }

    return new Promise((resolve, reject) => {
        const handleMessage = (event) => {
            if (event.data.type === 'INFERENCE_COMPLETE' && event.data.modelName === modelName) {
                window.removeEventListener('message', handleMessage);
                if (event.data.success) {
                    resolve(event.data.result);
                } else {
                    reject(new Error(event.data.error));
                }
            }
        };
        
        window.addEventListener('message', handleMessage);
        sandboxFrame.contentWindow.postMessage({
            type: 'RUN_INFERENCE',
            modelName: modelName,
            inputData: inputData,
            inputShape: inputShape
        }, '*');
        
        setTimeout(() => {
            window.removeEventListener('message', handleMessage);
            reject(new Error('Inference timeout'));
        }, 30000);
    });
}

/**
 * Computes similarity between two face embeddings
 * @param {Float32Array} embedding1 - First face embedding
 * @param {Float32Array} embedding2 - Second face embedding
 * @returns {Promise<number>} Similarity score between 0 and 1
 */
async function computeFaceSimilarity(embedding1, embedding2) {
    if (!modelStatus.myModel.loaded) {
        throw new Error('Similarity model not loaded');
    }

    return new Promise((resolve, reject) => {
        const handleMessage = (event) => {
            if (event.data.type === 'SIMILARITY_COMPUTED') {
                window.removeEventListener('message', handleMessage);
                if (event.data.success) {
                    resolve(event.data.similarity);
                } else {
                    reject(new Error(event.data.error));
                }
            }
        };
        
        window.addEventListener('message', handleMessage);
        sandboxFrame.contentWindow.postMessage({
            type: 'COMPUTE_SIMILARITY',
            embedding1: Array.from(embedding1),
            embedding2: Array.from(embedding2)
        }, '*');
        
        setTimeout(() => {
            window.removeEventListener('message', handleMessage);
            reject(new Error('Similarity computation timeout'));
        }, 30000);
    });
}

// Add retry mechanism for model loading
async function retryModelLoad(loadFunction, maxAttempts = 3, delayMs = 1000) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            console.log(`Attempt ${attempt}/${maxAttempts} to load model...`);
            return await loadFunction();
        } catch (error) {
            console.warn(`Load attempt ${attempt} failed:`, error);
            lastError = error;
            
            if (attempt < maxAttempts) {
                console.log(`Waiting ${delayMs}ms before retry...`);
                await new Promise(resolve => setTimeout(resolve, delayMs * attempt));
            }
        }
    }
    
    throw lastError;
}

// Modify loadTFModel to use retry mechanism
async function loadTFModel(modelName, modelPath) {
    if (modelStatus[modelName].loaded) return;
    
    if (modelStatus[modelName].loading) {
        const startTime = Date.now();
        while (modelStatus[modelName].loading && (Date.now() - startTime) < 30000) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        if (modelStatus[modelName].loaded) return;
        if (modelStatus[modelName].loading) {
            modelStatus[modelName].loading = false;
            throw new Error(`${modelName} model load timeout`);
        }
    }
    
    try {
        modelStatus[modelName].loading = true;
        modelStatus[modelName].error = null;
        
        await createSandboxFrame();
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        await retryModelLoad(async () => {
            return new Promise((resolve, reject) => {
                const handleMessage = (event) => {
                    if (event.data.type === 'MODEL_LOADED') {
                        window.removeEventListener('message', handleMessage);
                        if (event.data.success) {
                            modelStatus[modelName].loaded = true;
                            resolve();
                        } else {
                            reject(new Error(event.data.error || `${modelName} model loading failed`));
                        }
                    }
                };
                
                window.addEventListener('message', handleMessage);
                sandboxFrame.contentWindow.postMessage({ 
                    type: 'LOAD_MODEL',
                    modelName: modelName,
                    modelPath: modelPath
                }, '*');
                
                setTimeout(() => {
                    window.removeEventListener('message', handleMessage);
                    reject(new Error('Model load response timeout'));
                }, 30000);
            });
        });
        
    } catch (error) {
        modelStatus[modelName].error = error;
        throw error;
    } finally {
        modelStatus[modelName].loading = false;
    }
}

// Add missing helper functions
function clearModelStatus() {
    // Only clear error states and loading flags, preserve loaded states
    Object.keys(modelStatus).forEach(key => {
        if (modelStatus[key].error) {
            modelStatus[key].error = null;
        }
        if (modelStatus[key].loading) {
            modelStatus[key].loading = false;
        }
    });
}

// Modify createScaledImage function
async function createScaledImage(img) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Calculate scale factor to keep image within reasonable bounds
    const MAX_SIZE = 640;
    let width = img.width || img.naturalWidth;
    let height = img.height || img.naturalHeight;
    let scaleFactor = 1;
    
    if (width > MAX_SIZE || height > MAX_SIZE) {
        scaleFactor = MAX_SIZE / Math.max(width, height);
        width = Math.floor(width * scaleFactor);
        height = Math.floor(height * scaleFactor);
    }
    
    canvas.width = width;
    canvas.height = height;
    
    // Draw scaled image
    try {
        ctx.drawImage(img, 0, 0, width, height);
        canvas.scaleFactor = scaleFactor;
    } catch (error) {
        console.error('Error drawing image to canvas:', error);
        throw new Error('Failed to create scaled image');
    }
    
    return canvas;
}

/**
 * Creates a canvas for drawing face detections
 * @param {HTMLImageElement} img - The source image
 * @returns {HTMLCanvasElement} A canvas for drawing detections
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
    return canvas;
}

/**
 * Draws face detection boxes on a canvas
 * @param {HTMLCanvasElement} canvas - The canvas to draw on
 * @param {Array} detections - Array of face detections
 * @param {HTMLImageElement} img - The source image
 */
function drawDetections(canvas, detections, img) {
    const ctx = canvas.getContext('2d');
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#FF0000';
    ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
    
    detections.forEach((detection, index) => {
        const { x, y, width, height } = detection.box;
        ctx.strokeRect(x, y, width, height);
        
        if (flagShowFrameonImage.addLabel) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(x, y - 20, 70, 20);
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '12px Arial';
            ctx.fillText(`Face ${index + 1}`, x + 5, y - 5);
        }
    });
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
 * Extracts a face region from an image
 * @param {HTMLImageElement} img - The source image
 * @param {Object} detection - The face detection data
 * @returns {HTMLCanvasElement} A canvas containing the face region
 */
async function extractFaceRegion(img, detection) {
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 160;
    const ctx = canvas.getContext('2d');
    
    const { x, y, width, height } = detection.box;
    ctx.drawImage(img, x, y, width, height, 0, 0, 160, 160);
    
    return canvas;
}

// Add helper function to check model status
function checkModelStatus() {
    const status = {
        faceApi: modelStatus.faceApi.loaded,
        faceNet: modelStatus.faceNet.loaded,
        errors: []
    };
    
    if (!modelStatus.faceApi.loaded && modelStatus.faceApi.error) {
        status.errors.push(`FaceAPI: ${modelStatus.faceApi.error.message}`);
    }
    if (!modelStatus.faceNet.loaded && modelStatus.faceNet.error) {
        status.errors.push(`FaceNet: ${modelStatus.faceNet.error.message}`);
    }
    
    return status;
}

// Add helper function to check model loading status
function getModelLoadingStatus() {
    return {
        faceApi: {
            ...modelStatus.faceApi,
            loading: modelStatus.faceApi.loading
        },
        faceNet: {
            ...modelStatus.faceNet,
            loading: modelStatus.faceNet.loading
        },
        isLoadingModels,
        attempts: state.modelLoadAttempts
    };
}

// Add function to check TF status
async function checkTensorFlowStatus() {
    if (!sandboxFrame || !sandboxFrame.contentWindow) {
        return { initialized: false, error: 'No sandbox frame' };
    }

    return new Promise((resolve) => {
        const handleResponse = (event) => {
            if (event.data && event.data.type === 'TF_STATUS') {
                window.removeEventListener('message', handleResponse);
                resolve(event.data.status);
            }
        };

        window.addEventListener('message', handleResponse);
        sandboxFrame.contentWindow.postMessage({ type: 'GET_TF_STATUS' }, '*');

        // Add timeout
        setTimeout(() => {
            window.removeEventListener('message', handleResponse);
            resolve({ initialized: false, error: 'Status check timeout' });
        }, 5000);
    });
}