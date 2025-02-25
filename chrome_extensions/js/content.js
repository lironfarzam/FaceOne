/**
 * @fileoverview Content script for FaceOne Chrome extension.
 * Provides real-time face detection and embedding generation for web images.
 * Uses FaceAPI.js for detection and FaceNet for embedding generation.
 * @author Liron Farzam
 * @version 1.0.0
 */

//=============================================================================
// 1. Core Configuration and Types
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

//=============================================================================
// 2. Model Management
//=============================================================================
const modelStatus = {
    faceApi: { loaded: false, loading: false, error: null },
    faceNet: { loaded: false, loading: false, error: null },
    myModel: { loaded: false, loading: false, error: null }
};

//=============================================================================
// 3. Settings Management
//=============================================================================
let flagShowFrameonImage = {
    frameProsessedImage: true,
    frameFaceDetected: true,    
    addLabel: true,
    autoProcessImages: true,
    minimumImageSize: 100,
    confidenceThreshold: 70,
    processingMode: 'face_detection'
};

//=============================================================================
// 4. Constants and Configuration
//=============================================================================
const MODEL_SELECTION_THRESHOLDS = {
    get MINIMUM_SIZE() {
        return Math.max(16, flagShowFrameonImage.minimumImageSize);
    },
    get SMALL_IMAGE() {
        return Math.max(96, this.MINIMUM_SIZE * 2);
    },
    get LARGE_IMAGE() {
        return Math.max(320, this.SMALL_IMAGE * 2);
    }
};

//=============================================================================
// Positive Embeddings Management
//=============================================================================

/** @type {Float32Array[]} */
let positiveEmbeddings = [];
let isPositiveEmbeddingsLoaded = false;

/**
 * Loads positive embeddings from JSON file
 * @returns {Promise<void>}
 */
async function loadPositiveEmbeddings() {
    if (isPositiveEmbeddingsLoaded) return;

    try {
        const embeddingsPath = chrome.runtime.getURL('models/embeddings/positive_embeddings.json');
        const response = await fetch(embeddingsPath);
        if (!response.ok) {
            throw new Error(`Failed to load positive embeddings: ${response.statusText}`);
        }

        const data = await response.json();
        if (!Array.isArray(data)) {
            throw new Error('Invalid positive embeddings format: expected array');
        }

        // Take only the first 10 embeddings
        const limitedData = data.slice(0, 10);
        
        // Convert embeddings to Float32Array for efficient comparison
        positiveEmbeddings = limitedData.map(embedding => {
            if (!Array.isArray(embedding) || embedding.length !== 512) {
                throw new Error('Invalid embedding format: expected 512-dimensional array');
            }
            return new Float32Array(embedding);
        });

        console.log(`Loaded ${positiveEmbeddings.length} positive embeddings (limited to first 10)`);
        isPositiveEmbeddingsLoaded = true;
    } catch (error) {
        console.error('Error loading positive embeddings:', error);
        throw error;
    }
}

/**
 * Compares a face embedding with all positive embeddings
 * @param {Float32Array} faceEmbedding - The embedding to compare
 * @returns {Promise<{maxSimilarity: number, matchIndex: number}>}
 */
async function compareWithPositiveEmbeddings(faceEmbedding) {
    if (!isPositiveEmbeddingsLoaded) {
        throw new Error('Positive embeddings not loaded');
    }

    let maxSimilarity = -1;
    let matchIndex = -1;

    for (let i = 0; i < positiveEmbeddings.length; i++) {
        try {
            const similarity = await computeFaceSimilarity(faceEmbedding, positiveEmbeddings[i]);
            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                matchIndex = i;
            }
        } catch (error) {
            console.warn(`Error comparing with positive embedding ${i}:`, error);
        }
    }

    return { maxSimilarity, matchIndex };
}

//=============================================================================
// Model Management
//=============================================================================

let sandboxFrame = null;
let faceNetModel = null;

/**
 * Creates sandbox iframe for TensorFlow operations and waits for it to be ready
 */
async function createSandboxFrame(timeout) {
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
                }, timeout);
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
    if (modelStatus.faceNet.loaded) {
        return;
    }

    const modelPath = chrome.runtime.getURL('models/FaceNet/Facenet512_tfjs_graph_model/model.json');
    return new Promise((resolve, reject) => {
        const handleMessage = (event) => {
            if (event.data.type === 'MODEL_LOADED' && event.data.modelName === 'faceNet') {
                window.removeEventListener('message', handleMessage);
                if (event.data.success) {
                    if (event.data.modelInfo && event.data.modelInfo.warmedUp) {
                        modelStatus.faceNet.loaded = true;
                        state.faceNetLoaded = true;
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
 * Configuration options for Face API detection
 */
const FACE_API_DETECTION_OPTIONS = {
    ssdMobilenetv1: {
        minConfidence: 0.2, // Reduced from 0.3 for better detection
        inputSize: 640,
        scoreThreshold: 0.2,
        maxNumBoxes: 100,
        scaleFactor: 0.8,
        iouThreshold: 0.5
    },
    tinyFaceDetector: {
        inputSize: 416,
        scoreThreshold: 0.01,
        minFaceSize: 20,
        scaleFactor: 0.709,
        maxNumBoxes: 100,
        iouThreshold: 0.3
    }
};

/**
 * Loads all required models with retry mechanism
 */
async function loadFaceApiModels() {
    // Check if models are already loaded
    if (modelStatus.faceApi.loaded && modelStatus.faceNet.loaded && modelStatus.myModel.loaded) {
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
        if (modelStatus.faceApi.loaded && modelStatus.faceNet.loaded && modelStatus.myModel.loaded) {
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
        
        // Create and wait for sandbox frame with dynamic timeout
        if (!sandboxFrame || !sandboxFrame.contentWindow) {
            console.log('Creating sandbox frame...');
            await createSandboxFrame(state.modelLoadAttempts);
            // Reduced wait time but still ensure frame is ready
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        // Verify sandbox frame is properly initialized
        if (!sandboxFrame || !sandboxFrame.contentWindow) {
            throw new Error('Sandbox frame not properly initialized');
        }
        
        // Load both models
        if (!modelStatus.faceApi.loaded && !modelStatus.faceApi.loading) {
            console.log('Loading FaceAPI models...');
            modelStatus.faceApi.loading = true;
            modelStatus.faceApi.error = null;
            
            try {
                const modelPath = chrome.runtime.getURL('models/FaceAPI');
                await retryModelLoad(async () => {
                    try {
                        // Load both models in parallel with correct subdirectory paths
                        await Promise.all([
                            faceapi.nets.ssdMobilenetv1.loadFromUri(`${modelPath}/ssd_mobilenetv1`),
                            faceapi.nets.tinyFaceDetector.loadFromUri(`${modelPath}/tiny_face_detector`)
                        ]);
                        modelStatus.faceApi.loaded = true;
                        console.log('FaceAPI models loaded successfully');
                        return true;
                    } catch (error) {
                        console.error('FaceAPI load attempt failed:', error);
                        return false;
                    }
                }, 3, 1000);
            } catch (error) {
                modelStatus.faceApi.error = error;
                console.error('Failed to load FaceAPI models:', error);
                throw error;
            } finally {
                modelStatus.faceApi.loading = false;
            }
        }
        
        // Load FaceNet model if not already loaded
        if (!modelStatus.faceNet.loaded && !modelStatus.faceNet.loading) {
            console.log('Loading FaceNet model...');
            await loadFaceNetModel();
        }

        // Load similarity model if not already loaded
        if (!modelStatus.myModel.loaded && !modelStatus.myModel.loading) {
            console.log('Loading similarity model...');
            modelStatus.myModel.loading = true;
            modelStatus.myModel.error = null;
            
            try {
                const modelPath = chrome.runtime.getURL('models/myModel/tfjs_graph_model/model.json');
                await new Promise((resolve, reject) => {
                    const handleMessage = (event) => {
                        if (event.data.type === 'MODEL_LOADED' && event.data.modelName === 'myModel') {
                            window.removeEventListener('message', handleMessage);
                            if (event.data.success) {
                                console.log('Similarity model loaded successfully');
                                modelStatus.myModel.loaded = true;
                                resolve();
                            } else {
                                reject(new Error(event.data.error || 'Similarity model loading failed'));
                            }
                        }
                    };
                    
                    window.addEventListener('message', handleMessage);
                    console.log('Sending load request for similarity model...');
                    sandboxFrame.contentWindow.postMessage({
                        type: 'LOAD_MODEL',
                        modelName: 'myModel',
                        modelPath: modelPath,
                        waitForWarmup: true
                    }, '*');
                    
                    setTimeout(() => {
                        window.removeEventListener('message', handleMessage);
                        reject(new Error('Similarity model load timeout'));
                    }, 45000);
                });
                
                console.log('Similarity model loaded and warmed up successfully');
            } catch (error) {
                modelStatus.myModel.error = error;
                console.error('Failed to load similarity model:', error);
                throw error;
            } finally {
                modelStatus.myModel.loading = false;
            }
        }
        
        // Final verification of all models
        if (!modelStatus.faceApi.loaded || !modelStatus.faceNet.loaded || !modelStatus.myModel.loaded) {
            const errors = [];
            if (!modelStatus.faceApi.loaded) errors.push('FaceAPI');
            if (!modelStatus.faceNet.loaded) errors.push('FaceNet');
            if (!modelStatus.myModel.loaded) errors.push('Similarity');
            throw new Error(`Models not loaded: ${errors.join(', ')}`);
        }
        
        // Set state after all models are confirmed loaded
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

// Replace image tracking with simplified system
const imageTracker = {
    images: new Map(), // Map<string, ImageInfo>
    maxSize: 1000,
    cleanupInterval: 60000, // Cleanup every minute
    // maxAge: 5 * 60 * 1000, // Keep items for 5 minutes
    maxAge: 1 * 10 * 1000, // Keep items for 10 seconds

    
    constructor() {
        setInterval(() => this.cleanup(), this.cleanupInterval);
    },
    
    cleanup() {
        const now = Date.now();
        for (const [src, info] of this.images) {
            if (now - info.timestamp > this.maxAge) {
                this.images.delete(src);
            }
        }
    },
    
    add(src, info = {}) {
        if (this.images.size >= this.maxSize) {
            // Remove oldest entry
            const oldestKey = Array.from(this.images.keys())[0];
            this.images.delete(oldestKey);
        }
        
        this.images.set(src, {
            isProcessed: false,
            shouldDisplay: true, // Default to true until we process faces
            hasSimilarFaces: false,
            timestamp: Date.now(),
            ...info
        });
    },
    
    markProcessed(src, shouldDisplay, hasSimilarFaces = false) {
        const info = this.images.get(src) || {};
        const newInfo = {
            ...info,
            isProcessed: true,
            shouldDisplay: shouldDisplay,
            hasSimilarFaces: hasSimilarFaces,
            timestamp: Date.now(),
            processingComplete: true // Add flag to indicate complete processing
        };
        this.images.set(src, newInfo);
        
        // Log processing completion
        console.log(`Image processing complete: ${src}`, newInfo);
    },
    
    shouldProcess(src) {
        const info = this.images.get(src);
        return !info || !info.isProcessed;
    },
    
    shouldDisplay(src) {
        const info = this.images.get(src);
        return info ? info.shouldDisplay : true;
    },
    
    has(src) {
        return this.images.has(src);
    },
    
    clear() {
        this.images.clear();
    }
};

/**
 * Checks if an element is valid for processing
 * @param {Element} element - The element to validate
 * @returns {boolean} True if the element is valid for processing
 */
function isValidElement(element) {
    // Check if element exists
    if (!element) return false;

    // Handle all possible image types
    const isImg = element.tagName === 'IMG';
    const isSvgImage = element.tagName.toLowerCase() === 'image';
    const hasSvgImage = element.querySelector && element.querySelector('image[xlink\\:href]');
    const hasBackgroundImage = window.getComputedStyle(element).backgroundImage !== 'none';

    if (!isImg && !isSvgImage && !hasSvgImage && !hasBackgroundImage) return false;

    // Get the source URL
    let src;
    if (isImg) {
        src = element.src;
    } else if (isSvgImage || hasSvgImage) {
        const imageElement = isSvgImage ? element : element.querySelector('image');
        src = imageElement?.getAttribute('xlink:href') || 
              imageElement?.getAttribute('href') ||
              imageElement?.getAttribute('src');
    } else if (hasBackgroundImage) {
        src = window.getComputedStyle(element).backgroundImage.slice(4, -1).replace(/["']/g, "");
    }

    if (!src || src.startsWith('data:') || src.includes('emoji')) return false;

    // Remove size validation - we'll handle any size
    const isHidden = element.offsetParent === null || 
                    window.getComputedStyle(element).display === 'none' ||
                    window.getComputedStyle(element).visibility === 'hidden';
    const isProcessed = element.closest('.face-detection-wrapper');
    
    return !isHidden && !isProcessed;
}

// Add cross-origin image handling function
async function createProxyImage(originalImg) {
    return new Promise((resolve, reject) => {
        const createBlobUrl = async (url) => {
            try {
                const response = await fetch(url, { mode: 'cors' });
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
            try {
                const src = originalImg.tagName === 'IMG' ? 
                    originalImg.src : 
                    originalImg.getAttribute('xlink:href');
                
                const blobUrl = await createBlobUrl(src);
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
        if (originalImg.tagName === 'IMG') {
            img.src = originalImg.src;
        } else {
            img.src = originalImg.getAttribute('xlink:href');
        }
    });
}

function roundToMultipleOf32(num) {
    return Math.ceil(num / 32) * 32;
}

// Add new function to handle tiny image processing
async function processTinyImage(img) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Calculate dimensions that are multiples of 32
    const minSize = 32; // Minimum size required by TinyFaceDetector
    const scaleFactor = Math.max(2, Math.ceil(32 / Math.min(img.width, img.height)));
    const targetWidth = roundToMultipleOf32(img.width * scaleFactor);
    const targetHeight = roundToMultipleOf32(img.height * scaleFactor);
    
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    
    // Use better upscaling algorithm
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
    
    return {
        canvas,
        scaleFactor: {
            x: targetWidth / img.width,
            y: targetHeight / img.height
        }
    };
}

function selectFaceDetectionModel(img) {
    const width = img.width || img.naturalWidth;
    const height = img.height || img.naturalHeight;
    const minDimension = Math.min(width, height);
    const maxDimension = Math.max(width, height);
    
    // Skip images smaller than minimum size
    if (minDimension < MODEL_SELECTION_THRESHOLDS.MINIMUM_SIZE) {
        throw new Error(`Image too small for face detection (${width}x${height}). Minimum size required: ${MODEL_SELECTION_THRESHOLDS.MINIMUM_SIZE}px`);
    }

    // For small images, use tinyFaceDetector with optimized settings
    if (minDimension <= MODEL_SELECTION_THRESHOLDS.SMALL_IMAGE) {
        const inputSize = roundToMultipleOf32(Math.max(32, minDimension));
        return {
            model: 'tinyFaceDetector',
            options: new faceapi.TinyFaceDetectorOptions({
                ...FACE_API_DETECTION_OPTIONS.tinyFaceDetector,
                inputSize: inputSize,
                scoreThreshold: 0.01,  // More lenient threshold for small images
                minFaceSize: Math.max(16, Math.floor(minDimension * 0.3)), // Smaller minimum face size
                scaleFactor: 0.5  // More granular scale steps
            })
        };
    }

    // For large images, use ssdMobilenetv1
    if (minDimension >= MODEL_SELECTION_THRESHOLDS.LARGE_IMAGE) {
        return {
            model: 'ssdMobilenetv1',
            options: new faceapi.SsdMobilenetv1Options({
                ...FACE_API_DETECTION_OPTIONS.ssdMobilenetv1,
                inputSize: roundToMultipleOf32(Math.min(640, minDimension))
            })
        };
    }

    // For medium-sized images, choose based on aspect ratio and image quality
    const aspectRatio = width / height;
    const isSquarish = aspectRatio > 0.7 && aspectRatio < 1.3;

    // Prefer ssdMobilenetv1 for well-proportioned medium images
    if (isSquarish && minDimension >= MODEL_SELECTION_THRESHOLDS.SMALL_IMAGE * 1.5) {
        return {
            model: 'ssdMobilenetv1',
            options: new faceapi.SsdMobilenetv1Options({
                ...FACE_API_DETECTION_OPTIONS.ssdMobilenetv1,
                inputSize: roundToMultipleOf32(Math.min(640, minDimension))
            })
        };
    }

    // Default to tinyFaceDetector for other cases with adaptive input size
    return {
        model: 'tinyFaceDetector',
        options: new faceapi.TinyFaceDetectorOptions({
            ...FACE_API_DETECTION_OPTIONS.tinyFaceDetector,
            inputSize: roundToMultipleOf32(Math.max(32, minDimension)),
            minFaceSize: Math.max(16, Math.floor(minDimension * 0.15))
        })
    };
}

// Update normalizeImageRotation to work without requiring EXIF
async function normalizeImageRotation(img) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set dimensions
    canvas.width = img.width || img.naturalWidth;
    canvas.height = img.height || img.naturalHeight;
    
    // Basic draw without rotation
    try {
        ctx.drawImage(img, 0, 0);
        
        // Try to detect rotation by analyzing the image content
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const shouldRotate = detectImageRotation(imageData);
        
        if (shouldRotate) {
            // Create new canvas with swapped dimensions
            const rotatedCanvas = document.createElement('canvas');
            const rotatedCtx = rotatedCanvas.getContext('2d');
            rotatedCanvas.width = canvas.height;
            rotatedCanvas.height = canvas.width;
            
            // Rotate 90 degrees clockwise
            rotatedCtx.translate(rotatedCanvas.width/2, rotatedCanvas.height/2);
            rotatedCtx.rotate(Math.PI/2);
            rotatedCtx.drawImage(canvas, -canvas.width/2, -canvas.height/2);
            
            return rotatedCanvas;
        }
        
        return canvas;
    } catch (error) {
        console.error('Error in image rotation:', error);
        return canvas; // Return original canvas if rotation fails
    }
}

// Add helper function to detect if image needs rotation
function detectImageRotation(imageData) {
    // Simple heuristic: check if height is significantly larger than width
    // This assumes portrait photos are more likely to need rotation
    const aspectRatio = imageData.width / imageData.height;
    return aspectRatio < 0.7; // Arbitrary threshold for portrait orientation
}

// Add helper function to detect Facebook profile images
function isFacebookProfileImage(element) {
    // Check if element is within Facebook's profile picture container
    return element.closest('[data-visualcompletion="media-vc-image"]') !== null ||
           element.closest('[data-type="profile_picture"]') !== null ||
           element.closest('.profile-photo-container') !== null;
}

// Update the detectFacesWithFaceApi function to use the helper
async function detectFacesWithFaceApi(img) {
    const wrapper = createWrapper(img);
    const processingKey = `processing_${Date.now()}`;
    wrapper.setAttribute('data-processing-key', processingKey);
    
    try {
        await Promise.all([
            ensureModelsLoaded(),
            loadPositiveEmbeddings()
        ]);
        
        const src = img.tagName === 'IMG' ? img.src : img.getAttribute('xlink:href');
        
        if (flagShowFrameonImage.addLabel) {
            addLoadingIndicator(wrapper);
        }

        let proxyImg;
        try {
            proxyImg = await createProxyImage(img);
            proxyImg = await normalizeImageRotation(proxyImg);
        } catch (error) {
            console.warn('Image normalization failed, proceeding with original:', error);
            proxyImg = img; // Fall back to original image
        }

        // Create scaled version of the image if needed
        const scaledCanvas = await createScaledImage(proxyImg);
        const scaleFactor = scaledCanvas.scaleFactor || 1;

        // Try multiple angles if initial detection fails
        let detections = [];
        // Expanded angles array to handle more orientations
        const angles = [0, -15, 15, -30, 30, -45, 45, 90, -90]; 
        
        for (const angle of angles) {
            if (detections.length === 0) {
                const rotatedCanvas = await rotateImage(scaledCanvas, angle);
                const { model, options } = selectFaceDetectionModel(rotatedCanvas);
                
                try {
                    const angleDetections = await (model === 'ssdMobilenetv1' 
                        ? faceapi.detectAllFaces(rotatedCanvas, options)
                        : faceapi.detectAllFaces(rotatedCanvas, options));
                    
                    if (angleDetections.length > 0) {
                        detections = adjustDetectionCoordinates(angleDetections, angle, rotatedCanvas);
                        console.log(`Found faces at ${angle} degrees rotation`);
                        break;
                    }
                } catch (error) {
                    console.warn(`Detection failed at ${angle} degrees:`, error);
                    continue;
                }
            }
        }

        // Scale back detections if we upscaled
        if (scaleFactor !== 1) {
            detections = detections.map(detection => ({
                ...detection,
                box: {
                    x: detection.box.x / scaleFactor,
                    y: detection.box.y / scaleFactor,
                    width: detection.box.width / scaleFactor,
                    height: detection.box.height / scaleFactor
                }
            }));
        }

        const faceEmbeddings = [];
        for (const detection of detections) {
            const faceCanvas = await extractFaceRegion(proxyImg, detection);
            try {
                const embedding = await generateEmbedding(faceCanvas);
                const comparison = await compareWithPositiveEmbeddings(embedding);
                
                faceEmbeddings.push({
                    embedding,
                    detection,
                    similarity: comparison.maxSimilarity,
                    matchIndex: comparison.matchIndex
                });
                
                console.log(`Face detected with similarity score: ${(comparison.maxSimilarity * 100).toFixed(2)}%`);
            } catch (error) {
                console.error('Embedding generation error:', error);
            }
        }

        // Store embeddings and comparison results
        if (faceEmbeddings.length > 0) {
            imageTracker.add(src, {
                embeddings: faceEmbeddings,
                timestamp: Date.now()
            });

            // Handle different processing modes
            if (flagShowFrameonImage.processingMode === 'face_detection') {
                // Face detection mode: show frames and labels
                updateVisualizationWithSimilarity(wrapper, img, faceEmbeddings);
            } else if (flagShowFrameonImage.processingMode === 'blur') {
                // Blur mode: blur faces that match the criteria
                const shouldBlur = faceEmbeddings.some(face => 
                    face.similarity >= flagShowFrameonImage.confidenceThreshold / 100
                );
                
                if (shouldBlur) {
                    // Apply blur effect to the image
                    img.style.filter = 'blur(10px)';
                    if (flagShowFrameonImage.addLabel) {
                        addResultIndicator(wrapper, 'Image blurred - Similar faces detected');
                    }
                } else {
                    img.style.filter = 'none';
                    if (flagShowFrameonImage.addLabel) {
                        addResultIndicator(wrapper, 'No matching faces detected');
                    }
                }
            }
        } else {
            // Clear any existing visualizations
            const existingCanvas = wrapper.querySelector('.face-detection-canvas');
            const processingIndicator = wrapper.querySelector('.processing-indicator');
            if (existingCanvas) existingCanvas.remove();
            if (processingIndicator) processingIndicator.remove();

            if (flagShowFrameonImage.addLabel) {
                addResultIndicator(wrapper, `No faces detected (${scaledCanvas.width}x${scaledCanvas.height})`);
            }
            
            // Ensure no blur is applied when no faces are detected
            if (flagShowFrameonImage.processingMode === 'blur') {
                img.style.filter = 'none';
            }
        }
        
        // Mark image as processed
        imageTracker.markProcessed(src, true);
        
        // Verify wrapper still exists and matches our processing key
        if (!wrapper.isConnected || wrapper.getAttribute('data-processing-key') !== processingKey) {
            console.warn('Wrapper was removed or replaced during processing');
            return;
        }
        
        if (isFacebookProfileImage(img)) {
            // Handle Facebook profile picture specific cleanup
            const svgParent = wrapper.previousSibling;
            if (svgParent?.tagName.toLowerCase() === 'svg') {
                const image = svgParent.querySelector('image');
                if (image) {
                    image.style.opacity = '0';
                }
            }
        }
        
    } catch (error) {
        console.error('Face detection error:', error);
        // Ensure image remains visible on error
        if (img.style.visibility === 'hidden') {
            img.style.visibility = 'visible';
        }
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
    batchSize: 5,  // Increased from 3 to 5 for better throughput
    processingTimeout: 20000, // 20 second timeout for processing
    
    add(element, priority = false) {
        const item = { 
            element, 
            priority,
            timestamp: Date.now() 
        };
        
        if (priority) {
            this.items.unshift(item);
        } else {
            this.items.push(item);
        }
        
        // Start processing if not already running
        if (!this.processing) {
            this.process();
        }
    },
    
    async process() {
        if (this.processing || this.items.length === 0) return;
        
        this.processing = true;
        while (this.items.length > 0) {
            // Process items in batches with timeout protection
            const batch = this.items.splice(0, this.batchSize);
            const promises = batch.map(async ({ element, timestamp }) => {
                try {
                    // Skip if item is too old
                    if (Date.now() - timestamp > this.processingTimeout) {
                        console.log('Skipping stale item');
                        return;
                    }
                    
                    const src = element.tagName === 'IMG' ? 
                        element.src : element.getAttribute('xlink:href');
                    
                    // Check cache first
                    if (imageTracker.has(src)) {
                        return;
                    }
                    
                    await Promise.race([
                        detectFacesWithFaceApi(element),
                        new Promise((_, reject) => 
                            setTimeout(() => reject(new Error('Processing timeout')), 
                            this.processingTimeout)
                        )
                    ]);
                } catch (error) {
                    console.error('Processing error:', error);
                }
            });
            
            await Promise.all(promises);
            
            // Add small delay between batches but make it dynamic
            const delay = Math.max(50, Math.min(batch.length * 20, 200));
            await new Promise(resolve => setTimeout(resolve, delay));
        }
        this.processing = false;
    }
};

// Helper functions for improved visualization
function createWrapper(element) {
    const existingWrapper = element.closest('.face-detection-wrapper');
    if (existingWrapper) return existingWrapper;

    const wrapper = document.createElement('div');
    wrapper.className = 'face-detection-wrapper';

    // Handle Facebook-style SVG profile pictures
    const isSvgImage = element.tagName.toLowerCase() === 'image';
    const svgParent = element.closest('svg');
    const isFacebookProfile = isFacebookProfileImage(element);

    if (isSvgImage) {
        const svgUrl = element.getAttribute('xlink:href') || element.getAttribute('href');
        if (!svgUrl) return null;

        // Create img element while preserving original structure
        const imgElement = document.createElement('img');
        imgElement.src = svgUrl;
        
        // Copy original dimensions and styling
        const originalRect = element.getBoundingClientRect();
        wrapper.style.width = `${originalRect.width}px`;
        wrapper.style.height = `${originalRect.height}px`;
        
        // Preserve Facebook-specific styling
        if (isFacebookProfile) {
            // Keep original container's position and structure
            wrapper.style.position = 'absolute';
            wrapper.style.inset = '0';
            wrapper.style.zIndex = '1'; // Place above original but below other UI
            
            // Copy mask and border radius if present
            const mask = svgParent.querySelector('mask');
            if (mask) {
                const rect = mask.querySelector('rect');
                if (rect) {
                    wrapper.style.borderRadius = `${rect.getAttribute('rx')}px`;
                }
            }
            
            // Maintain original image position
            imgElement.style.position = 'absolute';
            imgElement.style.width = '100%';
            imgElement.style.height = '100%';
            imgElement.style.objectFit = 'cover';
            
            // Store reference to original elements
            wrapper.setAttribute('data-original-container', isFacebookProfile.className);
            wrapper.setAttribute('data-image-type', 'facebook-profile');
            
            // Insert wrapper next to original SVG
            svgParent.parentNode.insertBefore(wrapper, svgParent.nextSibling);
            
            // Don't hide original immediately
            element.style.opacity = '0.01';
        } else {
            // Regular SVG image handling
            wrapper.style.position = 'relative';
            wrapper.style.display = 'inline-block';
            imgElement.style.width = '100%';
            imgElement.style.height = '100%';
            imgElement.style.objectFit = 'contain';
            
            element.parentNode.insertBefore(wrapper, element);
        }
        
        wrapper.appendChild(imgElement);
        return wrapper;
    }

    // Regular image handling remains unchanged
    wrapper.style.position = 'relative';
    wrapper.style.display = 'inline-block';
    wrapper.style.width = element.offsetWidth + 'px';
    wrapper.style.height = element.offsetHeight + 'px';
    
    element.style.width = '100%';
    element.style.height = '100%';
    element.style.objectFit = 'contain';
    
    element.parentNode.insertBefore(wrapper, element);
    wrapper.appendChild(element);
    
    return wrapper;
}

// Add cleanup function specific to Facebook profile pictures
function cleanupFacebookProfileWrapper(wrapper) {
    if (wrapper.getAttribute('data-image-type') === 'facebook-profile') {
        const originalContainer = wrapper.getAttribute('data-original-container');
        const svgParent = wrapper.previousSibling;
        if (svgParent?.tagName.toLowerCase() === 'svg') {
            const image = svgParent.querySelector('image');
            if (image) {
                image.style.opacity = '1';
            }
        }
    }
    wrapper.remove();
}

// Add cleanup function for SVG images
function cleanupSvgWrapper(wrapper) {
    const originalSvgId = wrapper.getAttribute('data-original-svg-id');
    if (originalSvgId) {
        const originalSvg = document.getElementById(originalSvgId);
        if (originalSvg) {
            originalSvg.style.opacity = '1';
        }
    }
    wrapper.remove();
}

// Add back the missing loading indicator function
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

// Add back the missing detection canvas function
function createDetectionCanvas(img) {
    const canvas = document.createElement('canvas');
    canvas.className = 'face-detection-canvas';
    
    // Set canvas dimensions to match original image
    const width = img.naturalWidth || img.width || parseInt(img.getAttribute('width'));
    const height = img.naturalHeight || img.height || parseInt(img.getAttribute('height'));
    canvas.width = width;
    canvas.height = height;
    
    // Position canvas as overlay
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    
    return canvas;
}

// Add back the missing result indicator function
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

// Add cleanup function for when processing is done
function cleanupWrapper(wrapper) {
    if (!wrapper) return;

    try {
        // Get original element and stored data
        const originalStyles = JSON.parse(wrapper.getAttribute('data-original-styles') || '{}');
        const imageType = wrapper.getAttribute('data-image-type');
        
        let element;
        if (imageType === 'img') {
            element = wrapper.querySelector('img');
            if (element) {
                // Restore original styles
                element.style.cssText = originalStyles.element;
                element.className = originalStyles.classes;
                wrapper.parentElement.insertBefore(element, wrapper);
            }
        } else if (imageType === 'svg' || imageType === 'svg-nested') {
            const originalId = wrapper.getAttribute('data-original-svg-id');
            const originalSvg = document.getElementById(originalId);
            if (originalSvg) {
                originalSvg.style.opacity = '1';
                originalSvg.style.cssText = originalStyles.element;
                originalSvg.className = originalStyles.classes;
            }
        } else if (imageType === 'background') {
            const bgElement = document.querySelector(`[data-original-background]`);
            if (bgElement) {
                bgElement.style.background = bgElement.getAttribute('data-original-background');
                bgElement.removeAttribute('data-original-background');
                bgElement.style.cssText = originalStyles.element;
                bgElement.className = originalStyles.classes;
            }
        }

        // Remove wrapper
        if (wrapper.parentElement) {
            wrapper.parentElement.removeChild(wrapper);
        }
    } catch (error) {
        console.error('Error cleaning up wrapper:', error);
    }
}

// Modify updateVisualizationWithSimilarity to ensure image visibility
function updateVisualizationWithSimilarity(wrapper, img, faceEmbeddings) {
    // Remove existing canvas and indicators
    const existingCanvas = wrapper.querySelector('.face-detection-canvas');
    const processingIndicator = wrapper.querySelector('.processing-indicator');
    if (existingCanvas) existingCanvas.remove();
    if (processingIndicator) processingIndicator.remove();

    // Ensure image is visible
    img.style.visibility = 'visible';
    img.style.display = 'block';

    if (faceEmbeddings.length === 0) {
        if (flagShowFrameonImage.addLabel) {
            addResultIndicator(wrapper, 'No Faces Detected');
        }
        return;
    }
    
    // Create and setup canvas
    const canvas = createDetectionCanvas(img);
    
    // Set canvas size to match actual image dimensions
    const width = img.naturalWidth || img.width || parseInt(img.getAttribute('width'));
    const height = img.naturalHeight || img.height || parseInt(img.getAttribute('height'));
    canvas.width = width;
    canvas.height = height;
    
    if (flagShowFrameonImage.frameFaceDetected) {
        drawDetectionsWithSimilarity(canvas, faceEmbeddings);
    }
    
    if (flagShowFrameonImage.addLabel) {
        const matchCount = faceEmbeddings.filter(f => f.similarity > 0.7).length;
        addResultIndicator(wrapper, 
            `${faceEmbeddings.length} Face(s) Detected, ${matchCount} Match(es)`);
    }
    
    wrapper.appendChild(canvas);
}

/**
 * Draws face detection boxes with similarity information
 * @param {HTMLCanvasElement} canvas - The canvas to draw on
 * @param {Array} faceEmbeddings - Array of face embeddings with similarity info
 */
function drawDetectionsWithSimilarity(canvas, faceEmbeddings) {
    const ctx = canvas.getContext('2d');
    ctx.lineWidth = 2;
    
    faceEmbeddings.forEach((face, index) => {
        const { x, y, width, height } = face.detection.box;
        const similarity = face.similarity || 0;
        const similarityPercentage = similarity * 100;
        
        // Calculate frame dimensions with minimal padding
        const padding = Math.min(width, height) * 0.02;
        const boxX = x - padding;
        const boxY = y - padding;
        const boxWidth = width + (padding * 2);
        const boxHeight = height + (padding * 2);
        const cornerRadius = Math.min(boxWidth, boxHeight) * 0.05;
        
        // Set colors based on similarity threshold
        let strokeColor, fillColor, labelColor;
        if (similarityPercentage >= flagShowFrameonImage.confidenceThreshold) {
            strokeColor = 'rgba(0, 255, 0, 0.9)';  // Green for match
            fillColor = 'rgba(0, 255, 0, 0.05)';
            labelColor = 'rgba(0, 255, 0, 1.0)';
        } else {
            strokeColor = 'rgba(255, 0, 0, 0.9)';  // Red for no match
            fillColor = 'rgba(255, 0, 0, 0.05)';
            labelColor = 'rgba(255, 0, 0, 1.0)';
        }
        
        // Draw rounded rectangle for face frame
        ctx.beginPath();
        ctx.moveTo(boxX + cornerRadius, boxY);
        ctx.lineTo(boxX + boxWidth - cornerRadius, boxY);
        ctx.quadraticCurveTo(boxX + boxWidth, boxY, boxX + boxWidth, boxY + cornerRadius);
        ctx.lineTo(boxX + boxWidth, boxY + boxHeight - cornerRadius);
        ctx.quadraticCurveTo(boxX + boxWidth, boxY + boxHeight, boxX + boxWidth - cornerRadius, boxY + boxHeight);
        ctx.lineTo(boxX + cornerRadius, boxY + boxHeight);
        ctx.quadraticCurveTo(boxX, boxY + boxHeight, boxX, boxY + boxHeight - cornerRadius);
        ctx.lineTo(boxX, boxY + cornerRadius);
        ctx.quadraticCurveTo(boxX, boxY, boxX + cornerRadius, boxY);
        ctx.closePath();
        
        // Draw frame with subtle shadow
        ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
        ctx.shadowBlur = 2;
        ctx.strokeStyle = strokeColor;
        ctx.stroke();
        ctx.shadowColor = 'transparent';
        
        // Fill with very subtle color
        ctx.fillStyle = fillColor;
        ctx.fill();
        
        if (flagShowFrameonImage.addLabel) {
            // Create compact label
            const labelHeight = 20;
            const labelWidth = 100;
            const labelX = boxX;
            const labelY = Math.max(0, boxY - labelHeight - 2);
            
            // Draw label background
            ctx.beginPath();
            const labelRadius = 2;
            ctx.moveTo(labelX + labelRadius, labelY);
            ctx.lineTo(labelX + labelWidth - labelRadius, labelY);
            ctx.quadraticCurveTo(labelX + labelWidth, labelY, labelX + labelWidth, labelY + labelRadius);
            ctx.lineTo(labelX + labelWidth, labelY + labelHeight - labelRadius);
            ctx.quadraticCurveTo(labelX + labelWidth, labelY + labelHeight, labelX + labelWidth - labelRadius, labelY + labelHeight);
            ctx.lineTo(labelX + labelRadius, labelY + labelHeight);
            ctx.quadraticCurveTo(labelX, labelY + labelHeight, labelX, labelY + labelHeight - labelRadius);
            ctx.lineTo(labelX, labelY + labelRadius);
            ctx.quadraticCurveTo(labelX, labelY, labelX + labelRadius, labelY);
            ctx.closePath();
            
            // Fill label background
            const gradient = ctx.createLinearGradient(labelX, labelY, labelX, labelY + labelHeight);
            gradient.addColorStop(0, 'rgba(0, 0, 0, 0.8)');
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0.6)');
            ctx.fillStyle = gradient;
            ctx.fill();
            
            // Add subtle label border
            ctx.strokeStyle = labelColor;
            ctx.lineWidth = 1;
            ctx.stroke();
            
            // Draw text with improved styling
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '11px Arial';
            ctx.textBaseline = 'middle';
            const matchText = similarityPercentage >= flagShowFrameonImage.confidenceThreshold ? 'Match' : 'No match';
            ctx.fillText(
                `Face ${index + 1} (${similarityPercentage.toFixed(1)}% - ${matchText})`,
                labelX + 4,
                labelY + (labelHeight / 2)
            );
        }
    });
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

// Add a function to ensure models are loaded
async function ensureModelsLoaded() {
    if (!modelStatus.faceApi.loaded || !modelStatus.faceNet.loaded || !modelStatus.myModel.loaded) {
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
            if (!modelStatus.myModel.loaded) {
                throw new Error('Similarity model failed to load');
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
        ...Array.from(document.querySelectorAll('svg image[xlink\\:href]')),
        ...Array.from(document.querySelectorAll('image[preserveAspectRatio="xMidYMid slice"]'))
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
            ...Array.from(document.querySelectorAll('svg image[xlink\\:href]')),
            ...Array.from(document.querySelectorAll('image[preserveAspectRatio="xMidYMid slice"]'))
        ].filter(element => isValidElement(element));
        
        console.log(`Found ${elements.length} valid images to process`);
        
        // Process images in batches with delay between batches
        const batchSize = 1;  // Keep this at 1 for better stability
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
            
            // Increase delay between batches
            await new Promise(resolve => setTimeout(resolve, 500)); // Increased from 100ms to 500ms
        }
        
        console.log('Finished processing existing images');
    } catch (error) {
        console.error('Error processing existing images:', error);
    }
}

// Modify generateEmbedding to ensure model readiness
async function generateEmbedding(faceCanvas) {
    return new Promise((resolve, reject) => {
        const checkModelStatus = (event) => {
            if (event.data.type === 'MODEL_STATUS') {
                window.removeEventListener('message', checkModelStatus);
                if (event.data.isReady) {
                    proceedWithEmbedding();
                } else {
                    reject(new Error('FaceNet model not ready: ' + (event.data.reason || 'Unknown reason')));
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
                        reject(new Error(event.data.error || 'Embedding generation failed'));
                    }
                }
            };

            window.addEventListener('message', handleEmbedding);
            const imageData = faceCanvas.getContext('2d').getImageData(0, 0, faceCanvas.width, faceCanvas.height);
            
            sandboxFrame.contentWindow.postMessage({
                type: 'GENERATE_EMBEDDING',
                imageData: Array.from(imageData.data)
            }, '*');

            setTimeout(() => {
                window.removeEventListener('message', handleEmbedding);
                reject(new Error('Embedding generation timeout'));
            }, 30000);
        };

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

// Modify loadTFModel to use retry mechanism with longer timeout
async function loadTFModel(modelName, modelPath) {
    if (modelStatus[modelName].loaded) return;
    
    if (modelStatus[modelName].loading) {
        const startTime = Date.now();
        while (modelStatus[modelName].loading && (Date.now() - startTime) < 60000) {  // Increased to 60 seconds
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
        
        // Ensure sandbox frame is ready
        if (!sandboxFrame || !sandboxFrame.contentWindow) {
            console.log('Creating sandbox frame for model loading...');
            await createSandboxFrame();
        }
        
        // Add delay to ensure frame is fully ready
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        await retryModelLoad(async () => {
            return new Promise((resolve, reject) => {
                const handleMessage = (event) => {
                    if (event.data.type === 'MODEL_LOADED' && event.data.modelName === modelName) {
                        window.removeEventListener('message', handleMessage);
                        if (event.data.success) {
                            console.log(`${modelName} model loaded successfully`);
                            modelStatus[modelName].loaded = true;
                            resolve();
                        } else {
                            console.error(`${modelName} model loading failed:`, event.data.error);
                            reject(new Error(event.data.error || `${modelName} model loading failed`));
                        }
                    }
                };
                
                window.addEventListener('message', handleMessage);
                console.log(`Sending load request for ${modelName} model...`);
                sandboxFrame.contentWindow.postMessage({ 
                    type: 'LOAD_MODEL',
                    modelName: modelName,
                    modelPath: modelPath,
                    waitForWarmup: true
                }, '*');
                
                setTimeout(() => {
                    window.removeEventListener('message', handleMessage);
                    reject(new Error('Model load response timeout'));
                }, 45000);  // Increased timeout for model loading
            });
        }, 3, 5000);  // Increased retry delay
        
    } catch (error) {
        console.error(`Error loading ${modelName} model:`, error);
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

// Update createScaledImage to handle small images
async function createScaledImage(img) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Get original dimensions
    const originalWidth = img.width || img.naturalWidth;
    const originalHeight = img.height || img.naturalHeight;
    
    // Calculate minimum required size
    const minRequiredSize = MODEL_SELECTION_THRESHOLDS.MINIMUM_SIZE;
    const smallestDimension = Math.min(originalWidth, originalHeight);
    
    let targetWidth = originalWidth;
    let targetHeight = originalHeight;
    let scaleFactor = 1;
    
    // Scale up if image is too small
    if (smallestDimension < minRequiredSize) {
        scaleFactor = Math.ceil(minRequiredSize / smallestDimension);
        targetWidth = Math.round(originalWidth * scaleFactor);
        targetHeight = Math.round(originalHeight * scaleFactor);
        
        // Ensure dimensions are multiples of 32 for better model performance
        targetWidth = Math.ceil(targetWidth / 32) * 32;
        targetHeight = Math.ceil(targetHeight / 32) * 32;
    }
    
    // Set canvas dimensions
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    
    // Use better quality settings for upscaling
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    try {
        // Draw image with scaling if needed
        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
        
        // Store scale factor for later use
        canvas.scaleFactor = scaleFactor;
        
        return canvas;
    } catch (error) {
        console.error('Error creating scaled image:', error);
        throw new Error('Failed to create scaled image canvas');
    }
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
        myModel: modelStatus.myModel.loaded,
        errors: []
    };
    
    if (!modelStatus.faceApi.loaded && modelStatus.faceApi.error) {
        status.errors.push(`FaceAPI: ${modelStatus.faceApi.error.message}`);
    }
    if (!modelStatus.faceNet.loaded && modelStatus.faceNet.error) {
        status.errors.push(`FaceNet: ${modelStatus.faceNet.error.message}`);
    }
    if (!modelStatus.myModel.loaded && modelStatus.myModel.error) {
        status.errors.push(`Similarity: ${modelStatus.myModel.error.message}`);
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
        myModel: {
            ...modelStatus.myModel,
            loading: modelStatus.myModel.loading
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

// Add function to compute similarities when needed
async function computeImageSimilarities(src) {
    const imageInfo = imageTracker.images.get(src);
    if (!imageInfo || !imageInfo.embeddings || imageInfo.embeddings.length <= 1) {
        return;
    }

    // Only compute similarities if we haven't already
    if (!imageInfo.similarities) {
        try {
            const similarities = [];
            const embeddings = imageInfo.embeddings;
            
            for (let i = 0; i < embeddings.length; i++) {
                for (let j = i + 1; j < embeddings.length; j++) {
                    try {
                        const similarity = await computeFaceSimilarity(
                            embeddings[i].embedding,
                            embeddings[j].embedding
                        );
                        similarities.push({
                            face1: i + 1,
                            face2: j + 1,
                            similarity
                        });
                    } catch (error) {
                        if (error.message !== 'Similarity model not loaded') {
                            console.error(`Error computing similarity between faces ${i + 1} and ${j + 1}:`, error);
                        }
                    }
                }
            }
            
            imageInfo.similarities = similarities;
        } catch (error) {
            console.error('Error computing similarities:', error);
        }
    }
}

// Add message listener for settings updates
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'SETTINGS_UPDATED') {
        flagShowFrameonImage = {
            ...flagShowFrameonImage,
            ...message.settings
        };
        
        // Reprocess visible images with new settings
        if (flagShowFrameonImage.autoProcessImages) {
            processExistingImages();
        }
    }
});

// Load initial settings
chrome.storage.sync.get({
    frameProsessedImage: true,
    frameFaceDetected: true,
    addLabel: true,
    autoProcessImages: true,
    minimumImageSize: 100,
    confidenceThreshold: 70,
    processingMode: 'face_detection'  // Add default mode
}, function(items) {
    flagShowFrameonImage = {
        ...flagShowFrameonImage,
        ...items
    };
});

// Add function to check if image should be displayed
function shouldDisplayImage(src) {
    return imageTracker.shouldDisplay(src);
}

// Modify storage event listener to handle display updates and mode changes
chrome.storage.onChanged.addListener((changes, namespace) => {
    if (namespace === 'sync') {
        let needsReprocessing = false;
        
        // Check for relevant setting changes
        if (changes.confidenceThreshold || changes.processingMode) {
            needsReprocessing = true;
        }
        
        if (needsReprocessing) {
            // Clear processed status to allow reprocessing
            imageTracker.images.forEach((info, src) => {
                info.isProcessed = false;
            });
            
            // Reprocess visible images
            if (flagShowFrameonImage.autoProcessImages) {
                processExistingImages();
            }
        }
    }
});

function applyBlurEffect(element, shouldBlur) {
    const wrapper = element.closest('.face-detection-wrapper');
    if (!wrapper) return;

    const imageType = wrapper.getAttribute('data-image-type');
    const blurAmount = shouldBlur ? '10px' : '0px';
    
    // Use CSS transform to trigger GPU acceleration
    const transform = shouldBlur ? 'translateZ(0)' : 'none';
    
    switch (imageType) {
        case 'img':
            const img = wrapper.querySelector('img');
            if (img) {
                img.style.filter = `blur(${blurAmount})`;
                img.style.transform = transform;
                // Add will-change to hint browser about animation
                img.style.willChange = shouldBlur ? 'filter' : 'auto';
            }
            break;

        case 'svg':
        case 'svg-nested':
            const svgImage = wrapper.querySelector('image');
            if (svgImage) {
                if (shouldBlur) {
                    const filterId = `blur-${Math.random().toString(36).substr(2, 9)}`;
                    const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
                    filter.setAttribute('id', filterId);
                    const blur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
                    blur.setAttribute('stdDeviation', '5');
                    filter.appendChild(blur);
                    svgImage.closest('svg').appendChild(filter);
                    svgImage.setAttribute('filter', `url(#${filterId})`);
                } else {
                    svgImage.removeAttribute('filter');
                    const filters = svgImage.closest('svg').querySelectorAll('filter');
                    filters.forEach(filter => filter.remove());
                }
            }
            break;

        case 'background':
            const bgElement = wrapper.querySelector('img');
            if (bgElement) {
                bgElement.style.filter = `blur(${blurAmount})`;
                bgElement.style.transform = transform;
                bgElement.style.willChange = shouldBlur ? 'filter' : 'auto';
            }
            break;
    }
}

// Add debounced reprocess function
const debouncedReprocess = debounce(async () => {
    if (flagShowFrameonImage.autoProcessImages) {
        await processExistingImages();
        observeElements();
    }
}, 250);

// Helper debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Replace the old WorkerPool instantiation with EnhancedWorkerPool
const workerPool = new EnhancedWorkerPool({
    maxWorkers: 4,
    taskTimeout: 30000,
    retryAttempts: 2,
    batchSize: 4
});

// Initialize the worker pool during extension initialization
async function initializeExtension() {
    try {
        // Initialize worker pool
        await workerPool.initialize();
        
        // Load models and other initialization
        await loadFaceApiModels();
        await loadPositiveEmbeddings();
        
        if (flagShowFrameonImage.autoProcessImages) {
            await processExistingImages();
            observeElements();
        }
    } catch (error) {
        console.error('Extension initialization failed:', error);
    }
}

// Add helper function to rotate image
async function rotateImage(canvas, angle) {
    const rotatedCanvas = document.createElement('canvas');
    const ctx = rotatedCanvas.getContext('2d');
    
    // Calculate new dimensions to fit rotated image
    const radians = (angle * Math.PI) / 180;
    const sin = Math.abs(Math.sin(radians));
    const cos = Math.abs(Math.cos(radians));
    const width = canvas.width;
    const height = canvas.height;
    rotatedCanvas.width = width * cos + height * sin;
    rotatedCanvas.height = width * sin + height * cos;
    
    // Move to center and rotate
    ctx.translate(rotatedCanvas.width/2, rotatedCanvas.height/2);
    ctx.rotate(radians);
    ctx.drawImage(canvas, -width/2, -height/2);
    
    return rotatedCanvas;
}

// Add helper function to adjust detection coordinates
function adjustDetectionCoordinates(detections, angle, canvas) {
    const radians = (-angle * Math.PI) / 180;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    return detections.map(detection => {
        const { x, y, width, height } = detection.box;
        const cx = x + width/2 - centerX;
        const cy = y + height/2 - centerY;
        
        // Rotate coordinates back
        const rotatedX = cx * Math.cos(radians) - cy * Math.sin(radians);
        const rotatedY = cx * Math.sin(radians) + cy * Math.cos(radians);
        
        return {
            ...detection,
            box: {
                x: rotatedX + centerX - width/2,
                y: rotatedY + centerY - height/2,
                width,
                height
            }
        };
    });
}

// Add recovery function
function recoverFailedImage(img) {
    // Restore original visibility
    img.style.visibility = 'visible';
    img.style.opacity = '1';
    
    // Remove any processing-related classes/attributes
    const wrapper = img.closest('.face-detection-wrapper');
    if (wrapper) {
        const originalStyles = JSON.parse(wrapper.getAttribute('data-original-styles') || '{}');
        Object.assign(img.style, originalStyles);
        
        // Unwrap the image if needed
        if (wrapper.parentNode) {
            wrapper.parentNode.insertBefore(img, wrapper);
            wrapper.remove();
        }
    }
}

// Add to error handling
window.addEventListener('error', function(event) {
    if (event.target.tagName === 'IMG') {
        console.warn('Recovering failed image:', event.target.src);
        recoverFailedImage(event.target);
    }
});