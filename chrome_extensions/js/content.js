/**
 * @fileoverview Content script for FaceOne Chrome extension.
 * Provides real-time face detection and embedding generation for web images.
 * Uses FaceAPI.js for detection and FaceNet for embedding generation.
 * @author Liron Farzam
 * @version 1.0.0
 */

// Load utils.js dynamically
(function loadUtilsScript() {
    // Define a function to check if logging functions are available
    function checkLoggingFunctions() {
        return typeof logWithEmoji === 'function' && 
               typeof logFunctionEntry === 'function' && 
               typeof logError === 'function';
    }
    
    // If logging functions are already available, no need to load utils.js
    if (checkLoggingFunctions()) {
        console.log('✅ Logging functions already available');
        return;
    }
    
    try {
        // Check if we're in a content script context (chrome.runtime will be available)
        if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.getURL) {
            // Create a script element for utils.js
            const script = document.createElement('script');
            script.src = chrome.runtime.getURL('js/utils.js');
            script.onload = function() {
                if (!window.DEBUG) {
                    console.log('✅ utils.js loaded successfully from extension');
                }
                
                // Verify that logging functions are now available
                if (!checkLoggingFunctions()) {
                    console.warn('⚠️ utils.js loaded but logging functions not found');
                }
            };
            script.onerror = function(error) {
                console.error('❌ Failed to load utils.js from extension:', error);
            };
            
            // Add the script to the document
            (document.head || document.documentElement).appendChild(script);
        } else {
            console.warn('⚠️ Not in a content script context, cannot load utils.js from extension');
        }
    } catch (error) {
        console.error('❌ Error loading utils.js:', error);
    }
})();

// Fallback logging utilities in case utils.js is not yet loaded
if (typeof logWithEmoji !== 'function') {
    window.DEBUG = false; // Initialize with debugging off
    
    window.logWithEmoji = function(type, functionName, message, details = null) {
        // Critical messages are always shown (errors and warnings)
        const isCritical = ['error', 'warning'].includes(type);
        
        // Model and embedding loading messages are shown in both modes, but with different detail levels
        const isModelRelated = ['model', 'loading'].includes(type) && 
                             (functionName.includes('Model') || 
                              functionName.includes('load') || 
                              message.includes('model') || 
                              message.includes('embedding'));
        
        // Only log if either DEBUG is enabled, or it's a critical message, or it's a success related to models/embeddings
        const isSuccess = type === 'success' && isModelRelated;
        
        if (!window.DEBUG && !isCritical && !isSuccess) {
            return;
        }
        
        let emoji = '📝'; // Default emoji
        
        // Select emoji based on log type
        switch (type) {
            case 'info': emoji = '📋'; break;
            case 'success': emoji = '✅'; break;
            case 'warning': emoji = '⚠️'; break;
            case 'error': emoji = '❌'; break;
            case 'model': emoji = '🧠'; break;
            case 'image': emoji = '🖼️'; break;
            case 'loading': emoji = '🔄'; break;
            case 'setup': emoji = '🔧'; break;
            case 'timer': emoji = '⏱️'; break;
            case 'search': emoji = '🔍'; break;
            case 'lock': emoji = '🔒'; break;
            case 'unlock': emoji = '🔓'; break;
            case 'start': emoji = '🚀'; break;
            case 'draw': emoji = '🎨'; break;
            case 'performance': emoji = '📊'; break;
            case 'stats': emoji = '📈'; break;
            case 'network': emoji = '🌐'; break;
            case 'processing': emoji = '⚙️'; break;
        }
        
        // Basic logging for non-debug mode
        if (!window.DEBUG) {
            console.log(`Face One - ${emoji} ${functionName}: ${message}`);
            return;
        }
        
        // Enhanced logging for debug mode
        if (details) {
            console.log(`Face One - ${emoji} ${functionName}: ${message}`, details);
        } else {
            console.log(`Face One - ${emoji} ${functionName}: ${message}`);
        }
    };
}

if (typeof logFunctionEntry !== 'function') {
    window.logFunctionEntry = function(functionName) {
        if (window.DEBUG && typeof logWithEmoji === 'function') {
            logWithEmoji('setup', functionName, 'Function started');
        } else if (window.DEBUG) {
            console.log(`🔧 ${functionName}: Function started`);
        }
    };
}

if (typeof logError !== 'function') {
    window.logError = function(functionName, message, error = null) {
        if (typeof logWithEmoji === 'function') {
            logWithEmoji('error', functionName, message);
        } else {
            console.error(`❌ ${functionName}: ${message}`);
        }
        
        if (error && error.stack) {
            console.error(`${functionName} error stack:`, error.stack);
        } else if (error) {
            console.error(`${functionName} error details:`, error);
        }
    };
}

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
    minimumImageSize: 32,
    confidenceThreshold: 70,
    processingMode: 'face_detection'
};

//=============================================================================
// 4. Constants and Configuration
//=============================================================================
const MODEL_SELECTION_THRESHOLDS = {
    get MINIMUM_SIZE() {
        // Change minimum size to 32px (or user setting, whichever is smaller)
        return Math.min(32, flagShowFrameonImage.minimumImageSize || 32);
    },
    get SMALL_IMAGE() {
        return Math.max(64, this.MINIMUM_SIZE * 2);
    },
    get LARGE_IMAGE() {
        return Math.max(256, this.SMALL_IMAGE * 2);
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
    logFunctionEntry('loadPositiveEmbeddings');
    
    if (isPositiveEmbeddingsLoaded) {
        logWithEmoji('info', 'loadPositiveEmbeddings', 'Positive embeddings already loaded');
        return;
    }

    try {
        const embeddingsPath = chrome.runtime.getURL('models/embeddings/positive_embeddings.json');
        logWithEmoji('loading', 'loadPositiveEmbeddings', 'Loading positive embeddings', { path: embeddingsPath });
        
        const startTime = performance.now();
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
        logWithEmoji('processing', 'loadPositiveEmbeddings', 'Converting embeddings to Float32Array', 
            { count: limitedData.length, dimensions: limitedData[0]?.length || 0 });
        
        positiveEmbeddings = limitedData.map(embedding => {
            if (!Array.isArray(embedding) || embedding.length !== 512) {
                throw new Error('Invalid embedding format: expected 512-dimensional array');
            }
            return new Float32Array(embedding);
        });
        
        isPositiveEmbeddingsLoaded = true;
        const loadTime = Math.round(performance.now() - startTime);
        
        logWithEmoji('success', 'loadPositiveEmbeddings', 
            `Loaded ${positiveEmbeddings.length} positive embeddings successfully (${loadTime}ms)`, 
            { count: positiveEmbeddings.length, loadTimeMs: loadTime });
    } catch (error) {
        logWithEmoji('error', 'loadPositiveEmbeddings', `Error loading positive embeddings: ${error.message}`, error);
        throw error;
    }
}

/**
 * Compares a face embedding with all positive embeddings
 * @param {Float32Array} faceEmbedding - The embedding to compare
 * @returns {Promise<{maxSimilarity: number, matchIndex: number}>}
 */
async function compareWithPositiveEmbeddings(faceEmbedding) {
    logFunctionEntry('compareWithPositiveEmbeddings');
    logWithEmoji('model', 'compareWithPositiveEmbeddings', 'Comparing face embedding with positive embeddings');
    if (!isPositiveEmbeddingsLoaded) {
        logWithEmoji('error', 'compareWithPositiveEmbeddings', 'Positive embeddings not loaded');
        throw new Error('Positive embeddings not loaded');
    }
    logWithEmoji('success', 'compareWithPositiveEmbeddings', 'Positive embeddings loaded successfully');
    let maxSimilarity = -1;
    let matchIndex = -1;
    logWithEmoji('loading', 'compareWithPositiveEmbeddings', 'Comparing face embedding with positive embeddings');
    for (let i = 0; i < positiveEmbeddings.length; i++) {
        try {
            logWithEmoji('loading', 'compareWithPositiveEmbeddings', `Comparing with positive embedding ${i}`);
            const similarity = await computeFaceSimilarity(faceEmbedding, positiveEmbeddings[i]);
            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                matchIndex = i;
            }
            logWithEmoji('success', 'compareWithPositiveEmbeddings', `Positive embedding ${i} compared successfully`);
        } catch (error) {
            logWithEmoji('error', 'compareWithPositiveEmbeddings', `Error comparing with positive embedding ${i}:`, error);
        }
    }
    logWithEmoji('success', 'compareWithPositiveEmbeddings', 'All positive embeddings compared successfully');
    return { maxSimilarity, matchIndex };
}

//=============================================================================
// Model Management
//=============================================================================

let sandboxFrame = null;
let faceNetModel = null;

/**
 * Creates sandbox iframe for TensorFlow operations and waits for it to be ready
 * With built-in retry mechanism to avoid page refreshes
 */
async function createSandboxFrame(attemptCount = 1) {
    logFunctionEntry('createSandboxFrame');
    logWithEmoji('setup', 'createSandboxFrame', 'Creating sandbox iframe for TensorFlow operations');
    const maxRetries = 3;
    const baseTimeout = 5000;  // Reduced from 10 seconds to 5 seconds for faster loading
    const timeout = Math.min(baseTimeout * (1 + attemptCount * 0.5), 15000); // Reduced max to 15s from 30s
    
    // OPTIMIZATION: Check for existing frame first
    if (sandboxFrame && sandboxFrame.contentWindow) {
        logWithEmoji('success', 'createSandboxFrame', 'Existing sandbox frame found');
        // Check if already initialized using a faster method
        try {
            logWithEmoji('loading', 'createSandboxFrame', 'Checking existing sandbox frame status');
            
            // Use a cached status check with timeout
            const statusCheckPromise = checkTensorFlowStatus();
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Status check timeout')), 2000); // Shorter timeout for faster response
            });
            
            const status = await Promise.race([statusCheckPromise, timeoutPromise]);
            
            if (status.isInitialized && status.tfBackendInitialized) {
                logWithEmoji('success', 'createSandboxFrame', 'Existing sandbox frame is already initialized');
                return;
            } else {
                logWithEmoji('warning', 'createSandboxFrame', 'Existing sandbox frame found but TensorFlow not initialized, recreating...');
                // Continue to recreate the frame
            }
        } catch (e) {
            logWithEmoji('warning', 'createSandboxFrame', 'Error checking existing sandbox frame, will recreate: ' + e.message);
            // Continue to recreate the frame
        }
    }

    // Cleanup any existing frame
    if (sandboxFrame) {
        try {
            logWithEmoji('loading', 'createSandboxFrame', 'Removing existing sandbox frame');
            document.body.removeChild(sandboxFrame);
            logWithEmoji('success', 'createSandboxFrame', 'Existing sandbox frame removed successfully');
        } catch (e) {
            logWithEmoji('warning', 'createSandboxFrame', 'Error removing existing sandbox frame: ' + e.message);
        }
        sandboxFrame = null;
    }

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            logWithEmoji('loading', 'createSandboxFrame', `Creating new sandbox frame (attempt ${attempt}/${maxRetries})...`);
            await createFrame(timeout);
            logWithEmoji('success', 'createSandboxFrame', 'Sandbox frame created and TensorFlow initialized');
            return;
        } catch (error) {
            logWithEmoji('error', 'createSandboxFrame', `Failed to create sandbox frame on attempt ${attempt}/${maxRetries}: ${error.message}`);
            
            // Cleanup on error
            if (sandboxFrame) {
                try {
                    logWithEmoji('loading', 'createSandboxFrame', 'Removing existing sandbox frame during retry');
                    document.body.removeChild(sandboxFrame);
                    logWithEmoji('success', 'createSandboxFrame', 'Existing sandbox frame removed successfully during retry');
                } catch (e) {
                    logWithEmoji('warning', 'createSandboxFrame', 'Error removing sandbox frame during retry: ' + e.message);
                }
                sandboxFrame = null;
            }
            
            if (attempt < maxRetries) {
                const retryDelay = 500 * attempt; // Exponential backoff
                logWithEmoji('timer', 'createSandboxFrame', `Waiting ${retryDelay}ms before retrying...`);
                await new Promise(resolve => setTimeout(resolve, retryDelay));
            } else {
                logWithEmoji('error', 'createSandboxFrame', `Failed to create sandbox frame after ${maxRetries} attempts: ${error.message}`);
                throw error;
            }
        }
    }
}

async function createFrame(timeout) {
    logFunctionEntry('createFrame');
    return new Promise((resolve, reject) => {
        let loadTimeout = null;
        let tfInitTimeout = null;
        
        const cleanup = () => {
            if (loadTimeout) {
                clearTimeout(loadTimeout);
                loadTimeout = null;
            }
            if (tfInitTimeout) {
                clearTimeout(tfInitTimeout);
                tfInitTimeout = null;
            }
            window.removeEventListener('message', handleTfInit);
            if (sandboxFrame) {
                sandboxFrame.removeEventListener('load', handleLoad);
                sandboxFrame.removeEventListener('error', handleError);
            }
        };
        
        const handleTfInit = (event) => {
            logWithEmoji('loading', 'createFrame', 'Handling TensorFlow initialization');
            if (event.data && event.data.type === 'TF_INITIALIZED') {
                cleanup();
                if (event.data.success) {
                    logWithEmoji('success', 'createFrame', 'TensorFlow initialized successfully:', event.data.info);
                    resolve();
                } else {
                    logWithEmoji('error', 'createFrame', 'TF initialization failed:', event.data.error, 'Status:', event.data.status);
                    reject(new Error('TF initialization failed: ' + (event.data.error || 'Unknown error')));
                }
            }
        };
        
        const handleLoad = () => {
            logWithEmoji('loading', 'createFrame', 'Sandbox frame loaded, waiting for TF initialization...');
            window.addEventListener('message', handleTfInit);
            logWithEmoji('success', 'createFrame', 'TF initialization listener added');
            // OPTIMIZATION: Shorter TF init timeout for faster failure detection
            tfInitTimeout = setTimeout(() => {
                logWithEmoji('loading', 'createFrame', 'TF initialization timeout, cleaning up...');
                cleanup();
                reject(new Error(`TF initialization timeout after ${timeout}ms`));
            }, timeout);
        };
        
        const handleError = (error) => {
            cleanup();
            logWithEmoji('error', 'createFrame', 'Sandbox frame failed to load: ' + (error.message || 'Unknown error'));
            reject(new Error('Sandbox frame failed to load: ' + (error.message || 'Unknown error')));
        };
        
        // Create new iframe
        sandboxFrame = document.createElement('iframe');
        sandboxFrame.id = 'face-api-sandbox';
        sandboxFrame.style.display = 'none';
        sandboxFrame.setAttribute('sandbox', 'allow-scripts allow-same-origin');
        
        // Add load and error event listeners
        sandboxFrame.addEventListener('load', handleLoad);
        sandboxFrame.addEventListener('error', handleError);
        
        // Set load timeout
        loadTimeout = setTimeout(() => {
            logWithEmoji('loading', 'createFrame', 'Frame load timeout, cleaning up...');
            cleanup();
            reject(new Error(`Frame load timeout after ${timeout}ms`));
        }, timeout);
        
        // Set source and append to document
        sandboxFrame.src = chrome.runtime.getURL('sandbox.html');
        document.body.appendChild(sandboxFrame);
    });
}

/**
 * Loads the FaceNet model in sandbox with built-in retry mechanism
 */
async function loadFaceNetModel() {
    logFunctionEntry('loadFaceNetModel');
    logWithEmoji('model', 'loadFaceNetModel', 'Loading FaceNet model');
    if (modelStatus.faceNet.loaded) {
        logWithEmoji('success', 'loadFaceNetModel', 'FaceNet model already loaded');
        return;
    }

    logWithEmoji('loading', 'loadFaceNetModel', 'Starting FaceNet model load...');
    const modelPath = chrome.runtime.getURL('models/FaceNet/Facenet512_tfjs_graph_model/model.json');
    const maxRetries = 3;
    const startTime = performance.now();
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        logWithEmoji('loading', 'loadFaceNetModel', `FaceNet load attempt ${attempt}/${maxRetries}`);
        
        try {
            // Create sandbox frame if it doesn't exist
            if (!sandboxFrame || !sandboxFrame.contentWindow) {
                logWithEmoji('setup', 'loadFaceNetModel', 'Creating sandbox frame before loading FaceNet...');
                await createSandboxFrame(attempt);
            }
            
            // Verify TensorFlow is ready with shorter timeout
            const tfStatusPromise = checkTensorFlowStatus();
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('TensorFlow status check timeout')), 3000);
            });
            
            let tfStatus;
            try {
                tfStatus = await Promise.race([tfStatusPromise, timeoutPromise]);
            } catch (error) {
                logWithEmoji('error', 'loadFaceNetModel', 'TensorFlow status check failed: ' + error.message);
                // Try to recreate the sandbox frame
                if (sandboxFrame) {
                    try {
                        logWithEmoji('loading', 'loadFaceNetModel', 'Removing existing sandbox frame after status check failure');
                        document.body.removeChild(sandboxFrame);
                        logWithEmoji('success', 'loadFaceNetModel', 'Existing sandbox frame removed successfully');
                    } catch (e) {
                        logWithEmoji('warning', 'loadFaceNetModel', 'Error removing sandbox frame: ' + e.message);
                    }
                    sandboxFrame = null;
                }
                
                await createSandboxFrame(attempt + 1);
                tfStatus = await checkTensorFlowStatus();
            }
            
            if (!tfStatus.isInitialized || !tfStatus.tfBackendInitialized) {
                logWithEmoji('warning', 'loadFaceNetModel', 'TensorFlow not initialized properly. Status: ' + JSON.stringify(tfStatus));
                
                // Try to recreate the sandbox frame
                if (sandboxFrame) {
                    try {
                        logWithEmoji('loading', 'loadFaceNetModel', 'Removing existing sandbox frame');
                        document.body.removeChild(sandboxFrame);
                        logWithEmoji('success', 'loadFaceNetModel', 'Existing sandbox frame removed successfully');
                    } catch (e) {
                        logWithEmoji('warning', 'loadFaceNetModel', 'Error removing sandbox frame: ' + e.message);
                    }
                    sandboxFrame = null;
                }
                
                await createSandboxFrame(attempt + 1);
                // Verify TensorFlow status again
                const newStatus = await checkTensorFlowStatus();
                if (!newStatus.isInitialized || !newStatus.tfBackendInitialized) {
                    logWithEmoji('error', 'loadFaceNetModel', 'TensorFlow still not initialized after sandbox frame recreation');
                    throw new Error('TensorFlow still not initialized after sandbox frame recreation');
                }
            }
            
            // Load the model with a more efficient timeout management
            const result = await new Promise((resolve, reject) => {
                logWithEmoji('loading', 'loadFaceNetModel', 'Loading FaceNet model in sandbox...');
                
                // Create a cleanup function for all event listeners and timeouts
                let messageHandler = null;
                let timeoutId = null;
                
                const cleanup = () => {
                    if (timeoutId) clearTimeout(timeoutId);
                    if (messageHandler) window.removeEventListener('message', messageHandler);
                };
                
                messageHandler = (event) => {
                    if (event.data.type === 'MODEL_LOADED' && event.data.modelName === 'faceNet') {
                        cleanup();
                        if (event.data.success) {
                            const loadTime = performance.now() - startTime;
                            if (event.data.modelInfo && event.data.modelInfo.warmedUp) {
                                logWithEmoji('success', 'loadFaceNetModel', `FaceNet model loaded and warmed up successfully in ${Math.round(loadTime)}ms`);
                                resolve(true);
                            } else {
                                logWithEmoji('warning', 'loadFaceNetModel', `FaceNet model loaded but not warmed up in ${Math.round(loadTime)}ms`);
                                resolve(false);
                            }
                        } else {
                            logWithEmoji('error', 'loadFaceNetModel', 'FaceNet model loading failed:', event.data.error);
                            reject(new Error(event.data.error || 'FaceNet model loading failed'));
                        }
                    }   
                };
                
                // Set a timeout proportional to the attempt number with a maximum
                const timeoutDuration = Math.min(20000 + (attempt - 1) * 5000, 30000);
                timeoutId = setTimeout(() => {
                    cleanup();
                    logWithEmoji('error', 'loadFaceNetModel', `FaceNet model load timeout after ${timeoutDuration}ms`);
                    reject(new Error('FaceNet model load timeout'));
                }, timeoutDuration);
                
                window.addEventListener('message', messageHandler);
                
                try {
                    logWithEmoji('loading', 'loadFaceNetModel', 'Sending FaceNet load request to sandbox...');
                    sandboxFrame.contentWindow.postMessage({
                        type: 'LOAD_MODEL',
                        modelName: 'faceNet',
                        modelPath: modelPath,
                        waitForWarmup: true
                    }, '*');
                    logWithEmoji('success', 'loadFaceNetModel', 'FaceNet load request sent to sandbox successfully');
                } catch (e) {
                    cleanup();
                    logWithEmoji('error', 'loadFaceNetModel', 'Error sending load request:', e);
                    reject(new Error('Error sending load request: ' + e.message));
                }
            });
            
            if (result === true) {
                modelStatus.faceNet.loaded = true;
                state.faceNetLoaded = true;
                const totalLoadTime = performance.now() - startTime;
                logWithEmoji('success', 'loadFaceNetModel', `FaceNet model fully loaded and ready in ${Math.round(totalLoadTime)}ms`);
                return;
            } else {
                logWithEmoji('warning', 'loadFaceNetModel', 'FaceNet loaded but not warmed up, retrying...');
                if (attempt === maxRetries) {
                    // On last attempt, accept not warmed up
                    modelStatus.faceNet.loaded = true;
                    state.faceNetLoaded = true;
                    const totalLoadTime = performance.now() - startTime;
                    logWithEmoji('warning', 'loadFaceNetModel', `Accepting FaceNet model without warmup after ${Math.round(totalLoadTime)}ms as this was the last attempt`);
                    return;
                }
            }
        } catch (error) {
            logWithEmoji('error', 'loadFaceNetModel', `FaceNet load attempt ${attempt} failed: ${error.message}`);
            
            if (attempt === maxRetries) {
                modelStatus.faceNet.error = error;
                const totalTime = performance.now() - startTime;
                logWithEmoji('error', 'loadFaceNetModel', `Failed to load FaceNet after ${maxRetries} attempts in ${Math.round(totalTime)}ms: ${error.message}`);
                throw new Error(`Failed to load FaceNet after ${maxRetries} attempts: ${error.message}`);
            }
            
            // Wait before retry with increasing but smaller delay
            const delay = 1000 * attempt; // 1s, 2s, 3s (reduced from 2s base)
            logWithEmoji('timer', 'loadFaceNetModel', `Waiting ${delay}ms before retrying FaceNet load...`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
}

// Add loading lock
let isLoadingModels = false;
let modelLoadLockTimeout = null;

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
 * Loads specified face API models with optimized resource management
 * @param {Array<string>} modelList - List of models to load: ['tinyFaceDetector', 'ssdMobilenetv1', 'faceNet', 'myModel']
 * @returns {Promise<boolean>} - Promise resolving to true if models loaded successfully
 */
async function loadFaceApiModels(modelList = ['tinyFaceDetector', 'ssdMobilenetv1', 'faceNet', 'myModel']) {
    logFunctionEntry('loadFaceApiModels');
    
    // Prevent concurrent loading
    if (isLoadingModels) {
        logWithEmoji('lock', 'loadFaceApiModels', 'Model loading already in progress, waiting...');
        
        // Wait for current loading to complete with timeout
        return new Promise((resolve, reject) => {
            const checkLoaded = setInterval(() => {
                if (!isLoadingModels) {
                    clearInterval(checkLoaded);
                    
                    // Verify models were loaded
                    const allLoaded = modelList.every(model => {
                        switch (model) {
                            case 'tinyFaceDetector':
                                return modelStatus.faceApi.loaded;
                            case 'ssdMobilenetv1':
                                return modelStatus.faceApi.loaded;
                            case 'faceNet':
                                return modelStatus.faceNet.loaded;
                            case 'myModel':
                                return modelStatus.myModel.loaded;
                            default:
                                return false;
                        }
                    });
                    
                    if (allLoaded) {
                        resolve(true);
                    } else {
                        resolve(loadFaceApiModels(modelList)); // Try loading again
                    }
                }
            }, 100);
            
            // Set timeout
            setTimeout(() => {
                clearInterval(checkLoaded);
                reject(new Error('Timed out waiting for model loading'));
            }, 10000);
        });
    }
    
    // Set loading lock with timeout
    isLoadingModels = true;
    modelLoadLockTimeout = setTimeout(() => {
        isLoadingModels = false;
        logWithEmoji('warning', 'loadFaceApiModels', 'Model loading lock timed out, releasing');
    }, 30000);
    
    logWithEmoji('loading', 'loadFaceApiModels', `Loading models: ${modelList.join(', ')}`);
    
    try {
        // Determine which models need to be loaded
        const modelsToLoad = [];
        
        if (modelList.includes('tinyFaceDetector') && !modelStatus.faceApi.loaded) {
            modelsToLoad.push('tinyFaceDetector');
        }
        
        if (modelList.includes('ssdMobilenetv1') && !modelStatus.faceApi.loaded) {
            modelsToLoad.push('ssdMobilenetv1');
        }
        
        if (modelList.includes('faceNet') && !modelStatus.faceNet.loaded) {
            modelsToLoad.push('faceNet');
        }
        
        if (modelList.includes('myModel') && !modelStatus.myModel.loaded) {
            modelsToLoad.push('myModel');
        }
        
        if (modelsToLoad.length === 0) {
            logWithEmoji('success', 'loadFaceApiModels', 'All requested models already loaded');
            return true;
        }
        
        const loadPromises = [];
        
        // Load TinyFaceDetector if needed
        if (modelsToLoad.includes('tinyFaceDetector')) {
            logWithEmoji('model', 'loadFaceApiModels', 'Loading TinyFaceDetector model...');
            const tinyFaceDetectorPromise = (async () => {
                try {
                    const modelPath = chrome.runtime.getURL('models/FaceAPI/tiny_face_detector');
                    await faceapi.nets.tinyFaceDetector.loadFromUri(modelPath);
                    logWithEmoji('success', 'loadFaceApiModels', 'TinyFaceDetector loaded successfully');
                    modelStatus.faceApi.loaded = true;
                    return true;
                } catch (error) {
                    logWithEmoji('error', 'loadFaceApiModels', 'Failed to load TinyFaceDetector: ' + error.message);
                    throw error;
                }
            })();
            
            loadPromises.push(tinyFaceDetectorPromise);
        }
        
        // Load SSD MobileNet if needed
        if (modelsToLoad.includes('ssdMobilenetv1')) {
            logWithEmoji('model', 'loadFaceApiModels', 'Loading SSD MobileNet model...');
            const ssdPromise = (async () => {
                try {
                    const modelPath = chrome.runtime.getURL('models/FaceAPI/ssd_mobilenetv1');
                    await faceapi.nets.ssdMobilenetv1.loadFromUri(modelPath);
                    logWithEmoji('success', 'loadFaceApiModels', 'SSD MobileNet loaded successfully');
                    modelStatus.faceApi.loaded = true;
                    return true;
                } catch (error) {
                    logWithEmoji('error', 'loadFaceApiModels', 'Failed to load SSD MobileNet: ' + error.message);
                    throw error;
                }
            })();
            
            loadPromises.push(ssdPromise);
        }
        
        // Load FaceNet if needed
        if (modelsToLoad.includes('faceNet')) {
            logWithEmoji('model', 'loadFaceApiModels', 'Loading FaceNet model...');
            const faceNetPromise = (async () => {
                try {
                    await loadFaceNetModel();
                    logWithEmoji('success', 'loadFaceApiModels', 'FaceNet model loaded successfully');
                    return true;
                } catch (error) {
                    logWithEmoji('error', 'loadFaceApiModels', 'Failed to load FaceNet: ' + error.message);
                    throw error;
                }
            })();
            
            loadPromises.push(faceNetPromise);
        }
        
        // Load similarity model (myModel) if needed
        if (modelsToLoad.includes('myModel')) {
            logWithEmoji('model', 'loadFaceApiModels', 'Loading similarity model...');
            const myModelPromise = (async () => {
                try {
                    await loadMyModel();
                    logWithEmoji('success', 'loadFaceApiModels', 'Similarity model loaded successfully');
                    return true;
                } catch (error) {
                    logWithEmoji('warning', 'loadFaceApiModels', 'Failed to load similarity model: ' + error.message);
                    // Don't throw error for similarity model - allow graceful fallback
                    // Mark as loaded with fallback mode to prevent repeated loading attempts
                    modelStatus.myModel.loaded = true;
                    modelStatus.myModel.fallbackMode = true;
                    return true;
                }
            })();
            
            loadPromises.push(myModelPromise);
        }
        
        // Wait for all promises to complete
        await Promise.allSettled(loadPromises);
        
        // Check which models were actually loaded
        const failedModels = [];
        if (modelList.includes('tinyFaceDetector') && !modelStatus.faceApi.loaded) failedModels.push('tinyFaceDetector');
        if (modelList.includes('ssdMobilenetv1') && !modelStatus.faceApi.loaded) failedModels.push('ssdMobilenetv1');
        if (modelList.includes('faceNet') && !modelStatus.faceNet.loaded) failedModels.push('faceNet');
        if (modelList.includes('myModel') && !modelStatus.myModel.loaded && !modelStatus.myModel.fallbackMode) failedModels.push('myModel');
        
        if (failedModels.length > 0) {
            logWithEmoji('warning', 'loadFaceApiModels', `Some models failed to load: ${failedModels.join(', ')}`);
            
            // Only fail if critical models failed
            if (failedModels.includes('tinyFaceDetector') || failedModels.includes('faceNet')) {
                throw new Error(`Critical models failed to load: ${failedModels.join(', ')}`);
            } else {
                logWithEmoji('info', 'loadFaceApiModels', 'Non-critical models failed to load. Continuing with limited functionality.');
                // We'll continue with limited functionality
            }
        }
        
        logWithEmoji('success', 'loadFaceApiModels', 'Critical models loaded successfully');
        return true;
        
    } catch (error) {
        logWithEmoji('error', 'loadFaceApiModels', 'Error loading models: ' + error.message);
        throw error;
    } finally {
        // Always release the lock and clear the timeout
        isLoadingModels = false;
        logWithEmoji('unlock', 'loadFaceApiModels', 'Released model loading lock');
        if (modelLoadLockTimeout) {
            clearTimeout(modelLoadLockTimeout);
            modelLoadLockTimeout = null;
        }
    }
}

/**
 * Implements progressive model loading strategy for better performance
 * First loads critical models synchronously, then loads optional models asynchronously
 */
async function loadModelsOnDemand() {
    logFunctionEntry('loadModelsOnDemand');
    
    const startTime = performance.now();
    logWithEmoji('loading', 'loadModelsOnDemand', 'Starting progressive model loading');
    
    try {
        // 1. First load only critical face detection model synchronously
        logWithEmoji('model', 'loadModelsOnDemand', 'Loading critical model: tinyFaceDetector', 
            { priority: 'high', purpose: 'face detection' });
        
        const detectorStartTime = performance.now();
        await loadFaceApiModels(['tinyFaceDetector']);
        const detectorLoadTime = Math.round(performance.now() - detectorStartTime);
        
        logWithEmoji('success', 'loadModelsOnDemand', 
            `Loaded face detector model (${detectorLoadTime}ms)`, 
            { model: 'tinyFaceDetector', loadTimeMs: detectorLoadTime });
        
        // 2. Then load FaceNet model which is needed for embeddings
        logWithEmoji('model', 'loadModelsOnDemand', 'Loading critical model: faceNet', 
            { priority: 'high', purpose: 'face embeddings' });
        
        const faceNetStartTime = performance.now();
        await loadFaceApiModels(['faceNet']);
        const faceNetLoadTime = Math.round(performance.now() - faceNetStartTime);
        
        logWithEmoji('success', 'loadModelsOnDemand', 
            `Loaded faceNet model (${faceNetLoadTime}ms)`, 
            { model: 'faceNet', loadTimeMs: faceNetLoadTime });
        
        // 3. Load non-critical models asynchronously
        setTimeout(() => {
            logWithEmoji('model', 'loadModelsOnDemand', 'Loading optional models in background', 
                { priority: 'low', purpose: 'similarity comparison' });
                
            loadFaceApiModels(['myModel']).catch(error => {
                // Just log errors for optional models
                logWithEmoji('warning', 'loadModelsOnDemand', 
                    `Optional model loading failed: ${error.message}. Some features will be limited.`, 
                    { error, fallback: 'Using reduced feature set' });
                
                // Mark the model as in fallback mode to prevent repeated loading attempts
                if (error.message.includes('myModel') || error.message.includes('Similarity')) {
                    modelStatus.myModel.fallbackMode = true;
                }
            });
        }, 2000);
        
        const totalLoadTime = Math.round(performance.now() - startTime);
        logWithEmoji('success', 'loadModelsOnDemand', 
            `Critical models loaded successfully (${totalLoadTime}ms)`, 
            { criticalModels: ['tinyFaceDetector', 'faceNet'], totalTimeMs: totalLoadTime });
            
        return true;
    } catch (error) {
        const failTime = Math.round(performance.now() - startTime);
        logWithEmoji('error', 'loadModelsOnDemand', 
            `Model loading failed after ${failTime}ms: ${error.message}`, 
            { error, timeMs: failTime });
        throw error;
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
    maxAge: 3 * 60 * 1000, // Keep items for 180 seconds (3 minutes)

    
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
        
        // // Log processing completion
        // console.log(`Image processing complete: ${src}`, newInfo);
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
    },
    
    normalizeImageUrl(url) {
        // This method should be implemented to normalize image URLs
        // For example, you can use a URL normalization library or custom logic
        // Return a normalized URL string
        return url;
    }
};

/**
 * Checks if an element is a social media profile picture
 * @param {Element} element - The element to check
 * @returns {boolean} True if element is likely a profile picture
 */
function isProfilePicture(element) {
    if (!element) return false;

    // Check based on classes and attributes
    const isProfileByClass = element.classList && (
        element.classList.contains('profile-picture') ||
        element.classList.contains('avatar') ||
        element.classList.contains('user-photo') ||
        element.classList.contains('user-image') ||
        element.classList.contains('profile-img') ||
        element.classList.contains('user-img') ||
        element.classList.contains('x1lq5wgf') || // Facebook profile photo class
        element.classList.contains('xgqcy7u') ||
        element.classList.contains('x30kzoy')
    );

    // Check based on parent element attribute or class
    const parentHasProfileIndicator = element.parentElement && (
        element.parentElement.classList.contains('profile-pic-container') ||
        element.parentElement.classList.contains('avatar-container') ||
        element.parentElement.getAttribute('data-testid') === 'user-avatar' ||
        element.parentElement.getAttribute('aria-label')?.toLowerCase().includes('profile')
    );

    // Check if image has specific dimensions (common for profile pics)
    const hasProfileDimensions = (element.width === element.height) && 
                               (element.width === 36 || element.width === 40 || 
                                element.width === 48 || element.width === 64);
    
    // Check alt text
    const altTextIsProfile = element.alt && (
        element.alt.toLowerCase().includes('profile') ||
        element.alt.toLowerCase().includes('avatar') ||
        element.alt.toLowerCase().includes('user')
    );

    return isProfileByClass || parentHasProfileIndicator || hasProfileDimensions || altTextIsProfile;
}

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

    // Check if it's a profile picture - relax size requirements
    const isProfile = isProfilePicture(element);

    // For profile pictures, only check if hidden or processed
    if (isProfile) {
        const isHidden = element.offsetParent === null || 
                      window.getComputedStyle(element).display === 'none' ||
                      window.getComputedStyle(element).visibility === 'hidden';
        const isProcessed = element.closest('.face-detection-wrapper');
        
        return !isHidden && !isProcessed;
    }

    // For non-profile images, do regular checks
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

// Add new function to handle tiny image processing
/**
 * Process tiny images (below 64px in either dimension) with specialized upscaling
 * This function is specifically designed to handle small profile pictures, icons, and thumbnails
 * that would otherwise be skipped by the face detection system.
 * 
 * Key techniques:
 * 1. Intelligent upscaling - Uses a dynamic scale factor based on original image size
 * 2. Ensures dimensions are multiples of 32 (required by TinyYOLOv2 backbone in TinyFaceDetector)
 * 3. High-quality upscaling with improved image smoothing for better detail preservation
 * 4. Returns both the processed canvas and the scale factor for mapping detections back to original size
 * 
 * Designed to work with Facebook profile pictures, avatars, and other small face images down to 32px
 * 
 * @param {HTMLImageElement} img - The small image element to process
 * @returns {Object} Object containing the scaled canvas and scale factors
 */
async function processTinyImage(img) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Calculate dimensions that are multiples of 32
    const minSize = 32; // Absolute minimum size required by TinyFaceDetector
    
    // Dynamic scale factor calculation - more aggressive for smaller images
    // For very tiny images (e.g. 32px), this could be 2-3x scaling
    // For images near 64px, this might be 1.5x scaling
    const scaleFactor = Math.max(2, Math.ceil(32 / Math.min(img.width, img.height)));
    
    // Round dimensions up to nearest multiple of 32 for optimal model performance
    const targetWidth = roundToMultipleOf32(Math.ceil(img.width * scaleFactor));
    const targetHeight = roundToMultipleOf32(Math.ceil(img.height * scaleFactor));
    
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    
    // Use better upscaling algorithm - crucial for preserving facial details
    // This significantly improves detection quality on small images
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
    
    logWithEmoji('info', 'processTinyImage', `Upscaled small image from ${img.width}x${img.height} to ${targetWidth}x${targetHeight} (${scaleFactor}x scaling)`);
    
    return {
        canvas,
        scaleFactor: {
            x: targetWidth / img.width,
            y: targetHeight / img.height
        }
    };
}

/**
 * Select the appropriate face detection model based on image size and characteristics
 * 
 * This function implements an intelligent selection strategy:
 * - For tiny images (32-64px): Uses TinyFaceDetector with specialized settings
 * - For small images (64-128px): Uses TinyFaceDetector with optimized settings
 * - For medium images: Uses either model based on aspect ratio and quality
 * - For large images (256px+): Uses SSD MobileNet for best accuracy
 * 
 * Facebook profile pictures, avatars and thumbnails are handled by the tiny/small
 * image path, with optimized settings for detecting faces in those challenging cases.
 * 
 * @param {HTMLImageElement} img - The image element to analyze
 * @returns {Object} Selected model and its configuration options
 * @throws {Error} If the image is smaller than the absolute minimum size (32px)
 */
function selectFaceDetectionModel(img) {
    const width = img.width || img.naturalWidth;
    const height = img.height || img.naturalHeight;
    const minDimension = Math.min(width, height);
    const maxDimension = Math.max(width, height);
    
    // Skip images smaller than absolute minimum size
    if (minDimension < 32) {
        throw new Error(`Image too small for face detection (${width}x${height}). Minimum size required: 32px`);
    }

    // For very small images (32-64px), use tinyFaceDetector with highly optimized settings
    // This is specifically designed for small profile pics and avatars (e.g. Facebook thumbnails)
    if (minDimension < 64) {
        // Ensure input size is divisible by 32 (required by TinyYolov2)
        const inputSize = 64; // Fixed size for stability with very small images
        logWithEmoji('info', 'selectFaceDetectionModel', `Using specialized tiny image mode for ${width}x${height} image`);
        return {
            model: 'tinyFaceDetector',
            options: new faceapi.TinyFaceDetectorOptions({
                ...FACE_API_DETECTION_OPTIONS.tinyFaceDetector,
                inputSize: inputSize,
                scoreThreshold: 0.01,  // Very lenient threshold for tiny images
                minFaceSize: 16, // Absolute minimum face size
                scaleFactor: 0.5  // More granular scale steps
            })
        };
    }
    
    // For small images (64-128px), use tinyFaceDetector with optimized settings
    // This handles most social media profile pictures and small thumbnails
    if (minDimension <= MODEL_SELECTION_THRESHOLDS.SMALL_IMAGE) {
        // Ensure input size is divisible by 32 (required by TinyYolov2)
        const inputSize = roundToMultipleOf32(Math.max(64, minDimension));
        logWithEmoji('info', 'selectFaceDetectionModel', `Using optimized small image mode for ${width}x${height} image`);
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
                inputSize: Math.min(640, minDimension)
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
                inputSize: Math.min(640, minDimension)
            })
        };
    }

    // Default to tinyFaceDetector for other cases with adaptive input size
    return {
        model: 'tinyFaceDetector',
        options: new faceapi.TinyFaceDetectorOptions({
            ...FACE_API_DETECTION_OPTIONS.tinyFaceDetector,
            // Ensure input size is divisible by 32 (required by TinyYolov2)
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
// This function is now imported from utils.js
// function detectImageRotation(imageData) {
//     // Simple heuristic: check if height is significantly larger than width
//     // This assumes portrait photos are more likely to need rotation
//     const aspectRatio = imageData.width / imageData.height;
//     return aspectRatio < 0.7; // Arbitrary threshold for portrait orientation
// }

// Add helper function to detect Facebook profile images
// This function is now imported from utils.js
// function isFacebookProfileImage(element) {
//     // Check if element is within Facebook's profile picture container
//     return element.closest('[data-visualcompletion="media-vc-image"]') !== null ||
//            element.closest('[data-type="profile_picture"]') !== null ||
//            element.closest('.profile-photo-container') !== null;
// }

// Update the detectFacesWithFaceApi function to use the helper
async function detectFacesWithFaceApi(img) {
    
    // Get wrapper and check if it exists
    const wrapper = findOrCreateWrapper(img);
    if (!wrapper) return false;
    
    // Get image source
    const src = img.src || img.getAttribute('xlink:href');
    
    // Check if source exists
    if (!src) return false;
    
    // Check if this image is already in our blur tracker
    if (blurTracker && blurTracker.shouldBlur(src)) {
        // Apply blur effect directly
        img.style.filter = 'blur(10px)';
        img.classList.add('blurred-image');
        img.setAttribute('data-faceone-processed', 'blurred');
        
        if (flagShowFrameonImage.addLabel) {
            addResultIndicator(wrapper, 'Image blurred - From saved list');
        }
        
        // Mark as processed but don't run further detection
        try {
            imageTracker.markProcessed(src, true, true);
            // Also mark in processedImageTracker
            processedImageTracker.mark(img, 'blurred');
        } catch (error) {
            console.error('Error marking image as processed:', error);
        }
        return;
    }
    
    // Check if this image was previously blurred but has expired
    if (blurTracker && blurTracker.wasBlurredBefore(src)) {
        console.log('Detected previously blurred image, automatically renewing:', src.substring(0, 50) + '...');
        
        // Automatically reactivate blur
        blurTracker.markForBlur(src);
        
        // Apply blur effect
        img.style.filter = 'blur(10px)';
        img.classList.add('blurred-image');
        img.setAttribute('data-faceone-processed', 'blurred');
        img.setAttribute('data-auto-renewed', 'true');
        
        if (flagShowFrameonImage.addLabel) {
            addResultIndicator(wrapper, 'Image auto-reblurred - Previously detected');
        }
        
        // Mark as processed but don't run further detection
        try {
            imageTracker.markProcessed(src, true, true);
            // Also mark in processedImageTracker
            processedImageTracker.mark(img, 'blurred');
        } catch (error) {
            console.error('Error marking image as processed:', error);
        }
        return;
    }
    
    // Continue with regular processing for new images...
    const processingKey = `processing_${Date.now()}`;
    wrapper.setAttribute('data-processing-key', processingKey);
    
    try {
        await Promise.all([
            ensureModelsLoaded(),
            loadPositiveEmbeddings()
        ]);
        
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
        const angles = [0, -45, 45, 90, -90]; 
        
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
                    img.classList.add('blurred-image');
                    img.setAttribute('data-faceone-processed', 'blurred');
                    
                    if (flagShowFrameonImage.addLabel) {
                        addResultIndicator(wrapper, 'Image blurred - Similar faces detected');
                    }
                    
                    // Store in blurTracker for persistence
                    if (blurTracker) {
                        blurTracker.markForBlur(src);
                        console.log('Added to blur list:', src.substring(0, 50) + '...');
                    }
                } else {
                    img.style.filter = 'none';
                    img.classList.remove('blurred-image');
                    img.setAttribute('data-faceone-processed', 'normal');
                    
                    if (flagShowFrameonImage.addLabel) {
                        addResultIndicator(wrapper, 'No matching faces detected');
                    }
                    
                    // Remove from blur tracker if it was previously blurred
                    if (blurTracker && blurTracker.shouldBlur(src)) {
                        blurTracker.unmarkForBlur(src);
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
                
                // Remove from blur tracker if it was previously blurred
                if (blurTracker && blurTracker.shouldBlur(src)) {
                    blurTracker.unmarkForBlur(src);
                }
            }
        }
        
        // Mark image as processed
        imageTracker.markProcessed(src, true);
        processedImageTracker.mark(img, 'complete');
        
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
    logFunctionEntry('addResultIndicator');
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
    logFunctionEntry('drawDetectionsWithSimilarity');
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
 * 
 * Note: This function is also available in utils.js (duplicated here for now)
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
 * 
 * This function is now imported from utils.js
 */
// function preventTextSelection(e) {
//   if (e.target.tagName === 'IMG') {
//     e.preventDefault();
//     window.getSelection().removeAllRanges();
//   }
// }

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
 * Ensures all required models are loaded
 * @returns {Promise<boolean>} Promise resolving to true when models are loaded
 */
async function ensureModelsLoaded() {
    logFunctionEntry('ensureModelsLoaded');
    logWithEmoji('loading', 'ensureModelsLoaded', 'Checking if all models are loaded');
    
    // Add a timeout to prevent hanging
    const timeout = 30000; // 30 seconds timeout
    const modelCheckStart = Date.now();
    
    // Define critical models (required for basic functionality)
    const criticalModels = {
        faceApi: true,    // We need at least one face detection model
        faceNet: true     // We need face embedding generation for basic functionality
    };
    
    // Define optional models (enhance functionality but not strictly required)
    const optionalModels = {
        myModel: true     // Similarity model enhances functionality but isn't critical
    };
    
    try {
        // Check which models need loading
        const modelsToLoad = [];
        
        if (!modelStatus.faceApi.loaded) modelsToLoad.push('tinyFaceDetector');
        if (!modelStatus.faceNet.loaded) modelsToLoad.push('faceNet');
        if (!modelStatus.myModel.loaded && !modelStatus.myModel.fallbackMode) modelsToLoad.push('myModel');
        
        if (modelsToLoad.length > 0) {
            logWithEmoji('loading', 'ensureModelsLoaded', `Models not loaded, loading now... Status: ${JSON.stringify(modelStatus)}`);
            
            try {
                // Attempt to load models
                await loadFaceApiModels(modelsToLoad);
                
                // Verify critical models are loaded
                let missingCritical = [];
                if (!modelStatus.faceApi.loaded && criticalModels.faceApi) missingCritical.push('FaceAPI');
                if (!modelStatus.faceNet.loaded && criticalModels.faceNet) missingCritical.push('FaceNet');
                
                if (missingCritical.length > 0) {
                    throw new Error(`Critical models failed to load: ${missingCritical.join(', ')}`);
                }
                
                // Check optional models - log warnings but don't fail
                let missingOptional = [];
                if (!modelStatus.myModel.loaded && !modelStatus.myModel.fallbackMode && optionalModels.myModel) {
                    missingOptional.push('Similarity Model');
                    
                    // Mark in fallback mode to prevent repeated loading attempts
                    modelStatus.myModel.fallbackMode = true;
                }
                
                if (missingOptional.length > 0) {
                    logWithEmoji('warning', 'ensureModelsLoaded', 
                        `Optional models not available: ${missingOptional.join(', ')}. Some features will be limited.`);
                }
                
                logWithEmoji('success', 'ensureModelsLoaded', 'Critical models loaded successfully');
                state.modelsLoaded = true;
            } catch (error) {
                logWithEmoji('error', 'ensureModelsLoaded', 'Error loading models: ' + error.message);
                
                // Check how long we've been trying
                if (Date.now() - modelCheckStart > timeout) {
                    logWithEmoji('error', 'ensureModelsLoaded', 'Model loading timeout');
                    throw new Error(`Model loading timeout after ${timeout}ms: ${error.message}`);
                }
                
                // Try to help with debugging by showing which models failed
                const failedModels = [];
                if (!modelStatus.faceApi.loaded && criticalModels.faceApi) failedModels.push('FaceAPI');
                if (!modelStatus.faceNet.loaded && criticalModels.faceNet) failedModels.push('FaceNet');
                
                throw new Error(`Failed to load models: ${failedModels.join(', ')}. Error: ${error.message}`);
            }
        } else {
            logWithEmoji('success', 'ensureModelsLoaded', 'All required models already loaded');
        }
    } catch (error) {
        // Preserve the error message but add more context for debugging
        logWithEmoji('error', 'ensureModelsLoaded', `Model loading failed: ${error.message}`);
        throw error;
    }
    
    // Final verification to ensure we're ready for processing
    const criticalModelsLoaded = 
        (modelStatus.faceApi.loaded || !criticalModels.faceApi) && 
        (modelStatus.faceNet.loaded || !criticalModels.faceNet);
    
    // Update state based on critical model availability
    state.modelsLoaded = criticalModelsLoaded;
    
    return criticalModelsLoaded;
}

/**
 * Handle a single visible element for processing
 * @param {HTMLElement} element - The element to process
 * @returns {Promise<void>}
 */
async function handleVisibleElement(element) {
    logFunctionEntry('handleVisibleElement');
    logWithEmoji('image', 'handleVisibleElement', 'Processing visible image element');
    
    // Store the original element properties for recovery if needed
    const originalDisplay = element.style.display;
    const originalVisibility = element.style.visibility;
    const originalOpacity = element.style.opacity;
    let wrapper = null;
    
    try {
        // Skip if already processed or being processed
        if (processedImageTracker.isProcessed(element)) {
            logWithEmoji('skip', 'handleVisibleElement', 'Image already processed or being processed');
            return;
        }
        
        // Mark as pending to prevent double-processing
        processedImageTracker.mark(element, 'pending');
        
        // Create the detection wrapper first
        wrapper = createWrapper(element);
        
        // Only proceed if image meets size requirements
        const width = element.width || element.naturalWidth;
        const height = element.height || element.naturalHeight;
        
        // Skip only if image is smaller than absolute minimum (32px)
        // This allows processing of Facebook profile pics and other small avatars (which are typically 40-50px)
        // but skips icons, emojis, and decorative elements that are too small to contain meaningful faces
        if (width < 32 || height < 32) {
            logWithEmoji('info', 'handleVisibleElement', `Image too small for processing: ${width}x${height}. Minimum required: 32px.`);
            logWithEmoji('info', 'handleVisibleElement', `Likely an icon, emoji, or decoration - not a profile picture.`);
            processedImageTracker.mark(element, 'skipped');
            
            // Instead of removing the wrapper, just clear it of detection-related elements
            if (wrapper && wrapper.isConnected) {
                const canvas = wrapper.querySelector('.face-detection-canvas');
                const indicator = wrapper.querySelector('.processing-indicator');
                if (canvas) canvas.remove();
                if (indicator) indicator.remove();
                
                // Don't call cleanupWrapper - instead ensure image is visible
                element.style.display = originalDisplay;
                element.style.visibility = 'visible';
                element.style.opacity = '1';
            }
            return;
        }
        
        // Even if smaller than user setting but larger than 32px, we'll still process it
        // This allows processing of profile pictures even if the user has set a larger minimum size
        if (width < flagShowFrameonImage.minimumImageSize || height < flagShowFrameonImage.minimumImageSize) {
            logWithEmoji('info', 'handleVisibleElement', `Processing small image: ${width}x${height} (below user setting of ${flagShowFrameonImage.minimumImageSize}px)`);
            logWithEmoji('info', 'handleVisibleElement', `Small profile pictures and avatars will be processed with specialized techniques`);
        }
        
        try {
            // Ensure models are loaded but continue with limited functionality if only optional models fail
            await ensureModelsLoaded();
            
            // Load positive embeddings if similarity model is available
            if (modelStatus.myModel.loaded && !modelStatus.myModel.fallbackMode) {
                await loadPositiveEmbeddings();
            }
            
            // Process with face detection - this now handles its own errors and ensures image visibility
            await detectFacesWithFaceApi(element);
            
            // Mark as complete
            processedImageTracker.mark(element, 'complete');
            
        } catch (error) {
            logWithEmoji('error', 'handleVisibleElement', `Failed to load models for image processing: ${error.message}`);
            
            // Check if we can operate in fallback mode with limited features
            const hasCriticalModels = modelStatus.faceApi.loaded && modelStatus.faceNet.loaded;
            
            if (hasCriticalModels) {
                logWithEmoji('info', 'handleVisibleElement', 'Operating in fallback mode with limited functionality');
                
                try {
                    // Try to perform basic face detection without similarity checks
                    await performBasicFaceDetection(element, wrapper);
                    processedImageTracker.mark(element, 'complete');
                } catch (fallbackError) {
                    logWithEmoji('error', 'handleVisibleElement', `Fallback processing also failed: ${fallbackError.message}`);
                    
                    // IMPORTANT: Don't cleanup wrapper, just ensure image is visible
                    if (element) {
                        element.style.display = originalDisplay || '';
                        element.style.visibility = 'visible';
                        element.style.opacity = '1';
                    }
                    
                    processedImageTracker.mark(element, 'failed');
                }
            } else {
                // Critical models not available, ensure image visibility
                if (element) {
                    element.style.display = originalDisplay || '';
                    element.style.visibility = 'visible';
                    element.style.opacity = '1';
                }
                
                // Don't call cleanupWrapper as it might remove the image
                processedImageTracker.mark(element, 'failed');
            }
        }
    } catch (error) {
        logWithEmoji('error', 'handleVisibleElement', `Error processing element: ${error.message}`);
        
        // CRITICAL: Always ensure image visibility on errors
        if (element) {
            element.style.display = originalDisplay || '';
            element.style.visibility = 'visible';
            element.style.opacity = '1';
        }
        
        processedImageTracker.mark(element, 'failed');
    }
}

/**
 * Perform basic face detection without similarity matching
 * Used as a fallback when similarity model is not available
 * @param {HTMLElement} element - The image element to process
 * @param {HTMLElement} wrapper - The detection wrapper element
 */
async function performBasicFaceDetection(element, wrapper) {
    logFunctionEntry('performBasicFaceDetection');
    logWithEmoji('image', 'performBasicFaceDetection', 'Performing basic face detection without similarity checks');
    
    try {
        // Use image preprocessing to normalize rotation
        const proxyImg = await createProxyImage(element);
        const normalizedImg = await normalizeImageRotation(proxyImg);
        
        // Create scaled version for better detection
        const scaledCanvas = await createScaledImage(normalizedImg);
        const scaleFactor = scaledCanvas.scaleFactor || 1;
        
        // Get appropriate model and options
        const { model, options } = selectFaceDetectionModel(scaledCanvas);
        
        // Detect faces
        const detections = await (model === 'ssdMobilenetv1' 
            ? faceapi.detectAllFaces(scaledCanvas, options)
            : faceapi.detectAllFaces(scaledCanvas, options));
        
        // Process detections
        if (detections.length > 0) {
            // Scale back detections if we upscaled
            const scaledDetections = scaleFactor !== 1 
                ? scaleDetections(detections, scaleFactor) 
                : detections;
            
            // Remove duplicates
            const uniqueDetections = removeDuplicateDetections(scaledDetections);
            
            // Draw the detections without similarity information
            const canvas = createDetectionCanvas(element);
            if (canvas) {
                drawDetections(canvas, uniqueDetections, element);
                wrapper.appendChild(canvas);
            }
            
            // Add indicator
            if (flagShowFrameonImage.addLabel) {
                addResultIndicator(wrapper, `${uniqueDetections.length} face(s) detected (fallback mode)`);
            }
            
            logWithEmoji('success', 'performBasicFaceDetection', `Detected ${uniqueDetections.length} faces in fallback mode`);
        } else {
            // No faces detected
            if (flagShowFrameonImage.addLabel) {
                addResultIndicator(wrapper, 'No faces detected (fallback mode)');
            }
            logWithEmoji('info', 'performBasicFaceDetection', 'No faces detected in fallback mode');
        }
    } catch (error) {
        logWithEmoji('error', 'performBasicFaceDetection', `Fallback detection error: ${error.message}`);
        throw error;
    }
}

// Add scaleDetections function
function scaleDetections(detections, scaleFactor) {
    logFunctionEntry('scaleDetections');
    logWithEmoji('image', 'scaleDetections', 'Scaling detection coordinates');
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
    logFunctionEntry('observeElements');
    logWithEmoji('search', 'observeElements', 'Setting up observation of image elements');
    if (!state.modelsLoaded) {
        logWithEmoji('warning', 'observeElements', 'Cannot start observation: models not loaded');
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
    logFunctionEntry('startDocumentObserver');
    logWithEmoji('setup', 'startDocumentObserver', 'Starting document observer');
    
    // Create document observer instance
    const documentObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        // Query for all images
                        const images = node.querySelectorAll('img:not(.face-detection-canvas)');
                        
                        images.forEach((img) => {
                            if (isValidElement(img) && 
                                !img.closest('.face-detection-wrapper') && 
                                !processedImageTracker.isProcessed(img)) {
                                // Add to queue for processing
                                imageQueue.add(img);
                            }
                        });
                        
                        // Handle SVG images separately (they may be nested)
                        const svgImages = node.querySelectorAll('image[xlink\\:href]');
                        svgImages.forEach((img) => {
                            if (isValidElement(img) && 
                                !img.closest('.face-detection-wrapper') && 
                                !processedImageTracker.isProcessed(img)) {
                                // Add to queue for processing
                                imageQueue.add(img);
                            }
                        });
                    }
                });
            } else if (mutation.type === 'attributes') {
                // Handle attribute changes (like image src) only if the attribute
                // being changed is relevant to image content
                if (mutation.attributeName === 'src' || mutation.attributeName === 'xlink:href') {
                    const target = mutation.target;
                    if ((target.tagName === 'IMG' || target.tagName === 'image') && 
                        isValidElement(target) && 
                        !target.closest('.face-detection-wrapper')) {
                        
                        // Reset processed status when src changes
                        if (processedImageTracker.isProcessed(target)) {
                            processedImageTracker.processed.delete(processedImageTracker.getImageIdentifier(target));
                        }
                        
                        // Add to queue for processing
                        imageQueue.add(target);
                    }
                }
            }
        });
    });
    
    // Start observing document
    documentObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['src', 'xlink:href']
    });
    
    logWithEmoji('success', 'startDocumentObserver', 'Document observer started successfully');
    
    return documentObserver;
}

// Add initialization helper functions
async function initializeExtensionContext() {
    logFunctionEntry('initializeExtensionContext');
    logWithEmoji('loading', 'initializeExtensionContext', 'Initializing extension context');
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

// Store initialization attempts in session storage to prevent infinite loops
const MAX_INIT_ATTEMPTS = 3;
// These functions are now imported from utils.js
// function getInitAttempts() {
//     logFunctionEntry('getInitAttempts');
//     logWithEmoji('info', 'getInitAttempts', 'Getting initialization attempts count');
//     const attempts = sessionStorage.getItem('initAttempts') || 0;
//     return parseInt(attempts, 10);
// }
// 
// function incrementInitAttempts() {
//     logFunctionEntry('incrementInitAttempts');
//     logWithEmoji('info', 'incrementInitAttempts', 'Incrementing initialization attempts count');
//     const attempts = getInitAttempts() + 1;
//     sessionStorage.setItem('initAttempts', attempts);
//     return attempts;
// }
// 
// function resetInitAttempts() {
//     logFunctionEntry('resetInitAttempts');
//     logWithEmoji('info', 'resetInitAttempts', 'Resetting initialization attempts count');
//     sessionStorage.removeItem('initAttempts');
// }

// Add a helper function for logging with emojis
// ... existing code ...

// Use the new logging function in key places
window.addEventListener('load', async () => {
    logWithEmoji('start', 'init', 'Initializing FaceOne extension...');
    const initStartTime = performance.now(); // Add performance tracking
    
    // OPTIMIZATION: Preload TensorFlow before initializing models
    let preloadSandboxTime = 0;
    
    try {
        // Check for too many initialization attempts, but don't refresh
        const attempts = getInitAttempts();
        if (attempts >= MAX_INIT_ATTEMPTS) {
            logWithEmoji('error', 'init', `Maximum initialization attempts (${MAX_INIT_ATTEMPTS}) exceeded, stopping initialization`);
            clearModelStatus();
            resetInitAttempts();
            return; // Stop trying to initialize
        }
        
        // OPTIMIZATION: Start preloading sandbox frame immediately
        const preloadStartTime = performance.now();
        logWithEmoji('loading', 'init', 'Preloading sandbox frame and extension context in parallel');
        
        // OPTIMIZATION: Load extension context and sandbox frame in parallel
        const extensionContextPromise = initializeExtensionContext().then(() => {
            logWithEmoji('success', 'init', 'Extension context initialized');
        });
        
        const sandboxFramePromise = createSandboxFrame().then(() => {
            logWithEmoji('success', 'init', 'Sandbox frame created');
            // No additional wait - continue immediately
        });
        
        // Wait for both to complete
        await Promise.all([extensionContextPromise, sandboxFramePromise]);
        preloadSandboxTime = performance.now() - preloadStartTime;
        logWithEmoji('timer', 'init', `Preloading completed in ${Math.round(preloadSandboxTime)}ms`);
        
        // OPTIMIZATION: Start model loading with priority order
        logWithEmoji('model', 'init', 'Starting model loading sequence...');
        const modelLoadStartTime = performance.now();
        await loadFaceApiModels();
        const modelLoadTime = performance.now() - modelLoadStartTime;
        logWithEmoji('timer', 'init', `Model loading completed in ${Math.round(modelLoadTime)}ms`);
        
        // Verify all models are actually loaded
        if (!modelStatus.faceApi.loaded || !modelStatus.faceNet.loaded || !modelStatus.myModel.loaded) {
            throw new Error('Models failed to load properly. Status: ' + JSON.stringify(modelStatus));
        }
        
        // Reset init attempts on success
        resetInitAttempts();
        logWithEmoji('success', 'init', 'All models loaded successfully');
        
        // Start processing if auto-processing is enabled
        if (flagShowFrameonImage.autoProcessImages) {
            // OPTIMIZATION: Load positive embeddings in parallel with image processing
            logWithEmoji('loading', 'init', 'Loading positive embeddings in background');
            const embeddingsPromise = loadPositiveEmbeddings().catch(error => {
                logWithEmoji('warning', 'init', 'Error loading positive embeddings: ' + error.message);
                // Non-fatal error, continue with processing
            });
            
            logWithEmoji('search', 'init', 'Auto-processing enabled, starting image processing...');
            const processingStartTime = performance.now();
            await processExistingImages();
            observeElements();
            const processingTime = performance.now() - processingStartTime;
            logWithEmoji('timer', 'init', `Image processing completed in ${Math.round(processingTime)}ms`);
            
            // Wait for embeddings to finish loading
            await embeddingsPromise;
        } else {
            // Load embeddings sequentially if not processing images
            logWithEmoji('loading', 'init', 'Loading positive embeddings');
            await loadPositiveEmbeddings();
        }
        
        const totalInitTime = performance.now() - initStartTime;
        logWithEmoji('success', 'init', `FaceOne initialization completed in ${Math.round(totalInitTime)}ms`);
        
    } catch (error) {
        const failTime = performance.now() - initStartTime;
        logWithEmoji('error', 'init', `Initialization error after ${Math.round(failTime)}ms: ${error.message}`);
        logWithEmoji('error', 'init', 'Model status at error: ' + JSON.stringify(modelStatus, null, 2));
        
        // Clean up without refreshing
        if (sandboxFrame) {
            try {
                document.body.removeChild(sandboxFrame);
            } catch (e) {
                logWithEmoji('warning', 'init', 'Error removing sandbox frame: ' + e.message);
            }
            sandboxFrame = null;
        }
        
        clearModelStatus();
        
        // Record attempt
        const currentAttempts = incrementInitAttempts();
        logWithEmoji('warning', 'init', `Initialization attempt ${currentAttempts}/${MAX_INIT_ATTEMPTS} failed.`);
        
        // Instead of page refresh, retry initialization after a delay
        if (currentAttempts < MAX_INIT_ATTEMPTS) {
            const retryDelay = 3000; // Reduced from 5 seconds to 3 seconds
            logWithEmoji('loading', 'init', `Will retry initialization in ${retryDelay/1000} seconds...`);
            setTimeout(() => {
                // OPTIMIZATION: Recreate the sandbox and try loading models with better error handling
                logWithEmoji('loading', 'init', 'Retrying initialization');
                const retryStartTime = performance.now();
                
                createSandboxFrame()
                    .then(() => {
                        logWithEmoji('success', 'init', 'Delayed sandbox frame creation successful');
                        return loadFaceApiModels();
                    })
                    .then(() => {
                        logWithEmoji('success', 'init', 'Delayed model loading successful');
                        const retryTime = performance.now() - retryStartTime;
                        logWithEmoji('timer', 'init', `Retry initialization completed in ${Math.round(retryTime)}ms`);
                        
                        if (flagShowFrameonImage.autoProcessImages) {
                            return processExistingImages()
                                .then(() => {
                                    logWithEmoji('success', 'init', 'Delayed image processing successful');
                                    observeElements();
                                })
                                .catch(e => logWithEmoji('error', 'init', 'Error in delayed processing: ' + e.message));
                        }
                    })
                    .catch(e => logWithEmoji('error', 'init', 'Error in delayed initialization: ' + e.message));
            }, retryDelay);
        } else {
            logWithEmoji('error', 'init', `Maximum initialization attempts reached. Extension will not initialize.`);
            resetInitAttempts(); // Reset for next page load
        }
    }
});

// Add function to ensure TensorFlow is ready
async function ensureTensorFlowReady() {
    logFunctionEntry('ensureTensorFlowReady');
    logWithEmoji('loading', 'ensureTensorFlowReady', 'Ensuring TensorFlow is properly initialized');
    if (!sandboxFrame || !sandboxFrame.contentWindow) {
        throw new Error('Sandbox frame not available');
    }

    // OPTIMIZATION: Use a cached status check with a short timeout
    const statusCheckPromise = checkTensorFlowStatus();
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('TensorFlow status check timeout')), 3000);
    });
    
    try {
        const status = await Promise.race([statusCheckPromise, timeoutPromise]);
        if (!status.isInitialized || !status.tfBackendInitialized) {
            throw new Error('TensorFlow not properly initialized');
        }
        logWithEmoji('success', 'ensureTensorFlowReady', 'TensorFlow is properly initialized');
        return true;
    } catch (error) {
        logWithEmoji('error', 'ensureTensorFlowReady', 'TensorFlow initialization check failed: ' + error.message);
        throw error;
    }
}

/**
 * Process all images that already exist on the page
 * @returns {Promise<void>}
 */
async function processExistingImages() {
    logFunctionEntry('processExistingImages');
    logWithEmoji('image', 'processExistingImages', 'Processing existing images on page');
    
    try {
        // Ensure at least the critical models are loaded
        const requiredModels = ['tinyFaceDetector', 'faceNet'];
        
        try {
            logWithEmoji('loading', 'processExistingImages', 'Models not loaded, loading now...');
            await loadFaceApiModels(requiredModels);
        } catch (error) {
            logWithEmoji('error', 'processExistingImages', `Failed to load required models: ${error.message}`);
            return; // Can't proceed without critical models
        }
        
        // Find all images on the page
        logWithEmoji('search', 'processExistingImages', 'Scanning page for images...');
        
        let images = [];
        // Look for all images, not just those without data-faceone-processed attribute
        const imgElements = document.querySelectorAll('img[src]');
        
        // Add SVG images
        const svgImages = document.querySelectorAll('image[xlink\\:href]');
        
        // Collect valid elements that haven't been processed or are pending
        for (const img of imgElements) {
            if (isValidElement(img) && !processedImageTracker.isProcessed(img)) {
                images.push(img);
            }
        }
        
        for (const img of svgImages) {
            if (isValidElement(img) && !processedImageTracker.isProcessed(img)) {
                images.push(img);
            }
        }
        
        logWithEmoji('info', 'processExistingImages', `Found ${images.length} valid images to process`);
        
        // Process images in batches to avoid freezing the UI
        const batchSize = 5;
        const processBatch = async (startIndex) => {
            const endIndex = Math.min(startIndex + batchSize, images.length);
            const batch = images.slice(startIndex, endIndex);
            
            for (const img of batch) {
                try {
                    // No need to set attribute here, the tracker will handle it
                    // Process the image
                    await handleVisibleElement(img);
                } catch (error) {
                    logWithEmoji('error', 'processExistingImages', `Error processing image: ${error.message}`);
                    // Mark as failed in case of error
                    processedImageTracker.mark(img, 'failed');
                }
            }
            
            // Process next batch if there are more images
            if (endIndex < images.length) {
                setTimeout(() => {
                    processBatch(endIndex);
                }, 100);
            } else {
                logWithEmoji('success', 'processExistingImages', 'Finished processing existing images');
            }
        };
        
        // Start processing with the first batch
        if (images.length > 0) {
            processBatch(0);
        } else {
            logWithEmoji('info', 'processExistingImages', 'No valid images found to process');
        }
    } catch (error) {
        logWithEmoji('error', 'processExistingImages', `Error processing images: ${error.message}`);
    }
}

// Modify generateEmbedding to ensure model readiness
async function generateEmbedding(faceCanvas) {
    logFunctionEntry('generateEmbedding');
    logWithEmoji('model', 'generateEmbedding', 'Generating face embedding from canvas');
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
    logFunctionEntry('loadMyModel');
    logWithEmoji('model', 'loadMyModel', 'Loading similarity model');
    if (modelStatus.myModel.loaded) {
        logWithEmoji('success', 'loadMyModel', 'Similarity model already loaded');
        return;
    }
    
    const modelPath = chrome.runtime.getURL('models/myModel/tfjs_graph_model/model.json');
    return new Promise((resolve, reject) => {
        const handleMessage = (event) => {
            if (event.data.type === 'MODEL_LOADED' && event.data.modelName === 'myModel') {
                window.removeEventListener('message', handleMessage);
                if (event.data.success) {
                    modelStatus.myModel.loaded = true;
                    console.log('Similarity model loaded successfully');
                    resolve();
                } else {
                    reject(new Error(event.data.error || 'Similarity model loading failed'));
                }
            }
        };
        
        window.addEventListener('message', handleMessage);
        sandboxFrame.contentWindow.postMessage({
            type: 'LOAD_MODEL',
            modelName: 'myModel',
            modelPath: modelPath,
            waitForWarmup: true
        }, '*');
        
        setTimeout(() => {
            window.removeEventListener('message', handleMessage);
            reject(new Error('Similarity model load timeout'));
        }, 30000);
    });
}

// Add function to run inference with your model
async function runModelInference(modelName, inputData, inputShape) {
    logFunctionEntry('runModelInference');
    logWithEmoji('model', 'runModelInference', `Running inference with ${modelName} model`);
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
    logFunctionEntry('computeFaceSimilarity');
    logWithEmoji('model', 'computeFaceSimilarity', 'Computing similarity between face embeddings');
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

// Improve retry model load function to not refresh the page
async function retryModelLoad(loadFunction, maxAttempts = 3, delayMs = 1000) {
    logFunctionEntry('retryModelLoad');
    logWithEmoji('loading', 'retryModelLoad', `Attempting to load model with ${maxAttempts} retries`);
    let lastError = null;
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        console.log(`Attempt ${attempt}/${maxAttempts} to load model...`);
        
        try {
            const result = await loadFunction();
            if (result === false) {
                console.warn(`Load attempt ${attempt} returned false, retrying...`);
                if (attempt < maxAttempts) {
                    await new Promise(resolve => setTimeout(resolve, delayMs));
                    continue;
                } else {
                    throw new Error('Maximum load attempts reached with unsuccessful results');
                }
            }
            console.log(`Model load attempt ${attempt} succeeded`);
            return result;
        } catch (error) {
            console.error(`Error in load attempt ${attempt}:`, error);
            lastError = error;
            
            if (attempt < maxAttempts) {
                console.log(`Retrying after ${delayMs}ms...`);
                await new Promise(resolve => setTimeout(resolve, delayMs));
            }
        }
    }
    
    throw new Error(`Failed after ${maxAttempts} attempts. Last error: ${lastError?.message || 'Unknown error'}`);
}

// Modify loadTFModel to use retry mechanism with longer timeout
async function loadTFModel(modelName, modelPath) {
    logFunctionEntry('loadTFModel');
    logWithEmoji('model', 'loadTFModel', `Loading TensorFlow model ${modelName} from ${modelPath}`);
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
    logFunctionEntry('clearModelStatus');
    logWithEmoji('info', 'clearModelStatus', 'Resetting model loading status');
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
    logFunctionEntry('createScaledImage');
    logWithEmoji('image', 'createScaledImage', 'Creating scaled version of image');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Get original dimensions
    const originalWidth = img.width || img.naturalWidth;
    const originalHeight = img.height || img.naturalHeight;
    
    // Calculate minimum required size - for profile images, ensure at least 64px
    const minRequiredSize = 64; // Minimum size for reliable face detection
    const smallestDimension = Math.min(originalWidth, originalHeight);
    
    let targetWidth = originalWidth;
    let targetHeight = originalHeight;
    let scaleFactor = 1;
    
    // Scale up if image is too small
    if (smallestDimension < minRequiredSize) {
        // For very small images, use a more aggressive scaling factor
        if (smallestDimension < 40) {
            scaleFactor = Math.ceil(minRequiredSize / smallestDimension) * 2; // More aggressive for tiny images
        } else {
            scaleFactor = Math.ceil(minRequiredSize / smallestDimension);
        }
        
        targetWidth = Math.round(originalWidth * scaleFactor);
        targetHeight = Math.round(originalHeight * scaleFactor);
        
        // Ensure dimensions are multiples of 32 for better model performance
        targetWidth = roundToMultipleOf32(targetWidth);
        targetHeight = roundToMultipleOf32(targetHeight);
        
        logWithEmoji('info', 'createScaledImage', `Scaling small image up by ${scaleFactor}x: ${originalWidth}x${originalHeight} → ${targetWidth}x${targetHeight}`);
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
    logFunctionEntry('drawDetections');
    logWithEmoji('draw', 'drawDetections', 'Drawing face detection rectangles on canvas');
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
    logFunctionEntry('addResultIndicator');
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
    logFunctionEntry('extractFaceRegion');
    logWithEmoji('image', 'extractFaceRegion', 'Extracting face region from image');
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
    logFunctionEntry('checkModelStatus');
    
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
    
    const allLoaded = status.faceApi && status.faceNet && status.myModel;
    const essentialLoaded = status.faceApi && status.faceNet;
    
    if (allLoaded) {
        logWithEmoji('success', 'checkModelStatus', 'All models loaded successfully', status);
    } else if (essentialLoaded) {
        logWithEmoji('info', 'checkModelStatus', 'Essential models loaded, optional models pending', status);
    } else if (status.errors.length > 0) {
        logWithEmoji('warning', 'checkModelStatus', 
            `Model loading incomplete: ${status.errors.length} errors found`,
            { status, errors: status.errors });
    } else {
        logWithEmoji('info', 'checkModelStatus', 'Models still loading', status);
    }
    
    return status;
}

// Add helper function to check model loading status
function getModelLoadingStatus() {
    logFunctionEntry('getModelLoadingStatus');
    logWithEmoji('info', 'getModelLoadingStatus', 'Getting detailed model loading status');
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
    logFunctionEntry('checkTensorFlowStatus');
    logWithEmoji('info', 'checkTensorFlowStatus', 'Checking TensorFlow initialization status');
    if (!sandboxFrame || !sandboxFrame.contentWindow) {
        return { initialized: false, error: 'No sandbox frame' };
    }

    try {
        // Send message and wait for response with timeout
        sandboxFrame.contentWindow.postMessage({ type: 'GET_TF_STATUS' }, '*');
        
        const result = await new Promise((resolve, reject) => {
            const messageHandler = (event) => {
                if (event.data && event.data.type === 'TF_STATUS') {
                    cleanup();
                    resolve(event.data.status);
                }
            };
            
            const timeoutId = setTimeout(() => {
                cleanup();
                reject(new Error('Status check timeout'));
            }, 3000);
            
            const cleanup = () => {
                clearTimeout(timeoutId);
                window.removeEventListener('message', messageHandler);
            };
            
            window.addEventListener('message', messageHandler);
        });
        
        return result;
    } catch (error) {
        logError('checkTensorFlowStatus', 'Error checking TF status:', error);
        return { initialized: false, error: error.message || 'Status check failed' };
    }
}

// Add function to compute similarities when needed
async function computeImageSimilarities(src) {
    logFunctionEntry('computeImageSimilarities');
    logWithEmoji('model', 'computeImageSimilarities', 'Computing similarities for image');
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
    try {
        if (message.type === 'SETTINGS_UPDATED') {
            flagShowFrameonImage = {
                ...flagShowFrameonImage,
                ...message.settings
            };
            
            // Update DEBUG flag if it's included in the settings
            if (typeof message.settings.debugMode !== 'undefined') {
                window.DEBUG = message.settings.debugMode;
                
                if (window.DEBUG) {
                    console.log('🔧 Debug mode enabled - verbose console logging activated');
                } else {
                    console.log('🔧 Debug mode disabled - reduced console logging activated');
                }
            }
            
            // Reprocess visible images with new settings
            if (flagShowFrameonImage.autoProcessImages) {
                processExistingImages();
            }
            
            sendResponse({ success: true });
        }
        else if (message.action === 'updateSettings') {
            // Handle direct settings updates from popup
            if (message.settings) {
                // Handle debug mode toggle
                if (typeof message.settings.debugMode !== 'undefined') {
                    window.DEBUG = message.settings.debugMode;
                    
                    if (window.DEBUG) {
                        console.log('🔧 Debug mode enabled - verbose console logging activated');
                        // Add diagnostic info in debug mode
                        console.log('🔧 Current settings:', flagShowFrameonImage);
                        console.log('🔧 Current blur list size:', blurTracker ? blurTracker.blurredImages.size : 'blurTracker not loaded');
                    } else {
                        console.log('🔧 Debug mode disabled - reduced console logging activated');
                    }
                }
                
                // Update other settings if included
                if (Object.keys(message.settings).length > 0) {
                    flagShowFrameonImage = {
                        ...flagShowFrameonImage,
                        ...message.settings
                    };
                }
            }
            
            sendResponse({ success: true, debug: window.DEBUG });
        }
        else if (message.type === 'REPROCESS_IMAGES') {
            // Update settings if provided
            if (message.settings) {
                flagShowFrameonImage = {
                    ...flagShowFrameonImage,
                    ...message.settings
                };
            }
            
            // Clear processed status to allow reprocessing
            imageTracker.clear();
            processedImageTracker.clear();
            
            // Reprocess visible images
            processExistingImages();
            
            sendResponse({ success: true });
        }
        else if (message.type === 'CLEAR_BLUR_LIST') {
            // Handle clearing blur list
            let count = 0;
            if (blurTracker) {
                count = blurTracker.clear();
            }
            
            // Refresh all images after a short delay
            setTimeout(() => {
                forceProcessAllImages();
            }, 500);
            
            // Return the count of cleared images
            sendResponse({
                success: true,
                count: count
            });
        }
        else if (message.type === 'GET_BLUR_COUNT') {
            // Get count of blurred images
            let count = 0;
            let historicalCount = 0;
            let autoRenewedCount = 0;
            
            if (blurTracker) {
                count = blurTracker.count();
                
                // Get historical count if available
                if (typeof blurTracker.historicalCount === 'function') {
                    historicalCount = blurTracker.historicalCount();
                }
                
                // Count auto-renewed images
                document.querySelectorAll('[data-auto-renewed="true"]').forEach(() => {
                    autoRenewedCount++;
                });
            }
            
            sendResponse({
                success: true,
                count: count,
                historicalCount: historicalCount,
                autoRenewedCount: autoRenewedCount
            });
        }
        else if (message.type === 'MODE_CHANGED') {
            
            // The actual mode change is handled by the settings update
            sendResponse({ success: true });
        }
    } catch (error) {
        console.error('Error handling message:', error);
        sendResponse({ success: false, error: error.message });
    }
    
    // Keep the message channel open for async response
    return true;
});

// Load initial settings
chrome.storage.sync.get({
    frameProsessedImage: true,
    addLabel: true,
    showConsoleLog: false,
    autoProcessImages: true,
    processingMode: 'face_detection',
    confidenceThreshold: 70,
    debugMode: false
}, function(items) {
    // Update global debug flag
    window.DEBUG = items.debugMode || false;
    
    // Initialize global settings object
    flagShowFrameonImage = {
        ...flagShowFrameonImage,
        ...items
    };
    
    if (window.DEBUG) {
        logWithEmoji('setup', 'settings', 'Debug mode enabled from saved settings', {
            settings: items,
            debugMode: true
        });
    } else {
        logWithEmoji('setup', 'settings', 'Extension initialized with normal logging');
    }
});

// Add function to check if image should be displayed
function shouldDisplayImage(src) {
    logFunctionEntry('shouldDisplayImage');
    logWithEmoji('search', 'shouldDisplayImage', 'Determining if image should be displayed');
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
    logFunctionEntry('applyBlurEffect');
    logWithEmoji('image', 'applyBlurEffect', `${shouldBlur ? 'Applying' : 'Removing'} blur effect on element`);
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
// This function is now imported from utils.js
// function debounce(func, wait) {
//     logFunctionEntry('debounce');
//     logWithEmoji('info', 'debounce', 'Creating debounced function');
//     let timeout;
//     return function executedFunction(...args) {
//         const later = () => {
//             clearTimeout(timeout);
//             func(...args);
//         };
//         clearTimeout(timeout);
//         timeout = setTimeout(later, wait);
//     };
// }

// Replace the old WorkerPool instantiation with EnhancedWorkerPool
const workerPool = new EnhancedWorkerPool({
    maxWorkers: 4,
    taskTimeout: 30000,
    retryAttempts: 2,
    batchSize: 4
});

// Initialize the worker pool during extension initialization
async function initializeExtension() {
    logFunctionEntry('initializeExtension');
    
    // Detailed timing
    const startTime = performance.now();
    let stepStartTime = startTime;
    
    try {
        // 0. Preload critical resources
        preloadCriticalResources();
        
        const preloadTime = performance.now() - stepStartTime;
        logWithEmoji('timer', 'initializeExtension', `Resource preloading took ${preloadTime.toFixed(0)}ms`);
        stepStartTime = performance.now();
        
        // 1. Initialize the models
        await ensureModelsLoaded();
        
        const modelLoadTime = performance.now() - stepStartTime;
        logWithEmoji('timer', 'initializeExtension', `Model loading took ${modelLoadTime.toFixed(0)}ms`);
        stepStartTime = performance.now();
        
        // 2. Initialize worker pool (if available)
        if (!hasWorkerRestrictions()) {
            try {
                await initializeWorkerPool();
                workerPoolInitialized = true;
            } catch (error) {
                logWithEmoji('warning', 'initializeExtension', `Worker pool initialization failed: ${error.message}`);
            }
        } else {
            logWithEmoji('warning', 'initializeExtension', 'Worker pool disabled due to restrictions');
        }
        
        const workerTime = performance.now() - stepStartTime;
        logWithEmoji('timer', 'initializeExtension', `Worker initialization took ${workerTime.toFixed(0)}ms`);
        stepStartTime = performance.now();
        
        // 3. Set up image observers
        observeElements();
        
        const observerTime = performance.now() - stepStartTime;
        logWithEmoji('timer', 'initializeExtension', `Observer setup took ${observerTime.toFixed(0)}ms`);
        stepStartTime = performance.now();
        
        // 4. Set up dynamic CSS updater for Facebook images
        setupDynamicCssUpdater();
        logWithEmoji('setup', 'initializeExtension', 'Dynamic CSS updater initialized');
        
        // 5. Process existing images
        if (flagShowFrameonImage.autoProcessImages) {
            processExistingImages();
        }
        
        // Total initialization time
        const totalTime = performance.now() - startTime;
        logWithEmoji('success', 'initializeExtension', `Extension initialized in ${totalTime.toFixed(0)}ms`);
        
        return true;
    } catch (error) {
        logWithEmoji('error', 'initializeExtension', `Initialization failed: ${error.message}`);
        return false;
    }
}

// Add helper function to rotate image
async function rotateImage(canvas, angle) {
    logFunctionEntry('rotateImage');
    logWithEmoji('image', 'rotateImage', `Rotating image by ${angle} degrees`);
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
// This function is now imported from utils.js
// function adjustDetectionCoordinates(detections, angle, canvas) {
//     logFunctionEntry('adjustDetectionCoordinates');
//     logWithEmoji('image', 'adjustDetectionCoordinates', `Adjusting coordinates for ${angle} degree rotation`);
//     const radians = (-angle * Math.PI) / 180;
//     const centerX = canvas.width / 2;
//     const centerY = canvas.height / 2;
//     
//     return detections.map(detection => {
//         const { x, y, width, height } = detection.box;
//         const cx = x + width/2 - centerX;
//         const cy = y + height/2 - centerY;
//         
//         // Rotate coordinates back
//         const rotatedX = cx * Math.cos(radians) - cy * Math.sin(radians);
//         const rotatedY = cx * Math.sin(radians) + cy * Math.cos(radians);
//         
//         return {
//             ...detection,
//             box: {
//                 x: rotatedX - width/2 + centerX,
//                 y: rotatedY - height/2 + centerY,
//                 width,
//                 height
//             }
//         };
//     });
// }

// Add recovery function
// This function is now imported from utils.js
// function recoverFailedImage(img) {
//     logFunctionEntry('recoverFailedImage');
//     logWithEmoji('setup', 'recoverFailedImage', 'Attempting to recover failed image');
//     // Restore original visibility
//     img.style.visibility = 'visible';
//     img.style.opacity = '1';
//     
//     // Remove any processing-related classes/attributes
//     const wrapper = img.closest('.face-detection-wrapper');
//     if (wrapper) {
//         const originalStyles = JSON.parse(wrapper.getAttribute('data-original-styles') || '{}');
//         Object.assign(img.style, originalStyles);
//         
//         // Unwrap the image if needed
//         if (wrapper.parentNode) {
//             wrapper.parentNode.insertBefore(img, wrapper);
//             wrapper.remove();
//         }
//     }
// }

// Add to error handling
window.addEventListener('error', function(event) {
    if (event.target.tagName === 'IMG') {
        logWithEmoji('error', 'errorHandler', 'Recovering failed image: ' + event.target.src);
        recoverFailedImage(event.target);
    }
});

// Add a helper function for all functions to log their entry point
// This function is now imported from utils.js
// function logFunctionEntry(functionName) {
//     logWithEmoji('setup', functionName, 'Function started');
// }

// Improve the error handling in the sandbox.html communication by adding a special error handler function

// Add the logError function after logWithEmoji
// This function is now imported from utils.js
// function logError(functionName, message, error = null) {
//     logWithEmoji('error', functionName, message);
//     if (error && error.stack) {
//         console.error(`${functionName} error stack:`, error.stack);
//     } else if (error) {
//         console.error(`${functionName} error details:`, error);
//     }
// }

// Add a cleanup utility function for handling message event listeners
// This function is now imported from utils.js
// function createMessageHandler(expectedType, timeout, onSuccess, onError) {
//     return new Promise((resolve, reject) => {
//         let messageListener = null;
//         let timeoutId = null;
//         
//         const cleanup = () => {
//             if (timeoutId) clearTimeout(timeoutId);
//             if (messageListener) window.removeEventListener('message', messageListener);
//         };
//         
//         messageListener = (event) => {
//             if (event.data && event.data.type === expectedType) {
//                 cleanup();
//                 if (onSuccess) {
//                     try {
//                         const result = onSuccess(event.data);
//                         resolve(result);
//                     } catch (error) {
//                         logError('messageHandler', `Error handling successful ${expectedType} message:`, error);
//                         reject(error);
//                     }
//                 } else {
//                     resolve(event.data);
//                 }
//             }
//         };
//         
//         window.addEventListener('message', messageListener);
//         
//         timeoutId = setTimeout(() => {
//             cleanup();
//             const error = new Error(`Timeout waiting for ${expectedType} message (${timeout}ms)`);
//             if (onError) {
//                 try {
//                     onError(error);
//                 } catch (callbackError) {
//                     logError('messageHandler', `Error in timeout handler for ${expectedType}:`, callbackError);
//                 }
//             }
//             reject(error);
//         }, timeout);
//         
//         return { cleanup };
//     });
// }

// Add the utils.js script to content.js by creating a script element
document.addEventListener('DOMContentLoaded', function() {
    // Inject utils.js for logging functions
    const utilsScript = document.createElement('script');
    utilsScript.src = chrome.runtime.getURL('js/utils.js');
    utilsScript.onload = function() {
        console.log('Utils script loaded');
    };
    document.head.appendChild(utilsScript);
});

// This function is also available in utils.js (duplicated here for now)
function roundToMultipleOf32(num) {
    return Math.ceil(num / 32) * 32;
}

function setupTabVisibilityHandler() {
    logFunctionEntry('setupTabVisibilityHandler');
    logWithEmoji('setup', 'setupTabVisibilityHandler', 'Setting up tab visibility handler');
    
    // The TabResourceManager already handles this functionality,
    // but we'll add our own event listeners for enhanced functionality
    
    // Listen for custom events from TabResourceManager
    window.addEventListener('faceone:suspended', () => {
        logWithEmoji('lock', 'tabVisibility', 'Tab became inactive, extension processing suspended');
        
        // Additional tab-specific handling
        if (imageQueue) {
            imageQueue.pause();
            logWithEmoji('success', 'tabVisibility', 'Image queue processing paused');
        }
        
        // Mark pending images as needing reprocessing
        if (processedImageTracker) {
            let pendingCount = 0;
            processedImageTracker.processed.forEach((data, id) => {
                if (data.state === 'pending') {
                    // Delete pending states to force reprocessing when tab becomes active again
                    processedImageTracker.processed.delete(id);
                    pendingCount++;
                }
            });
            
            if (pendingCount > 0) {
                logWithEmoji('info', 'tabVisibility', `Cleared ${pendingCount} pending images from tracker`);
            }
        }
    });
    
    window.addEventListener('faceone:resumed', () => {
        logWithEmoji('unlock', 'tabVisibility', 'Tab became active, extension processing resumed');
        
        // Additional tab-specific handling
        if (imageQueue) {
            imageQueue.resume();
            logWithEmoji('success', 'tabVisibility', 'Image queue processing resumed');
        }
        
        // Reset image tracking for all non-processed images
        if (processedImageTracker) {
            // Only remove pending images, keep completed ones
            const completedImages = new Map();
            processedImageTracker.processed.forEach((data, id) => {
                if (data.state === 'complete') {
                    completedImages.set(id, data);
                }
            });
            
            // Clear tracker and restore completed images
            processedImageTracker.clear();
            completedImages.forEach((data, id) => {
                processedImageTracker.processed.set(id, data);
            });
            
            logWithEmoji('success', 'tabVisibility', 'Image tracking state reset for pending images');
        }
        
        // Process any images that became visible while tab was inactive
        if (isInitialized && flagShowFrameonImage.autoProcessImages) {
            setTimeout(() => {
                processExistingImages();
            }, 1000);
        }
    });
}

// Preload resources using resource hints for faster loading
function preloadCriticalResources() {
    logFunctionEntry('preloadCriticalResources');
    
    const resources = [
        { type: 'script', url: chrome.runtime.getURL('lib/tf.min.js') },
        { type: 'script', url: chrome.runtime.getURL('lib/face-api.min.js') },
        { type: 'fetch', url: chrome.runtime.getURL('models/FaceAPI/tiny_face_detector/model.json') }
    ];
    
    for (const resource of resources) {
        try {
            const link = document.createElement('link');
            link.rel = resource.type === 'script' ? 'preload' : 'prefetch';
            link.href = resource.url;
            link.as = resource.type;
            link.crossOrigin = 'anonymous';
            
            document.head.appendChild(link);
            
            logWithEmoji('setup', 'preloadCriticalResources', `Preloaded ${resource.type}: ${resource.url.split('/').pop()}`);
        } catch (error) {
            logWithEmoji('warning', 'preloadCriticalResources', `Failed to preload ${resource.url}: ${error.message}`);
        }
    }
}

// Dynamically load scripts only when needed
async function loadScriptOnDemand(scriptPath) {
    logFunctionEntry('loadScriptOnDemand');
    logWithEmoji('loading', 'loadScriptOnDemand', `Loading script: ${scriptPath.split('/').pop()}`);
    
    return new Promise((resolve, reject) => {
        const scriptUrl = chrome.runtime.getURL(scriptPath);
        
        // Check if script is already loaded
        const existingScript = document.querySelector(`script[src="${scriptUrl}"]`);
        if (existingScript) {
            logWithEmoji('info', 'loadScriptOnDemand', `Script already loaded: ${scriptPath.split('/').pop()}`);
            return resolve(existingScript);
        }
        
        // Create a new script element
        const script = document.createElement('script');
        script.src = scriptUrl;
        script.async = true;
        
        // Set up event handlers
        script.onload = () => {
            logWithEmoji('success', 'loadScriptOnDemand', `Script loaded successfully: ${scriptPath.split('/').pop()}`);
            resolve(script);
        };
        
        script.onerror = (error) => {
            logWithEmoji('error', 'loadScriptOnDemand', `Failed to load script: ${scriptPath.split('/').pop()}`);
            reject(new Error(`Failed to load script: ${scriptPath}`));
        };
        
        // Add script to document
        document.head.appendChild(script);
    });
}

// Cache for storing loaded resources
const resourceCache = {
    scripts: new Map(),
    models: new Map(),
    
    // Add a resource to the cache
    add(type, key, resource) {
        if (type === 'script') {
            this.scripts.set(key, { 
                resource, 
                timestamp: Date.now() 
            });
        } else if (type === 'model') {
            this.models.set(key, { 
                resource, 
                timestamp: Date.now() 
            });
        }
    },
    
    // Get a resource from the cache
    get(type, key) {
        if (type === 'script') {
            return this.scripts.get(key)?.resource;
        } else if (type === 'model') {
            return this.models.get(key)?.resource;
        }
        return null;
    },
    
    // Check if a resource exists in the cache
    has(type, key) {
        if (type === 'script') {
            return this.scripts.has(key);
        } else if (type === 'model') {
            return this.models.has(key);
        }
        return false;
    }
};

/**
 * Determines the optimal number of workers based on device capabilities
 * @returns {number} The optimal number of workers
 */
function determineOptimalWorkerCount() {
    // Get the number of CPU cores
    const cores = navigator.hardwareConcurrency || 4;
    
    // Check device memory if available
    let deviceMemoryGB = 4; // Default assumption: 4GB RAM
    if (navigator.deviceMemory) {
        deviceMemoryGB = navigator.deviceMemory;
    }
    
    // Low-end devices: Use fewer workers
    if (deviceMemoryGB <= 2 || cores <= 2) {
        return 1; // Single worker for low-end devices
    }
    
    // Mid-range devices: 2 workers
    if (deviceMemoryGB <= 4 || cores <= 4) {
        return 2;
    }
    
    // High-end devices: Use up to cores-1 workers, max 4
    return Math.min(cores - 1, 4);
}

/**
 * Initializes the worker pool with optimal settings for the current device
 * @returns {Promise<boolean>} True if initialization was successful
 */
async function initializeWorkerPool() {
    logFunctionEntry('initializeWorkerPool');
    
    try {
        // Check for CSP restrictions that would prevent workers
        if (hasWorkerRestrictions()) {
            logWithEmoji('warning', 'initializeWorkerPool', 'Worker creation restricted by CSP, skipping worker pool');
            return false;
        }
        
        // Determine optimal worker count
        const workerCount = determineOptimalWorkerCount();
        logWithEmoji('info', 'initializeWorkerPool', `Using ${workerCount} workers based on device capabilities`);
        
        // Initialize worker pool with adaptive settings
        workerPool = new WorkerPool({ 
            maxWorkers: workerCount,
            taskTimeout: 15000, // 15 seconds timeout for tasks
            batchSize: 4,
            // Reduce restart threshold for low-memory devices
            workerRestartThreshold: navigator.deviceMemory && navigator.deviceMemory <= 2 ? 500 : 1000
        });
        
        const initialized = await workerPool.initialize();
        
        if (initialized) {
            logWithEmoji('success', 'initializeWorkerPool', `Worker pool initialized with ${workerCount} workers`);
            return true;
        } else {
            logWithEmoji('error', 'initializeWorkerPool', 'Worker pool initialization failed');
            return false;
        }
    } catch (error) {
        logWithEmoji('error', 'initializeWorkerPool', `Failed to initialize worker pool: ${error.message}`);
        return false;
    }
}

/**
 * Marks an image for fallback processing when worker-based processing is restricted
 * This creates a wrapper and visual indicator to show the image is being handled by FaceOne
 * @param {HTMLElement} img - The image element to mark for fallback processing
 */
function markImageForFallbackProcessing(img) {
    logFunctionEntry('markImageForFallbackProcessing');
    
    // Skip if already processed or being processed
    if (img.closest('.face-detection-wrapper')) {
        return;
    }
    
    // Create a wrapper div for the image
    const wrapper = document.createElement('div');
    wrapper.className = 'face-detection-wrapper';
    wrapper.style.position = 'relative';
    wrapper.style.display = 'inline-block';
    wrapper.style.maxWidth = '100%';
    wrapper.style.lineHeight = '0';
    wrapper.setAttribute('data-faceone-indicator', 'true');
    
    // Store original styles
    const computedStyle = window.getComputedStyle(img);
    const originalStyles = {
        position: img.style.position,
        margin: img.style.margin,
        padding: img.style.padding,
        display: img.style.display
    };
    wrapper.setAttribute('data-original-styles', JSON.stringify(originalStyles));
    
    // Add the wrapper around the image
    if (img.parentNode) {
        img.parentNode.insertBefore(wrapper, img);
        wrapper.appendChild(img);
    }
    
    // Add a small indicator to show FaceOne is active
    const indicator = document.createElement('div');
    indicator.className = 'faceone-fallback-indicator';
    indicator.style.position = 'absolute';
    indicator.style.top = '5px';
    indicator.style.right = '5px';
    indicator.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    indicator.style.color = 'white';
    indicator.style.padding = '2px 5px';
    indicator.style.fontSize = '10px';
    indicator.style.borderRadius = '3px';
    indicator.style.zIndex = '9999';
    indicator.style.pointerEvents = 'none';
    indicator.textContent = 'FaceOne';
    
    wrapper.appendChild(indicator);
    
    // Mark the image as being processed
    img.setAttribute('data-faceone-fallback', 'true');
    
    logWithEmoji('success', 'markImageForFallbackProcessing', 'Image marked for fallback processing');
}

/**
 * Checks if the current environment has restrictions that prevent using Web Workers
 * This is typically due to Content Security Policy (CSP) restrictions
 * @returns {boolean} True if workers are restricted, false otherwise
 */
function hasWorkerRestrictions() {
    logFunctionEntry('hasWorkerRestrictions');
    
    try {
        // Try to detect CSP that blocks workers
        if (window.__worker_restrictions_detected) {
            return true; // Already detected restrictions
        }
        
        // Check if we're on a site known to use restrictive CSP
        const hostname = window.location.hostname;
        const knownRestrictiveSites = [
            'github.com',
            'gitlab.com',
            'bitbucket.org',
            'stackoverflow.com',
            'jira.atlassian.com'
        ];
        
        if (knownRestrictiveSites.some(site => hostname.includes(site))) {
            logWithEmoji('warning', 'hasWorkerRestrictions', `Detected known restrictive site: ${hostname}`);
            window.__worker_restrictions_detected = true;
            return true;
        }
        
        // Try creating a worker as a test
        try {
            // Create a simple worker from a blob
            const blob = new Blob(['self.onmessage = function() { self.postMessage("ok"); }'], 
                                { type: 'application/javascript' });
            const url = URL.createObjectURL(blob);
            const worker = new Worker(url);
            
            // Clean up
            worker.terminate();
            URL.revokeObjectURL(url);
            
            logWithEmoji('success', 'hasWorkerRestrictions', 'Worker creation test successful, no restrictions detected');
            return false;
        } catch (e) {
            logWithEmoji('warning', 'hasWorkerRestrictions', `Worker creation failed: ${e.message}`);
            window.__worker_restrictions_detected = true;
            return true;
        }
    } catch (error) {
        logWithEmoji('error', 'hasWorkerRestrictions', `Error checking worker restrictions: ${error.message}`);
        return false; // Default to no restrictions on error
    }
}

/**
 * Checks if an element is visible and how much of it is in the viewport
 * @param {HTMLElement} element - The element to check
 * @returns {Object} Object with visibility information
 */
function getElementVisibility(element) {
    logFunctionEntry('getElementVisibility');
    
    // Default return value for invalid elements
    if (!element || !element.isConnected) {
        return { isVisible: false, fullyVisible: false, visibilityRatio: 0 };
    }
    
    try {
        // Get element boundaries
        const rect = element.getBoundingClientRect();
        
        // Check if element has size
        if (rect.width === 0 || rect.height === 0) {
            return { isVisible: false, fullyVisible: false, visibilityRatio: 0 };
        }
        
        // Get viewport dimensions
        const viewportWidth = window.innerWidth || document.documentElement.clientWidth;
        const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
        
        // Check if any part of the element is in the viewport
        const isPartiallyVisible = (
            rect.top < viewportHeight &&
            rect.bottom > 0 &&
            rect.left < viewportWidth &&
            rect.right > 0
        );
        
        // Check if element is fully visible
        const isFullyVisible = (
            rect.top >= 0 &&
            rect.bottom <= viewportHeight &&
            rect.left >= 0 &&
            rect.right <= viewportWidth
        );
        
        // Calculate what percentage of the element is visible
        let visibilityRatio = 0;
        
        if (isPartiallyVisible) {
            const visibleWidth = Math.min(rect.right, viewportWidth) - Math.max(rect.left, 0);
            const visibleHeight = Math.min(rect.bottom, viewportHeight) - Math.max(rect.top, 0);
            const visibleArea = visibleWidth * visibleHeight;
            const totalArea = rect.width * rect.height;
            
            visibilityRatio = visibleArea / totalArea;
        }
        
        // Check if element is hidden by CSS
        const style = window.getComputedStyle(element);
        const isStyleVisible = (
            style.display !== 'none' &&
            style.visibility !== 'hidden' &&
            parseFloat(style.opacity) > 0
        );
        
        const isVisible = isPartiallyVisible && isStyleVisible;
        
        return {
            isVisible,
            fullyVisible: isFullyVisible && isStyleVisible,
            visibilityRatio: isVisible ? visibilityRatio : 0
        };
    } catch (error) {
        logWithEmoji('error', 'getElementVisibility', `Error checking visibility: ${error.message}`);
        return { isVisible: false, fullyVisible: false, visibilityRatio: 0 };
    }
}

/**
 * Checks if the current site has known restrictions that may affect extension functionality
 * @returns {boolean} True if the site has known restrictions
 */
function checkIfRestrictedSite() {
    logFunctionEntry('checkIfRestrictedSite');
    
    try {
        const hostname = window.location.hostname;
        
        // List of sites with known CSP restrictions
        const restrictedSites = [
            'github.com',
            'gitlab.com',
            'bitbucket.org',
            'stackoverflow.com',
            'jira.atlassian.com',
            'confluence.atlassian.com',
            'docs.google.com',
            'drive.google.com',
            'mail.google.com',
            'sheets.google.com',
            'slides.google.com'
        ];
        
        // Check if current site is in the restricted list
        const isRestricted = restrictedSites.some(site => hostname.includes(site));
        
        if (isRestricted) {
            logWithEmoji('warning', 'checkIfRestrictedSite', `Detected restricted site: ${hostname}`);
            return true;
        }
        
        return false;
    } catch (error) {
        logWithEmoji('error', 'checkIfRestrictedSite', `Error checking site restrictions: ${error.message}`);
        return false;
    }
}

/**
 * Tracker to manage the processing state of images to prevent double-processing
 */
const processedImageTracker = {
    // Map to store processed images with their state
    processed: new Map(),
    
    /**
     * Marks an image as processed with the given state
     * @param {HTMLElement} img - The image element to mark
     * @param {string} state - The processing state ('pending', 'complete', 'failed', 'skipped')
     */
    mark(img, state) {
        if (!img) return;
        
        const id = this.getImageIdentifier(img);
        this.processed.set(id, {
            state,
            timestamp: Date.now()
        });
        
        // Also add a data attribute for quick checks
        img.setAttribute('data-faceone-processed', state);
        
        // Debug log for tracking
        logWithEmoji('debug', 'processedImageTracker', `Image marked as '${state}': ${id.substring(0, 50)}...`);
    },
    
    /**
     * Checks if an image has already been processed
     * @param {HTMLElement} img - The image element to check
     * @returns {boolean} True if the image has been processed or is being processed
     */
    isProcessed(img) {
        if (!img) return false;
        
        // Quick check using attribute
        if (img.hasAttribute('data-faceone-processed')) {
            return true;
        }
        
        // Full check using identifier
        const id = this.getImageIdentifier(img);
        const result = this.processed.has(id);
        
        if (result) {
            const data = this.processed.get(id);
            // Debug log for tracking
            logWithEmoji('debug', 'processedImageTracker', `Image check: already ${data.state}: ${id.substring(0, 50)}...`);
        }
        
        return result;
    },
    
    /**
     * Gets the processing state of an image
     * @param {HTMLElement} img - The image element to check
     * @returns {string|null} The processing state or null if not processed
     */
    getState(img) {
        if (!img) return null;
        
        const id = this.getImageIdentifier(img);
        const data = this.processed.get(id);
        return data ? data.state : null;
    },
    
    /**
     * Generates a unique identifier for an image based on its attributes
     * @param {HTMLElement} img - The image element
     * @returns {string} A unique identifier string
     */
    getImageIdentifier(img) {
        if (!img) return '';
        
        // For standard images, use src
        if (img.tagName === 'IMG' && img.src) {
            return `img_${img.src}`;
        }
        
        // For SVG images, use xlink:href attribute
        if (img.tagName === 'image' && img.getAttributeNS('http://www.w3.org/1999/xlink', 'href')) {
            return `svg_${img.getAttributeNS('http://www.w3.org/1999/xlink', 'href')}`;
        }
        
        // Use DOM path as fallback
        return this.getElementPath(img);
    },
    
    /**
     * Generates a CSS path that uniquely identifies an element
     * @param {HTMLElement} element - The element to get a path for
     * @returns {string} The CSS path for the element
     */
    getElementPath(element) {
        if (!element || !element.parentElement) return '';
        
        let path = '';
        let current = element;
        
        while (current && current !== document.body) {
            let selector = current.tagName.toLowerCase();
            
            // Add ID if present
            if (current.id) {
                selector += `#${current.id}`;
            } else {
                // Otherwise add position among siblings
                let sibling = current;
                let position = 1;
                
                while (sibling = sibling.previousElementSibling) {
                    if (sibling.tagName === current.tagName) {
                        position++;
                    }
                }
                
                if (position > 1 || !current.nextElementSibling || 
                    current.nextElementSibling.tagName !== current.tagName) {
                    selector += `:nth-of-type(${position})`;
                }
            }
            
            path = selector + (path ? '>' + path : '');
            current = current.parentElement;
        }
        
        return path;
    },
    
    /**
     * Clears the tracking data
     */
    clear() {
        this.processed.clear();
    }
};

/**
 * Forces processing of all images on the page by clearing the tracker and reprocessing
 * This is useful when auto-processing seems to be stuck
 */
async function forceProcessAllImages() {
    logFunctionEntry('forceProcessAllImages');
    logWithEmoji('info', 'forceProcessAllImages', 'Forcing reprocessing of all images on page');
    
    // Clear the tracker completely
    if (processedImageTracker) {
        processedImageTracker.clear();
        logWithEmoji('success', 'forceProcessAllImages', 'Image tracker cleared');
    }
    
    // Clear any data-faceone-processed attributes on images
    const processedImages = document.querySelectorAll('[data-faceone-processed]');
    let clearedCount = 0;
    
    processedImages.forEach(img => {
        img.removeAttribute('data-faceone-processed');
        clearedCount++;
    });
    
    logWithEmoji('info', 'forceProcessAllImages', `Cleared ${clearedCount} processed image attributes`);
    
    // Now process all images again
    await processExistingImages();
}

// Find the handleImageVisible function and modify it to check blurTracker first
function handleImageVisible(entry) {
    const img = entry.target;
    
    if (!img || !isValidElement(img)) {
        return;
    }
    
    // Get image source
    const src = img.tagName === 'IMG' ? img.src : img.getAttribute('xlink:href');
    
    // Skip if no source
    if (!src) {
        return;
    }
    
    // Check if this image is currently in the blur list
    if (blurTracker && blurTracker.shouldBlur(src)) {
        // Create wrapper for the image if it doesn't exist
        const wrapper = findOrCreateWrapper(img);
        if (wrapper) {
            // Immediately apply blur without needing to process again
            img.style.filter = 'blur(10px)';
            img.classList.add('blurred-image');
            img.classList.add('blurred-by-url'); // Add this class for CSS targeting
            img.setAttribute('data-faceone-processed', 'blurred');
            
            // Set a custom attribute with the normalized URL for debugging
            try {
                const normalizedUrl = blurTracker.normalizeImageUrl(src);
                img.setAttribute('data-blurred-url', normalizedUrl.substring(0, 50) + '...');
                
                // Send diagnostic message to popup
                if (window.DEBUG) {
                    console.log(`🔍 Image blurred from saved list: ${normalizedUrl.substring(0, 50)}...`);
                    
                    // If in debug mode, add additional diagnostic information
                    if (typeof processedImageTracker !== 'undefined' && processedImageTracker) {
                        const id = processedImageTracker.getImageIdentifier(img);
                        console.log(`🔍 Debug: Image identifier for blurred image: ${id}`);
                    }
                }
            } catch (e) {
                // Ignore errors in URL normalization
                if (window.DEBUG) {
                    console.error('🔍 Error in URL normalization:', e);
                }
            }
            
            if (flagShowFrameonImage.addLabel) {
                addResultIndicator(wrapper, 'Image blurred - From saved list');
            }
        }
        
        // Mark as processed in imageTracker to prevent redundant processing
        if (imageTracker) {
            try {
                imageTracker.markProcessed(src, true, true);
            } catch (error) {
                console.error('Error marking image as processed:', error);
            }
        }
        
        // Also mark in processedImageTracker
        if (processedImageTracker && typeof processedImageTracker.mark === 'function') {
            processedImageTracker.mark(img, 'blurred');
        }
        
        return; // Skip processing - we already know it needs blurring
    }
    
    // Check if this image was previously blurred but has expired
    if (blurTracker && blurTracker.wasBlurredBefore(src)) {
        console.log('Found previously blurred image that expired, automatically renewing:', src.substring(0, 50) + '...');
        
        // Create wrapper for the image if it doesn't exist
        const wrapper = findOrCreateWrapper(img);
        if (wrapper) {
            // Automatically reactivate blur
            blurTracker.markForBlur(src);
            
            // Apply blur effect
            img.style.filter = 'blur(10px)';
            img.classList.add('blurred-image');
            img.classList.add('blurred-by-url');
            img.setAttribute('data-faceone-processed', 'blurred');
            img.setAttribute('data-auto-renewed', 'true');
            
            // Set a custom attribute with the normalized URL for debugging
            try {
                const normalizedUrl = blurTracker.normalizeImageUrl(src);
                img.setAttribute('data-blurred-url', normalizedUrl.substring(0, 50) + '...');
            } catch (e) {
                // Ignore errors in URL normalization
                if (window.DEBUG) {
                    console.error('🔍 Error in URL normalization:', e);
                }
            }
            
            if (flagShowFrameonImage.addLabel) {
                addResultIndicator(wrapper, 'Image auto-reblurred - Previously detected');
            }
        }
        
        // Mark as processed
        if (imageTracker) {
            try {
                imageTracker.markProcessed(src, true, true);
            } catch (error) {
                console.error('Error marking image as processed:', error);
            }
        }
        
        // Also mark in processedImageTracker
        if (processedImageTracker && typeof processedImageTracker.mark === 'function') {
            processedImageTracker.mark(img, 'blurred');
        }
        
        return; // Skip further processing
    }
    
    // Continue with normal processing for unrecognized images
    if (shouldProcessImage(img) && !processedImageTracker.isProcessed(img)) {
        processImage(img);
    }
}

/**
 * Finds an existing wrapper for an image or creates a new one
 * @param {HTMLElement} img - The image element
 * @returns {HTMLElement|null} - The wrapper element or null if not found/created
 */
function findOrCreateWrapper(img) {
    if (!img) return null;
    
    // Check if there's already a wrapper for this image
    if (img.parentElement && img.parentElement.classList.contains('faceone-wrapper')) {
        return img.parentElement;
    }
    
    // If not, create a new wrapper
    try {
        return createWrapper(img);
    } catch (error) {
        console.error('Error creating wrapper:', error);
        return null;
    }
}

// Inject dynamic CSS rules for specific URLs
function injectDynamicCssRules() {
    // Don't inject if blurTracker isn't available
    if (!blurTracker || blurTracker.blurredImages.size === 0) return;
    
    try {
        // Create a style element if it doesn't exist yet
        let styleEl = document.getElementById('faceone-dynamic-styles');
        if (!styleEl) {
            styleEl = document.createElement('style');
            styleEl.id = 'faceone-dynamic-styles';
            document.head.appendChild(styleEl);
        }
        
        // Generate CSS rules for each blurred URL
        let cssRules = '';
        for (const [url, _] of blurTracker.blurredImages.entries()) {
            // Create a safe selector from the URL
            try {
                // Extract just the filename part of the URL
                const urlObj = new URL(url);
                const pathParts = urlObj.pathname.split('/');
                const filename = pathParts[pathParts.length - 1].split('.')[0];
                
                if (filename && filename.length > 5) {
                    // Create a selector that targets images with this filename in their src
                    cssRules += `img[src*="${filename}"] { filter: blur(10px) !important; }\n`;
                }
            } catch (e) {
                console.error('Error generating CSS for URL', url, e);
            }
        }
        
        // Add the rules to the style element
        styleEl.textContent = cssRules;
        
        console.log(`Injected ${blurTracker.blurredImages.size} dynamic CSS rules for blurring`);
    } catch (error) {
        console.error('Error injecting dynamic CSS rules:', error);
    }
}

// Setup function to periodically update dynamic CSS
function setupDynamicCssUpdater() {
    // Initial injection
    injectDynamicCssRules();
    
    // Periodically update the CSS rules
    setInterval(() => {
        injectDynamicCssRules();
    }, 5000); // Every 5 seconds
}

// Add this to the initialization process
async function initializeExtension() {
    const startTime = performance.now();
    logWithEmoji('start', 'initializeExtension', 'Starting extension initialization');
    
    if (extensionInitialized) {
        logWithEmoji('info', 'initializeExtension', 'Extension already initialized, skipping');
        return true;
    }
        
    setupDynamicCssUpdater(); // Add this line to set up the dynamic CSS updater
    
}
