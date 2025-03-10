// Constants for refresh loop detection
const PAGE_REFRESH_KEY = 'faceone_page_refresh';
const MAX_REFRESHES = 15; // Increased from 10 to 15 for higher tolerance
const REFRESH_COOLDOWN = 180000; // 3 minutes in milliseconds (increased from 2 minutes)

// Sandbox constants
const SANDBOX_CREATION_TIMEOUT = 60000; // 60 seconds timeout for sandbox creation

// Add a flag to indicate if the refresh loop has been cleared
let refreshLoopCleared = false;

// Define processingQueue before it's used
const processingQueue = {
    queue: [],
    paused: false,
    
    pause() {
        this.paused = true;
        console.log('Processing queue paused');
    },
    
    resume() {
        this.paused = false;
        console.log('Processing queue resumed');
        
        // If we have items in the queue, start processing
        if (this.queue.length > 0) {
            this.process();
        }
    },
    
    add(element, priority = false) {
        if (!element) return;
        
        // Skip if already in queue
        if (this.queue.some(item => item.element === element)) {
            return;
        }
        
        // Add to queue with priority flag
        const item = { element, priority };
        
        if (priority) {
            this.queue.unshift(item); // Add to front if priority
        } else {
            this.queue.push(item); // Add to end otherwise
        }
        
        // Start processing if not paused
        if (!this.paused) {
            this.process();
        }
    },
    
    async process() {
        // Skip if paused or already processing
        if (this.paused || state.isProcessing) return;
        
        // Skip if queue is empty
        if (this.queue.length === 0) return;
        
        // Set processing flag
        state.isProcessing = true;
        
        try {
            // Get next item from queue
            const item = this.queue.shift();
            
            // Process the item
            await handleVisibleElement(item.element);
        } catch (error) {
            console.error('Error processing queue item:', error);
        } finally {
            // Reset processing flag
            state.isProcessing = false;
            
            // Continue processing if items remain and not paused
            if (this.queue.length > 0 && !this.paused) {
                // Use setTimeout to avoid blocking the main thread
                setTimeout(() => this.process(), 10);
            }
        }
    }
};

// Simple version of resetRefreshLoop that only clears storage and doesn't depend on variables
function clearRefreshLoopFlags() {
    try {
        // Clear session storage refresh counter
        sessionStorage.removeItem(PAGE_REFRESH_KEY);
        
        // Clear all disable flags
        localStorage.removeItem('faceone_disabled_until');
        localStorage.removeItem('faceone_permanently_disabled');
        localStorage.removeItem('faceone_permanent_disable'); // Old key for backward compatibility
        
        console.log('[FaceOne] Cleared refresh loop flags on startup');
        refreshLoopCleared = true;
    } catch (e) {
        console.warn('[FaceOne] Error clearing refresh loop flags:', e);
    }
}

// Clear refresh loop flags on startup
clearRefreshLoopFlags();

// Initialize sandboxManager
let sandboxManager = null;

// Main entry point
(function() {
    console.log('#### STARTING CONTENT.JS ####');

    // Check if we're in a frame
    if (window !== window.top) {
        console.log('Running in iframe, skipping initialization');
        return;
    }

    // Check if extension is disabled due to refresh loop
    if (isExtensionDisabled()) {
        console.log('[FaceOne] Extension is disabled due to refresh loop detection, skipping initialization');
        return;
    }

    // Start initialization
    startInitialization();
    
    // Listen for page visibility changes
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            console.log('Page hidden, pausing processing');
            processingQueue.pause();
        } else {
            console.log('Page visible, resuming processing');
            processingQueue.resume();
        }
    });

    // Initialize when document is ready
    async function startInitialization() {
        try {
            // Check if models directory exists
            const modelsExist = await checkDirectoryExists('models');
            if (!modelsExist) {
                console.error('Models directory not found. Extension initialization aborted.');
                return;
            }
            
            // Initialize the extension
            await initialize();
        } catch (error) {
            console.error('Error during initialization:', error);
        }
    }

    // Handle document readiness
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', startInitialization);
    } else {
        startInitialization();
    }

    // Safe check for extension disabled status - can be called anywhere
    function isExtensionDisabled() {
        try {
            const now = Date.now();
            
            // Check if the extension is temporarily disabled
            const disabledUntil = localStorage.getItem('faceone_disabled_until');
            if (disabledUntil && parseInt(disabledUntil) > now) {
                const remainingTime = Math.ceil((parseInt(disabledUntil) - now) / 60000);
                console.log(`[FaceOne] Extension temporarily disabled. Re-enabling in ${remainingTime} minutes.`);
                return true;
            }
            
            // Check if the extension is permanently disabled
            if (localStorage.getItem('faceone_permanently_disabled') === 'true') {
                console.log('[FaceOne] Extension permanently disabled due to severe refresh loop.');
                return true;
            }
            
            return false;
        } catch (e) {
            console.warn('[FaceOne] Error checking disabled status:', e);
            return false; // Assume not disabled on error
        }
    }

    // Add this to the main entry point function to check disabled status early
    if (isExtensionDisabled()) {
        console.log('Extension is disabled, skipping initialization');
        return;
    }
})();

// Global initialization variables
let isInitializing = false;
let initializationComplete = false;
let initializationAttempts = 0;
const MAX_INIT_ATTEMPTS = 3;
const INIT_COOLDOWN_PERIOD = 60000; // 1 minute cooldown between attempts
let lastInitAttempt = 0;

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
    faceApiLoaded: false,
    faceApiLoading: false,
    faceApiError: null,
    faceNetLoaded: false,
    faceNetLoading: false,
    faceNetError: null,
    myModelLoaded: false,
    myModelLoading: false,
    myModelError: null,
    modelsLoaded: false,
    modelLoadAttempts: 0,
    MAX_LOAD_ATTEMPTS: 3,
    isProcessing: false,
    processingQueue: []
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
    processingMode: 'face_detection', // 'face_detection' or 'blur_mode'
    maxImagesPerPage: 100, // Limit number of images processed per page
    progressiveProcessing: true, // Process large images in multiple passes
    memoryManagement: {
        enabled: true,
        maxTensors: 1000,
        maxBytes: 200 * 1024 * 1024, // 200MB
        cleanupInterval: 60000 // 1 minute
    }
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

// Compatibility layer for transitioning from sandboxFrame to sandboxManager
let sandboxFrame = null; // Keep for backward compatibility

// Helper function to ensure we're using sandboxManager
function ensureSandboxManager() {
    // If we already have a valid sandboxManager, return it
    if (sandboxManager && sandboxManager.iframe) {
        console.log('[FaceOne] Using existing sandbox manager');
        return sandboxManager;
    }
    
    console.log('[FaceOne] Creating new SandboxManager instance');
    
    // If we have a sandboxFrame but no sandboxManager, create one
    if (sandboxFrame && (!sandboxManager || !sandboxManager.iframe)) {
        console.log('[FaceOne] Creating sandboxManager from existing sandboxFrame');
        sandboxManager = new SandboxManager();
        sandboxManager.iframe = sandboxFrame;
        sandboxManager.isReady = true;
    } else if (!sandboxManager) {
        // Create a new sandbox manager if none exists
        console.log('[FaceOne] Creating brand new SandboxManager');
        sandboxManager = new SandboxManager();
    }
    
    // If we have a sandboxManager but no sandboxFrame, update the reference
    if (sandboxManager && sandboxManager.iframe && !sandboxFrame) {
        console.log('[FaceOne] Updating sandboxFrame reference from sandboxManager');
        sandboxFrame = sandboxManager.iframe;
    }
    
    return sandboxManager;
}

// Add a debug logging function at the top of the file
function debugLog(message, data = null) {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0]; // HH:MM:SS format
    const prefix = `[FaceOne ${timestamp}]`;
    
    if (data) {
        console.log(`${prefix} ${message}`, data);
    } else {
        console.log(`${prefix} ${message}`);
    }
}

// Add more detailed logging to createSandboxFrame
async function createSandboxFrame(timeout) {
    debugLog('Creating sandbox frame with timeout:', timeout);
    
    if (sandboxManager && sandboxManager.isReady) {
        debugLog('Sandbox manager already exists and is ready');
        return sandboxManager.iframe;
    }
    
    try {
        debugLog('Ensuring sandbox manager exists');
        ensureSandboxManager();
        
        // Check if the sandbox.html file can be accessed
        const sandboxUrl = chrome.runtime.getURL('sandbox.html');
        debugLog('Verifying sandbox URL:', sandboxUrl);
        
        try {
            const response = await fetch(sandboxUrl, { 
                method: 'HEAD',
                cache: 'no-cache' // Try without caching
            });
            
            if (!response.ok) {
                throw new Error(`Sandbox HTML file not accessible. Status: ${response.status}`);
            }
            
            debugLog('Sandbox HTML file is accessible');
        } catch (error) {
            debugLog('Error accessing sandbox HTML file:', error);
            throw new Error(`Failed to access sandbox.html: ${error.message}`);
        }
        
        debugLog('Initializing sandbox manager');
        await sandboxManager.initialize();
        
        debugLog('Waiting for sandbox to be ready');
        await sandboxManager.waitForReady();
        
        debugLog('Sandbox frame created successfully');
        return sandboxManager.iframe;
    } catch (error) {
        debugLog('Error creating sandbox frame:', error);
        throw error;
    }
}

/**
 * Load FaceNet model
 */
async function loadFaceNetModel() {
    if (state.faceNetLoaded) {
        console.log("FaceNet model already loaded");
        return;
    }

    if (state.faceNetLoading) {
        console.log("FaceNet model already loading");
        return;
    }

    state.faceNetLoading = true;
    state.faceNetError = null;

    try {
        // Ensure TensorFlow is ready
        try {
            await ensureTensorFlowReady();
            console.log("[FaceOne " + new Date().toLocaleTimeString() + "] TensorFlow ready for FaceNet model loading");
        } catch (error) {
            console.error("[FaceOne " + new Date().toLocaleTimeString() + "] TensorFlow not ready, cannot load FaceNet model:", error);
            throw new Error("TensorFlow not ready for FaceNet model: " + error.message);
        }
        
        // Ensure sandbox is ready
        if (!sandboxManager || !sandboxManager.isReady) {
            console.error("[FaceOne " + new Date().toLocaleTimeString() + "] Sandbox not ready for FaceNet model loading");
            throw new Error("Sandbox not ready for FaceNet model loading");
        }
        
        console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Loading FaceNet model...");
        
        // Define model paths to try
        const modelPaths = [
            'models/FaceNet/Facenet512_tfjs_graph_model/model.json',
            'models/FaceNet/model.json',
            'models/facenet/model.json'
        ];
        
        // Verify each model path exists before trying to load it
        for (const modelPath of modelPaths) {
            const fullPath = chrome.runtime.getURL(modelPath);
            console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Checking if model exists at: ${fullPath}`);
            
            try {
                const response = await fetch(fullPath, { 
                    method: 'HEAD',
                    cache: 'no-cache' // Try without caching
                });
                
                if (response.ok) {
                    console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Model file verified at: ${fullPath}`);
                } else {
                    console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Model file not found at: ${fullPath}, status: ${response.status}`);
                    continue; // Skip to next path
                }
            } catch (error) {
                console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Error checking model at ${fullPath}: ${error.message}`);
                continue; // Skip to next path
            }
            
            // Try to load the model
            try {
                console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Attempting to load FaceNet model from: ${fullPath}`);
                
                // Send load model command to sandbox with longer timeout
                const response = await sandboxManager.sendMessage({
                    type: 'LOAD_MODEL',
                    modelPath: fullPath,
                    timeout: 120000 // Increase timeout to 120 seconds for model loading
                });
                
                if (!response || !response.success) {
                    console.error(`[FaceOne ${new Date().toLocaleTimeString()}] Failed to load model: ${response?.error || "Unknown error"}`);
                    continue; // Skip to next path
                }
                
                console.log(`[FaceOne ${new Date().toLocaleTimeString()}] FaceNet model loaded successfully from: ${modelPath}`);
                state.faceNetLoaded = true;
                return true;
            } catch (error) {
                console.error(`[FaceOne ${new Date().toLocaleTimeString()}] Error loading FaceNet model from ${modelPath}:`, error);
            }
        }
        
        // If we get here, no model was loaded successfully
        throw new Error("Failed to load FaceNet model from any path");
    } catch (error) {
        console.error("[FaceOne " + new Date().toLocaleTimeString() + "] Error loading FaceNet model:", error);
        state.faceNetError = error;
        throw error;
    } finally {
        state.faceNetLoading = false;
    }
}

/**
 * Verify that the model is ready by performing a test operation
 */
async function verifyModelReady() {
    console.log('Verifying model is ready...');
    
    // Create a small test image
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 160;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Fill with a simple pattern
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 160, 160);
    ctx.fillStyle = 'black';
    ctx.fillRect(60, 60, 40, 40); // Simple "face" pattern
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, 160, 160);
    
    // Try to generate an embedding
    try {
        const result = await sandboxManager.sendMessage({
            type: 'GENERATE_EMBEDDING',
            imageData: imageData.data
        }, 30000);
        
        if (!result.success) {
            throw new Error(`Test embedding generation failed: ${result.error}`);
        }
        
        console.log('Model verification successful');
        return true;
    } catch (error) {
        console.error('Model verification failed:', error);
        throw new Error('Model verification failed: ' + error.message);
    }
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
 * Load FaceAPI models
 */
async function loadFaceApiModels() {
    if (state.faceApiLoaded) {
        console.log("FaceAPI models already loaded");
        return;
    }

    if (state.faceApiLoading) {
        console.log("FaceAPI models already loading");
        return;
    }

    state.faceApiLoading = true;
    state.faceApiError = null;

    try {
        // First, ensure TensorFlow is ready
        try {
            await ensureTensorFlowReady();
            console.log("TensorFlow ready for FaceAPI models");
        } catch (error) {
            console.error("[FaceOne " + new Date().toLocaleTimeString() + "] TensorFlow not ready, continuing with FaceAPI models:", error);
        }

        console.log("Loading FaceAPI models...");
        
        // Get the base URL for the extension
        const baseUrl = chrome.runtime.getURL('');
        console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Extension base URL:", baseUrl);
        
        // Debug: print absolute URL for model directories to verify paths
        console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Absolute URL for models/FaceAPI:", chrome.runtime.getURL('models/FaceAPI'));
        console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Absolute URL for models/FaceAPI/ssd_mobilenetv1:", chrome.runtime.getURL('models/FaceAPI/ssd_mobilenetv1'));
        
        // Try to load directly from the known location of models with explicit paths
        try {
            // First verify the model files exist
            const ssdModelPath = 'models/FaceAPI/ssd_mobilenetv1/ssd_mobilenetv1_model-weights_manifest.json';
            const landmarkModelPath = 'models/face_landmark_68_model-weights_manifest.json';
            const recognitionModelPath = 'models/face_recognition_model-weights_manifest.json';
            
            console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Checking model paths directly...");
            
            // Check SSD model
            const ssdUrl = chrome.runtime.getURL(ssdModelPath);
            try {
                const ssdResponse = await fetch(ssdUrl, { method: 'HEAD' });
                console.log("[FaceOne " + new Date().toLocaleTimeString() + "] SSD model check:", ssdResponse.ok ? "OK" : "Not found");
            } catch (e) {
                console.error("[FaceOne " + new Date().toLocaleTimeString() + "] SSD model not found:", e);
            }
            
            // Check landmark model
            const landmarkUrl = chrome.runtime.getURL(landmarkModelPath);
            try {
                const landmarkResponse = await fetch(landmarkUrl, { method: 'HEAD' });
                console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Landmark model check:", landmarkResponse.ok ? "OK" : "Not found");
            } catch (e) {
                console.error("[FaceOne " + new Date().toLocaleTimeString() + "] Landmark model not found:", e);
            }
            
            // Check recognition model
            const recognitionUrl = chrome.runtime.getURL(recognitionModelPath);
            try {
                const recognitionResponse = await fetch(recognitionUrl, { method: 'HEAD' });
                console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Recognition model check:", recognitionResponse.ok ? "OK" : "Not found");
            } catch (e) {
                console.error("[FaceOne " + new Date().toLocaleTimeString() + "] Recognition model not found:", e);
            }
            
            // Now try loading models using direct URIs with models folder
            console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Loading models directly...");
            
            console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Loading ssdMobilenetv1...");
            await faceapi.nets.ssdMobilenetv1.loadFromUri('models');
            
            console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Loading faceLandmark68Net...");
            await faceapi.nets.faceLandmark68Net.loadFromUri('models');
            
            console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Loading faceRecognitionNet...");
            await faceapi.nets.faceRecognitionNet.loadFromUri('models');
            
            console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Successfully loaded models directly");
            state.faceApiLoaded = true;
            state.faceApiLoading = false;
            return;
        } catch (directError) {
            console.error("[FaceOne " + new Date().toLocaleTimeString() + "] Failed to load models directly:", directError);
        }
        
        // If direct loading failed, try the original approach with multiple paths
        // Define model paths to try
        const modelPaths = [
            'models/FaceAPI',
            'models',
            'models/face-api',
            'models/faceapi',
            'face-api/models',
            'faceapi/models',
            '' // Root directory
        ];
        
        // Try each path until one works
        let loaded = false;
        let lastError = null;
        
        for (const path of modelPaths) {
            try {
                const modelUrl = chrome.runtime.getURL(path);
                console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Trying model path:", modelUrl);
                
                // Try to load the models with crossOrigin set to anonymous
                console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Loading ssdMobilenetv1 from:", modelUrl);
                await faceapi.nets.ssdMobilenetv1.loadFromUri(modelUrl);
                
                console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Loading faceLandmark68Net from:", modelUrl);
                await faceapi.nets.faceLandmark68Net.loadFromUri(modelUrl);
                
                console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Loading faceRecognitionNet from:", modelUrl);
                await faceapi.nets.faceRecognitionNet.loadFromUri(modelUrl);
                
                console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Successfully loaded models from:", path);
                loaded = true;
                break;
            } catch (error) {
                console.error("[FaceOne " + new Date().toLocaleTimeString() + "] Failed to load models from " + path + ":", error);
                lastError = error;
            }
        }
        
        if (!loaded) {
            throw lastError || new Error("Failed to load models from any path");
        }
        
        state.faceApiLoaded = true;
        state.faceApiLoading = false;
        console.log("FaceAPI models loaded successfully");
    } catch (error) {
        state.faceApiLoading = false;
        state.faceApiError = error;
        console.error("Error loading models:", error);
        throw error;
    }
}

//=============================================================================
// 4. Sandbox Management
//=============================================================================

/**
 * Enhanced sandbox manager with health monitoring and recovery
 */
class SandboxManager {
    constructor() {
        debugLog('Creating new SandboxManager instance');
        this.iframe = null;
        this.isReady = false;
        this.pendingRequests = new Map();
        this.lastActivity = Date.now();
        this.healthCheckInterval = null;
        this.readyResolve = null;
        this.readyReject = null;
    }

    async initialize() {
        debugLog('Initializing SandboxManager');
        
        if (this.iframe) {
            debugLog('SandboxManager already has an iframe, destroying it first');
            this.destroy();
        }
        
        debugLog('Creating sandbox iframe');
        this.iframe = document.createElement('iframe');
        this.iframe.id = 'faceone-sandbox';
        this.iframe.style.display = 'none';
        
        // Ensure proper sandbox attributes
        this.iframe.setAttribute('sandbox', 'allow-scripts allow-same-origin');
        
        // Create a promise that will be resolved when the sandbox is ready
        const readyPromise = new Promise((resolve, reject) => {
            this.readyResolve = resolve;
            this.readyReject = reject;
            
            // Set a timeout for sandbox initialization
            setTimeout(() => {
                if (!this.isReady) {
                    const error = new Error('Sandbox initialization timed out');
                    debugLog('Sandbox initialization timed out');
                    reject(error);
                }
            }, 30000); // 30 second timeout
        });
        
        // Add load event listener to debug iframe loading issues
        this.iframe.addEventListener('load', () => {
            debugLog('Sandbox iframe loaded');
        });
        
        this.iframe.addEventListener('error', (error) => {
            debugLog('Error loading sandbox iframe:', error);
            if (this.readyReject) {
                this.readyReject(new Error('Failed to load sandbox iframe'));
            }
        });
        
        // Set the src after adding event listeners
        const sandboxUrl = chrome.runtime.getURL('sandbox.html');
        debugLog('Setting sandbox iframe src to:', sandboxUrl);
        this.iframe.src = sandboxUrl;
        
        // Add to document
        document.body.appendChild(this.iframe);
        
        // Set up message listener
        debugLog('Setting up message listener');
        window.addEventListener('message', this.handleMessage.bind(this));
        
        // Wait for sandbox to be ready
        debugLog('Waiting for sandbox to be ready');
        try {
            await readyPromise;
            debugLog('Sandbox is ready');
        } catch (error) {
            debugLog('Error waiting for sandbox to be ready:', error);
            throw error;
        }
        
        // Start health check
        debugLog('Starting health check');
        this.startHealthCheck();
        
        debugLog('SandboxManager initialization complete');
    }
    
    async restart() {
        debugLog('Attempting to restart sandbox');
        
        if (this.isRestarting) {
            debugLog('Restart already in progress, skipping');
            return;
        }
        
        this.isRestarting = true;
        this.restartAttempts++;
        
        try {
            debugLog(`Sandbox restart attempt ${this.restartAttempts}/${this.MAX_RESTART_ATTEMPTS}`);
            
            // Stop health check
            if (this.healthCheckInterval) {
                debugLog('Stopping health check interval');
                clearInterval(this.healthCheckInterval);
                this.healthCheckInterval = null;
            }
            
            // Destroy existing iframe
            debugLog('Destroying existing iframe');
            this.destroy();
            
            // Wait a moment before creating a new one
            debugLog('Waiting before creating new iframe');
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Create new iframe
            debugLog('Creating new iframe');
            this.iframe = document.createElement('iframe');
            this.iframe.id = 'faceone-sandbox';
            this.iframe.style.display = 'none';
            this.iframe.sandbox = 'allow-scripts allow-same-origin';
            this.iframe.src = chrome.runtime.getURL('sandbox.html');
            
            // Add to document
            document.body.appendChild(this.iframe);
            
            // Wait for sandbox to be ready
            debugLog('Waiting for sandbox to be ready');
            await this.waitForReady();
            
            // Reinitialize TensorFlow
            debugLog('Reinitializing TensorFlow');
            await this.sendMessage({ type: 'INIT' }, 30000);
            
            // Restart health check
            debugLog('Restarting health check');
            this.startHealthCheck();
            
            debugLog('Sandbox restart completed successfully');
            this.isRestarting = false;
            return true;
        } catch (error) {
            debugLog('Error during sandbox restart:', error);
            
            // If we've reached max attempts, give up
            if (this.restartAttempts >= this.MAX_RESTART_ATTEMPTS) {
                debugLog('Max restart attempts reached, giving up');
                this.isRestarting = false;
                throw new Error(`Failed to restart sandbox after ${this.MAX_RESTART_ATTEMPTS} attempts: ${error.message}`);
            }
            
            // Otherwise, try again
            debugLog('Trying restart again');
            this.isRestarting = false;
            return this.restart();
        }
    }
    
    async checkHealth() {
        // Skip if health check is already in progress
        if (this.healthCheckInProgress) {
            debugLog('Health check already in progress, skipping');
            return;
        }
        
        // Skip if we checked recently (within last 10 seconds)
        const now = Date.now();
        if (now - this.lastHealthCheck < 10000) {
            debugLog('Health check performed recently, skipping');
            return;
        }
        
        this.healthCheckInProgress = true;
        this.lastHealthCheck = now;
        
        debugLog('Performing sandbox health check...');
        
        try {
            if (!this.iframe || !this.iframe.contentWindow) {
                debugLog('Sandbox iframe not available, restarting');
                await this.restart();
                this.healthCheckInProgress = false;
                return;
            }
            
            const response = await this.sendMessage({ type: 'HEALTH_CHECK' }, 10000);
            debugLog('Health check response:', response);
            
            if (response.status !== 'ok') {
                debugLog('Health check failed, restarting sandbox');
                await this.restart();
            } else {
                debugLog('Health check passed');
            }
        } catch (error) {
            debugLog('Health check error:', error);
            
            // If we get a timeout or other error, restart the sandbox
            try {
                debugLog('Restarting sandbox due to health check error');
                await this.restart();
            } catch (restartError) {
                debugLog('Failed to restart sandbox:', restartError);
            }
        } finally {
            this.healthCheckInProgress = false;
        }
    }

    async waitForReady() {
        debugLog('Waiting for sandbox to be ready...');
        
        // Already ready
        if (this.isReady) {
            debugLog('Sandbox already ready');
            return;
        }
        
        // If we don't have an iframe, we can't wait for it to be ready
        if (!this.iframe) {
            debugLog('No iframe available, cannot wait for ready');
            throw new Error('No iframe available');
        }
        
        // Create a new promise that will be resolved when the sandbox is ready
        const readyPromise = new Promise((resolve, reject) => {
            debugLog('Setting up ready promise');
            
            // Ensure any existing readyResolve/readyReject are cleared
            this.readyResolve = resolve;
            this.readyReject = reject;
            
            // Set up a timeout
            const timeout = setTimeout(() => {
                debugLog('Sandbox ready timeout exceeded');
                reject(new Error('Timeout waiting for sandbox to be ready'));
            }, 30000); // 30 second timeout
            
            // Set up a one-time event listener for the ready message
            const readyHandler = (event) => {
                if (event.source === this.iframe.contentWindow && 
                    event.data && 
                    event.data.type === 'READY') {
                    
                    debugLog('Received READY message from sandbox');
                    clearTimeout(timeout);
                    window.removeEventListener('message', readyHandler);
                    this.isReady = true;
                    resolve();
                }
            };
            
            // Add the event listener
            window.addEventListener('message', readyHandler);
            
            // Send a ping to check if the sandbox is already ready
            if (this.iframe.contentWindow) {
                debugLog('Sending ping to sandbox');
                try {
                    this.iframe.contentWindow.postMessage({ type: 'PING' }, '*');
                } catch (error) {
                    debugLog('Error sending ping to sandbox:', error);
                    // Don't reject here, the timeout will handle it
                }
            }
        });
        
        // Wait for the promise to resolve or reject
        await readyPromise;
        debugLog('Sandbox is now ready');
    }

    startHealthCheck() {
        // Clear any existing interval first
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }
        
        // Set up health check interval (less frequent)
        this.healthCheckInterval = setInterval(() => {
            this.checkHealth();
        }, 300000); // Check every 5 minutes instead of every 30 seconds
    }

    async loadModels() {
        // Implementation depends on your model loading logic
        // This is a placeholder
        const faceNetModelPath = chrome.runtime.getURL('models/FaceNet/model.json');
        await this.sendMessage({ 
            type: 'LOAD_MODEL',
            modelPath: faceNetModelPath
        });
    }

    /**
     * Send a message to the sandbox and wait for a response
     * @param {Object} message - The message to send
     * @param {number} timeout - Timeout in milliseconds
     * @returns {Promise<Object>} - The response from the sandbox
     */
    sendMessage(message, timeout = 30000) {
        // Initialize pendingRequests if it doesn't exist
        if (!this.pendingRequests) {
            this.pendingRequests = new Map();
        }
        
        // Update last activity timestamp
        this.lastActivity = Date.now();
        
        return new Promise((resolve, reject) => {
            try {
                // Check if sandbox is ready
                if (!this.iframe || !this.iframe.contentWindow) {
                    return reject(new Error('Sandbox frame not ready'));
                }
                
                // Generate a unique ID for this request
                const requestId = 'req_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                
                // Add request ID to message
                const messageWithId = {
                    ...message,
                    requestId
                };
                
                // Set up timeout
                const timeoutId = setTimeout(() => {
                    // Remove from pending requests
                    if (this.pendingRequests.has(requestId)) {
                        this.pendingRequests.delete(requestId);
                    }
                    
                    // Reject with timeout error
                    reject(new Error(`Sandbox request timeout for ${message.type}`));
                }, timeout);
                
                // Store in pending requests
                this.pendingRequests.set(requestId, {
                    resolve,
                    reject,
                    timeoutId,
                    message: messageWithId,
                    timestamp: Date.now()
                });
                
                // Send message to sandbox
                this.iframe.contentWindow.postMessage(messageWithId, '*');
                
                // Log message sent
                console.log(`[SandboxManager] Sent message: ${message.type} (ID: ${requestId})`);
            } catch (error) {
                console.error('[SandboxManager] Error sending message:', error);
                reject(error);
            }
        });
    }

    /**
     * Handle messages from the sandbox
     * @param {MessageEvent} event - The message event
     */
    handleMessage(event) {
        try {
            // Ensure event.data exists
            if (!event.data) {
                return;
            }
            
            const data = event.data;
            
            // Update last activity timestamp
            this.lastActivity = Date.now();
            
            // Handle ready message from sandbox
            if (data.type === 'SANDBOX_READY' || data.type === 'READY') {
                console.log('[SandboxManager] Sandbox reported ready');
                this.isReady = true;
                
                // Resolve the ready promise if it exists
                if (this.readyResolve) {
                    this.readyResolve();
                    this.readyResolve = null;
                    this.readyReject = null;
                }
                return;
            }
            
            // Handle response messages - Support both RESPONSE and other message types like TF_STATUS
            if (data.requestId || (data.type && data.type.includes('_RESPONSE')) || data.type === 'TF_STATUS') {
                // Initialize pendingRequests if it doesn't exist
                if (!this.pendingRequests) {
                    this.pendingRequests = new Map();
                }
                
                // Find the pending request by requestId or infer from response type
                let requestId = data.requestId;
                
                // If no requestId but it's a response to a known request type
                if (!requestId && data.type) {
                    // Try to find a pending request with a matching request type
                    for (const [id, req] of this.pendingRequests.entries()) {
                        if (req.message && req.message.type && 
                            data.type === req.message.type + '_RESPONSE' || 
                            (req.message.type === 'GET_TF_STATUS' && data.type === 'TF_STATUS')) {
                            requestId = id;
                            break;
                        }
                    }
                }
                
                // Find the pending request
                const request = requestId ? this.pendingRequests.get(requestId) : null;
                
                if (request) {
                    // Clear the timeout
                    if (request.timeoutId) {
                        clearTimeout(request.timeoutId);
                    }
                    
                    // Remove from pending requests
                    this.pendingRequests.delete(requestId);
                    
                    // Resolve or reject based on response
                    if (data.success === true || data.type === 'TF_STATUS' || data.type === 'HEALTH_CHECK_RESPONSE') {
                        request.resolve(data);
                    } else {
                        // If the response has an error property, use it
                        const errorMessage = data.error || 'Unknown error in sandbox response';
                        request.reject(new Error(errorMessage));
                    }
                } else if (data.requestId) {
                    console.warn(`[SandboxManager] Received response for unknown request: ${data.requestId}`);
                }
                return;
            }
            
            // Handle other message types
            console.log(`[SandboxManager] Received message: ${data.type}`);
        } catch (error) {
            console.error('[SandboxManager] Error handling message:', error);
        }
    }

    destroy() {
        // Clear interval
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
            this.healthCheckInterval = null;
        }

        // Clear all pending requests
        this.pendingRequests.forEach(request => {
            clearTimeout(request.timeoutId);
            request.reject(new Error('Sandbox destroyed'));
        });
        this.pendingRequests.clear();

        // Remove iframe
        if (this.iframe) {
            document.body.removeChild(this.iframe);
            this.iframe = null;
        }

        this.isReady = false;
    }
}

// Initialize the sandboxManager if it doesn't exist yet
if (!sandboxManager) {
    sandboxManager = new SandboxManager();
}

//=============================================================================
// 5. Image Processing
//=============================================================================

/**
 * Process an image with memory-aware scheduling and progressive processing
 * @param {HTMLImageElement} img - The image to process
 * @returns {Promise<void>}
 */
async function processImage(img) {
    // Skip if already processed or processing
    if (img.dataset.processed === 'true' || img.dataset.processing === 'true') {
        return;
    }
    
    // Skip if image is too small
    if (img.width < flagShowFrameonImage.minimumImageSize || 
        img.height < flagShowFrameonImage.minimumImageSize) {
        img.dataset.processed = 'true';
        img.dataset.reason = 'too_small';
        return;
    }
    
    // Mark as processing
    img.dataset.processing = 'true';
    
    try {
        // Use progressive processing for large images
        if (flagShowFrameonImage.progressiveProcessing && 
            Math.max(img.width, img.height) > 1024) {
            await processImageProgressively(img);
        } else {
            await processImageStandard(img);
        }
        
        // Increment processed images counter
        incrementProcessedImagesCount();
        
        // Mark as processed
        img.dataset.processed = 'true';
        img.dataset.processing = 'false';
        
        // Apply visual indicator if enabled
        if (flagShowFrameonImage.frameProsessedImage) {
            addProcessedFrame(img);
        }
    } catch (error) {
        console.error(`Error processing image (${img.src.slice(0, 50)}...):`, error);
        img.dataset.processing = 'false';
        img.dataset.error = error.message;
    }
}

/**
 * Process an image progressively (low-res first, then high-res)
 * @param {HTMLImageElement} img - The image to process
 * @returns {Promise<void>}
 */
async function processImageProgressively(img) {
    // First pass: low resolution for quick results
    const maxDimension = Math.max(img.width, img.height);
    const scale = 512 / maxDimension;
    
    // Get scaled image data
    const lowResData = await getScaledImageData(img, scale);
    
    // Detect faces at low resolution
    const lowResDetections = await detectFaces(lowResData);
    
    // Show preliminary results if faces found
    if (lowResDetections.length > 0) {
        visualizeResults(img, lowResDetections, { preliminary: true });
    }
    
    // Second pass: full resolution for accuracy (only if faces detected)
    if (lowResDetections.length > 0) {
        const fullResData = await getImageData(img);
        const fullResDetections = await detectFaces(fullResData);
        
        // Update with final results
        visualizeResults(img, fullResDetections, { preliminary: false });
        
        // Apply actions based on mode
        if (flagShowFrameonImage.processingMode === 'blur_mode') {
            applyBlurToFaces(img, fullResDetections);
        }
    }
}

/**
 * Process an image using standard approach
 * @param {HTMLImageElement} img - The image to process
 * @returns {Promise<void>}
 */
async function processImageStandard(img) {
    // Get image data
    const imageData = await getImageData(img);
    
    // Detect faces
    const detections = await detectFaces(imageData);
    
    // Visualize results
    visualizeResults(img, detections, { preliminary: false });
    
    // Apply actions based on mode
    if (flagShowFrameonImage.processingMode === 'blur_mode') {
        applyBlurToFaces(img, detections);
    }
}

/**
 * Get image data at a specific scale
 * @param {HTMLImageElement} img - The image to process
 * @param {number} scale - Scale factor (0-1)
 * @returns {ImageData} Scaled image data
 */
async function getScaledImageData(img, scale) {
    const canvas = document.createElement('canvas');
    // Set willReadFrequently to true for better performance with multiple readbacks
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Set canvas dimensions to scaled size
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    
    // Draw image at scaled size
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    
    // Get image data
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/**
 * Detect faces in an image using the sandbox
 * @param {ImageData} imageData - The image data to process
 * @returns {Promise<Array>} Array of face detections
 */
async function detectFaces(imageData) {
    try {
        // Check if sandbox is ready
        if (!sandboxManager.isReady) {
            await sandboxManager.initialize();
        }
        
        // Send message to sandbox for face detection
        const response = await sandboxManager.sendMessage({
            type: 'DETECT_FACES',
            imageData: imageData
        });
        
        if (!response.success) {
            throw new Error(response.error || 'Face detection failed');
        }
        
        return response.detections;
    } catch (error) {
        console.error('Face detection error:', error);
        throw error;
    }
}

/**
 * Generate embedding for a face
 * @param {ImageData} faceData - The face image data
 * @returns {Promise<Array>} Face embedding vector
 */
async function generateEmbedding(faceData) {
    try {
        // Check if sandbox is ready
        if (!sandboxManager.isReady) {
            await sandboxManager.initialize();
        }
        
        // Send message to sandbox for embedding generation
        const response = await sandboxManager.sendMessage({
            type: 'GENERATE_EMBEDDING',
            imageData: faceData
        });
        
        if (!response.success) {
            throw new Error(response.error || 'Embedding generation failed');
        }
        
        return response.embedding;
    } catch (error) {
        console.error('Embedding generation error:', error);
        throw error;
    }
}

//=============================================================================
// 6. Initialization
//=============================================================================

/**
 * Load user settings from chrome.storage
 * @returns {Promise<Object>} The loaded settings
 */
async function loadSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get({
      // Default settings
      enabled: true,
      threshold: 0.6,
      mode: 'highlight',
      blurIntensity: 10,
      showLabels: true,
      processAll: false,
      maxImagesPerPage: 50,
      progressiveProcessing: true,
      memoryManagement: true
    }, (settings) => {
      state.settings = settings;
      console.log('Settings loaded:', settings);
      resolve(settings);
    });
  });
}

// Function to check if models are in the root directory
async function checkModelsInRootDirectory() {
    console.log("Checking if models are in the root directory...");
    
    // List of model files that might be in the root directory
    const modelFiles = [
        'face_recognition_model-weights_manifest.json',
        'face_landmark_68_model-weights_manifest.json',
        'ssd_mobilenetv1_model-weights_manifest.json'
    ];
    
    // Check if any of these files exist in the root directory
    let foundInRoot = false;
    for (const file of modelFiles) {
        try {
            const fileUrl = chrome.runtime.getURL(file);
            const response = await fetch(fileUrl, { method: 'HEAD' });
            if (response.ok) {
                console.log(`Found model file in root directory: ${file}`);
                foundInRoot = true;
            }
        } catch (error) {
            // Continue to next file
        }
    }
    
    if (foundInRoot) {
        console.warn("Models found in root directory. Please move them to the 'models' directory for better organization.");
        return true;
    }
    
    return false;
}

/**
 * Initialize the extension
 */
async function initialize() {
    // Check if already initialized or initializing
    if (initializationComplete) {
        console.log("FaceOne extension already initialized");
        return;
    }

    if (isInitializing) {
        console.log("FaceOne extension already initializing");
        return;
    }

    // Check for refresh loop
    if (checkRefreshLoop()) {
        console.error("Detected refresh loop, disabling extension initialization");
        return;
    }

    // Set initialization flags
    debugLog("Setting initialization flags");
    isInitializing = true;
    initializationError = null;
    initializationComplete = false;

    try {
        // Load settings
        debugLog("Loading settings");
        await loadSettings();

        // Create sandbox frame
        debugLog("Creating sandbox frame");
        try {
            await createSandboxFrame(60000); // Increase timeout to 60 seconds
            debugLog("Sandbox frame created successfully");
        } catch (error) {
            // If sandbox creation fails, try again once more
            debugLog("Error creating sandbox frame:", error);
            debugLog("Retrying sandbox frame creation...");
            await createSandboxFrame(90000); // Even longer timeout on retry
        }

        // Load FaceAPI models
        debugLog("Loading FaceAPI models");
        try {
            await loadFaceApiModels();
        } catch (error) {
            debugLog("Error loading FaceAPI models:", error);
            
            // Try recovery method
            debugLog("Attempting model loading recovery...");
            try {
                await recoverModelLoading();
                debugLog("Model loading recovery successful");
            } catch (recoveryError) {
                debugLog("Model loading recovery failed:", recoveryError);
                throw new Error(`Failed to load models: ${error.message}. Recovery also failed: ${recoveryError.message}`);
            }
        }
        
        // Continue with the rest of initialization...
        // Load FaceNet model
        debugLog("Loading FaceNet model");
        try {
            await loadFaceNetModel();
        } catch (error) {
            debugLog("Error loading FaceNet model:", error);
            console.log("Model status at error:", getModelLoadingStatus());
            // Continue anyway since we can still use FaceAPI models
            debugLog("Continuing without FaceNet model");
        }
        
        // Start processing existing images
        processExistingImages();
        
        // Start observing for new images
        observeElements();
        
        // Start document observer
        startDocumentObserver();
        
        // Set up periodic resource verification (every 5 minutes instead of every minute)
        setInterval(verifyResources, 5 * 60 * 1000);
        
        // Mark initialization as complete
        initializationComplete = true;
        debugLog("FaceOne extension initialized successfully");
        
        // Reset initialization flags
        isInitializing = false;
        
        return true;
    } catch (error) {
        console.error("Initialization error:", error);
        
        // Set initialization error
        initializationError = error;
        
        // Show notification
        showNotification(`Initialization error: ${error.message}`);
        
        // Reset initialization flags
        isInitializing = false;
        
        // Throw the error up
        throw error;
    }
}

/**
 * Process existing images on the page
 */
function processExistingImages() {
    const images = document.querySelectorAll('img:not([data-processed]):not([data-processing])');
    
    // Limit number of images to process
    const imagesToProcess = Array.from(images).slice(0, flagShowFrameonImage.maxImagesPerPage);
    
    console.log(`Processing ${imagesToProcess.length} existing images...`);
    
    // Process images with a small delay between each to avoid freezing the page
    imagesToProcess.forEach((img, index) => {
        setTimeout(() => {
            if (img.complete && img.naturalWidth > 0) {
                processImage(img).catch(console.error);
            } else {
                img.onload = () => processImage(img).catch(console.error);
            }
        }, index * 100); // 100ms delay between each image
    });
}

/**
 * Show a notification to the user
 * @param {string} message - The message to display
 */
function showNotification(message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        z-index: 10000;
        font-family: Arial, sans-serif;
        font-size: 14px;
        max-width: 300px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    `;
    notification.textContent = message;
    
    // Add to document
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transition = 'opacity 0.5s';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 500);
    }, 5000);
}

// Initialize on document ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}

//=============================================================================
// 7. UI Management
//=============================================================================

// Optimize queue processing with batching and prioritization
// processingQueue is now defined at the top of the file
// const processingQueue = {
//     queue: [],
//     paused: false,
//     
//     pause() {
//         this.paused = true;
//         console.log('Processing queue paused');
//     },
//     
//     resume() {
//         this.paused = false;
//         console.log('Processing queue resumed');
//         
//         // If we have items in the queue, start processing
//         if (this.queue.length > 0) {
//             this.process();
//         }
//     },
//     
//     add(element, priority = false) {
//         if (!element) return;
//         
//         // Skip if already in queue
//         if (this.queue.some(item => item.element === element)) {
//             return;
//         }
//         
//         // Add to queue with priority flag
//         const item = { element, priority };
//         
//         if (priority) {
//             this.queue.unshift(item); // Add to front if priority
//         } else {
//             this.queue.push(item); // Add to end otherwise
//         }
//         
//         // Start processing if not paused
//         if (!this.paused) {
//             this.process();
//         }
//     },
//     
//     async process() {
//         // Skip if paused or already processing
//         if (this.paused || state.isProcessing) return;
//         
//         // Skip if queue is empty
//         if (this.queue.length === 0) return;
//         
//         // Set processing flag
//         state.isProcessing = true;
//         
//         try {
//             // Get next item from queue
//             const item = this.queue.shift();
//             
//             // Process the item
//             await handleVisibleElement(item.element);
//         } catch (error) {
//             console.error('Error processing queue item:', error);
//         } finally {
//             // Reset processing flag
//             state.isProcessing = false;
//             
//             // Continue processing if items remain and not paused
//             if (this.queue.length > 0 && !this.paused) {
//                 // Use setTimeout to avoid blocking the main thread
//                 setTimeout(() => this.process(), 10);
//             }
//         }
//     }
// };

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
            
            // Insert wrapper next to original SVG - with null check
            if (svgParent && svgParent.parentNode) {
                svgParent.parentNode.insertBefore(wrapper, svgParent.nextSibling);
                
                // Don't hide original immediately
                element.style.opacity = '0.01';
            } else {
                console.log("[FaceOne] Could not insert wrapper: parent node not found");
            }
        } else {
            // Regular SVG image handling
            wrapper.style.position = 'relative';
            wrapper.style.display = 'inline-block';
            imgElement.style.width = '100%';
            imgElement.style.height = '100%';
            imgElement.style.objectFit = 'contain';
            
            // Add null check before inserting
            if (element && element.parentNode) {
                element.parentNode.insertBefore(wrapper, element);
            } else {
                console.log("[FaceOne] Could not insert wrapper: parent node not found");
                return null; // Return null to indicate failure
            }
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
    
    // Add null check before inserting
    if (element && element.parentNode) {
        element.parentNode.insertBefore(wrapper, element);
        wrapper.appendChild(element);
    } else {
        console.log("[FaceOne] Could not insert wrapper: parent node not found");
        return null; // Return null to indicate failure
    }
    
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

// Check if we're in an infinite refresh loop
function checkRefreshLoop() {
    try {
        // First check if the extension is already disabled
        if (isExtensionDisabled()) {
            return true;
        }
        
        // Get current refresh data from session storage
        const refreshData = sessionStorage.getItem(PAGE_REFRESH_KEY);
        let refreshCount = 0;
        let lastRefresh = 0;
        
        if (refreshData) {
            const data = JSON.parse(refreshData);
            refreshCount = data.count || 0;
            lastRefresh = data.timestamp || 0;
        }
        
        const now = Date.now();
        
        // Check if we're within the cooldown period
        if (now - lastRefresh < REFRESH_COOLDOWN) {
            // Increment refresh count
            refreshCount++;
            
            // Log refresh count
            console.log(`Page refresh detected (${refreshCount}/${MAX_REFRESHES})`);
            
            // Update session storage
            sessionStorage.setItem(PAGE_REFRESH_KEY, JSON.stringify({
                count: refreshCount,
                timestamp: now
            }));
            
            // Check if we've exceeded the maximum number of refreshes
            if (refreshCount > MAX_REFRESHES) {
                // Set a flag to disable the extension for a longer period (30 minutes)
                const disableUntil = now + 30 * 60 * 1000; // 30 minutes
                localStorage.setItem('faceone_disabled_until', disableUntil.toString());
                
                console.error(`Refresh loop detected (${refreshCount} refreshes in ${REFRESH_COOLDOWN/1000}s). Extension disabled for 30 minutes.`);
                return true;
            }
            
            // Check if we should permanently disable the extension
            if (refreshCount > MAX_REFRESHES * 3) {
                console.error(`Severe refresh loop detected (${refreshCount} refreshes). Extension permanently disabled.`);
                localStorage.setItem('faceone_permanently_disabled', 'true');
                return true;
            }
        } else {
            // Reset refresh count if outside cooldown period
            sessionStorage.setItem(PAGE_REFRESH_KEY, JSON.stringify({
                count: 1,
                timestamp: now
            }));
        }
        
        return false;
    } catch (error) {
        console.warn(`Error checking refresh loop:`, error);
        return false;
    }
}

// Reset the refresh loop counter
function resetRefreshLoop() {
    try {
        // Clear session storage refresh counter
        sessionStorage.removeItem(PAGE_REFRESH_KEY);
        
        // Clear all disable flags
        localStorage.removeItem('faceone_disabled_until');
        localStorage.removeItem('faceone_permanently_disabled');
        localStorage.removeItem('faceone_permanent_disable'); // Old key for backward compatibility
        
        console.log('[FaceOne] Refresh loop counter and disable flags reset');
        
        // Force re-initialization if extension was previously disabled
        // Only check initializationComplete if it's defined
        if (typeof initializationComplete !== 'undefined' && !initializationComplete) {
            console.log('[FaceOne] Re-initializing extension after reset');
            
            // Make sure initialize is defined before trying to call it
            if (typeof initialize === 'function') {
                setTimeout(() => {
                    initialize().catch(error => {
                        console.error('[FaceOne] Re-initialization failed:', error);
                    });
                }, 1000);
            } else {
                console.log('[FaceOne] Cannot re-initialize: initialize function not available yet');
            }
        }
    } catch (e) {
        console.warn('[FaceOne] Error resetting refresh loop:', e);
    }
}

// Add a message listener for commands from the popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'resetRefreshLoop') {
        resetRefreshLoop();
        sendResponse({ success: true, message: 'Refresh loop counter reset' });
        return true;
    }
    
    if (message.action === 'getStatus') {
        // Get current status
        const status = {
            initialized: typeof initializationComplete !== 'undefined' && initializationComplete,
            initializing: typeof isInitializing !== 'undefined' && isInitializing,
            attempts: typeof initializationAttempts !== 'undefined' ? initializationAttempts : 0,
            sandboxReady: sandboxManager && sandboxManager.isReady,
            refreshLoopActive: checkRefreshLoop()
        };
        
        sendResponse({ success: true, status });
        return true;
    }
    
    // Add handler for GET_MEMORY_STATS
    if (message.type === 'GET_MEMORY_STATS') {
        try {
            // Get model loading status
            const modelStatus = getModelLoadingStatus();
            
            // Get memory stats from tf if available
            let memoryStats = {
                numTensors: 0,
                memoryMB: 0,
                imagesProcessed: imagesProcessed || 0,
                cacheSize: 0
            };
            
            // Try to get stats from sandbox if available
            if (sandboxManager && sandboxManager.isReady) {
                sandboxManager.sendMessage({ type: 'GET_TF_STATUS' }, 5000)
                    .then(response => {
                        if (response && response.status && response.status.memory) {
                            const memory = response.status.memory;
                            memoryStats.numTensors = memory.numTensors || 0;
                            memoryStats.memoryMB = Math.round((memory.numBytes || 0) / (1024 * 1024));
                        }
                        
                        sendResponse({ 
                            success: true, 
                            stats: memoryStats,
                            modelStatus: modelStatus
                        });
                    })
                    .catch(error => {
                        console.error('Error getting memory stats from sandbox:', error);
                        sendResponse({ 
                            success: true, 
                            stats: memoryStats,
                            modelStatus: modelStatus,
                            error: 'Failed to get detailed stats from sandbox'
                        });
                    });
                
                // Return true to indicate we'll send the response asynchronously
                return true;
            } else {
                // No sandbox available, return basic stats
                sendResponse({ 
                    success: true, 
                    stats: memoryStats,
                    modelStatus: modelStatus,
                    sandboxStatus: 'Not available'
                });
                return true;
            }
        } catch (error) {
            console.error('Error getting memory stats:', error);
            sendResponse({ 
                success: false, 
                error: 'Error retrieving memory stats: ' + error.message
            });
            return true;
        }
    }
    
    return false;
});

/**
 * Computes similarity between two face embeddings
 * @param {Float32Array} embedding1 - First face embedding
 * @param {Float32Array} embedding2 - Second face embedding
 * @returns {Promise<number>} Similarity score between 0 and 1
 */
async function computeFaceSimilarity(embedding1, embedding2) {
    try {
        // Ensure sandbox manager exists
        ensureSandboxManager();
        
        // Check if sandbox is ready
        if (!sandboxManager || !sandboxManager.isReady) {
            throw new Error('Sandbox not ready for similarity computation');
        }
        
        // Validate embeddings
        if (!embedding1 || !embedding2 || !Array.isArray(embedding1) || !Array.isArray(embedding2)) {
            throw new Error('Invalid embeddings provided for similarity computation');
        }
        
        // Send compute request to sandbox
        const response = await sandboxManager.sendMessage({
            type: 'COMPUTE_SIMILARITY',
            embedding1,
            embedding2
        }, 10000); // 10 second timeout
        
        if (!response || !response.success) {
            throw new Error(response?.error || 'Failed to compute similarity');
        }
        
        return response.similarity;
    } catch (error) {
        console.error('Error computing face similarity:', error);
        throw error;
    }
}

// Add function to ensure TensorFlow is ready
async function ensureTensorFlowReady() {
    debugLog('Ensuring TensorFlow is ready');
    
    // Check if sandbox is ready
    if (!sandboxManager || !sandboxManager.isReady) {
        throw new Error('Sandbox not ready for TensorFlow check');
    }
    
    // Check TensorFlow status
    const status = await checkTensorFlowStatus();
    
    if (!status || !status.isInitialized) {
        throw new Error('TensorFlow not properly initialized');
    }
    
    debugLog('TensorFlow is ready');
    return true;
}

// Add function to check TensorFlow status
async function checkTensorFlowStatus() {
    debugLog('Checking TensorFlow status');
    
    // Check if sandbox is ready
    if (!sandboxManager || !sandboxManager.isReady) {
        throw new Error('Sandbox not ready for TensorFlow status check');
    }
    
    try {
        debugLog('Sending GET_TF_STATUS message to sandbox');
        
        // Send status request to sandbox with a longer timeout
        const response = await sandboxManager.sendMessage({
            type: 'GET_TF_STATUS'
        }, 20000); // Increased to 20 second timeout
        
        if (!response) {
            debugLog('No response received from sandbox for TensorFlow status');
            throw new Error('No response received from sandbox for TensorFlow status');
        }
        
        if (!response.success) {
            debugLog('Error in response from sandbox:', response.error);
            throw new Error(response.error || 'Failed to get TensorFlow status');
        }
        
        debugLog('TensorFlow status:', response.status);
        return response.status;
    } catch (error) {
        debugLog('Error checking TensorFlow status:', error);
        
        // If it's a timeout, provide more detailed error
        if (error.message && error.message.includes('timeout')) {
            debugLog('TensorFlow status check timed out, sandbox may be stuck');
            throw new Error('Sandbox request timeout for GET_TF_STATUS - try reloading the page');
        }
        
        throw error;
    }
}

/**
 * Verify that all required resources are available and ready
 * @returns {Promise<boolean>} True if all resources are ready
 */
async function verifyResources() {
    console.log("Verifying resources...");
    
    try {
        // Check if models directory exists
        const modelsExist = await checkDirectoryExists('models');
        if (!modelsExist) {
            console.warn("Models directory not found. Resources verification failed.");
            return false;
        }
        
        // Check sandbox status
        let sandboxReady = false;
        if (sandboxManager && sandboxManager.isReady) {
            console.log("Sandbox is ready");
            sandboxReady = true;
        } else {
            console.warn("Sandbox is not ready");
        }
        
        // Check TensorFlow status
        let tensorflowReady = false;
        if (sandboxReady) {
            try {
                await checkTensorFlowStatus();
                console.log("TensorFlow is ready");
                tensorflowReady = true;
            } catch (error) {
                console.warn("TensorFlow is not ready:", error);
            }
        }
        
        // Check model loading status
        const modelStatus = getModelLoadingStatus();
        console.log("Model loading status:", modelStatus);
        
        // Check if all required resources are ready
        const allReady = sandboxReady && 
                         tensorflowReady && 
                         modelStatus.faceApi.loaded && 
                         modelStatus.faceNet.loaded;
        
        console.log("Resources verification result:", allReady ? "SUCCESS" : "FAILED");
        return allReady;
    } catch (error) {
        console.error("Error verifying resources:", error);
        return false;
    }
}

/**
 * Check if a directory exists
 * @param {string} path - The path to check
 * @returns {Promise<boolean>} - True if the directory exists
 */
async function checkDirectoryExists(path) {
    try {
        console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Checking if directory exists: ${path}`);
        
        // For Chrome extensions, we can't directly check if a directory exists
        // Instead, we'll check for specific files that should be in the directory
        
        // Don't try to access the directory directly - Chrome doesn't allow this
        // Instead, check for specific model files
        
        // For models directory, check for specific model files
        if (path === 'models') {
            // Define files to check for
            const files = [
                // Face recognition model
                'models/face_recognition_model-weights_manifest.json',
                // Face landmark model
                'models/face_landmark_68_model-weights_manifest.json',
                // SSD Mobilenet (if inside FaceAPI directory)
                'models/FaceAPI/ssd_mobilenetv1/ssd_mobilenetv1_model-weights_manifest.json'
            ];
            
            // Check each file
            for (const file of files) {
                try {
                    const fileUrl = chrome.runtime.getURL(file);
                    console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Testing model file: ${fileUrl}`);
                    
                    const response = await fetch(fileUrl, { 
                        method: 'HEAD',
                        cache: 'no-cache'
                    });
                    
                    if (response.ok) {
                        console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Found model file: ${file.split('/').pop()}`);
                        return true;
                    }
                } catch (error) {
                    console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Failed to check model file ${file}: ${error.message}`);
                    // Continue checking other files
                }
            }
            
            // Check for files in root directory for backward compatibility
            const rootFiles = [
                'face_recognition_model-weights_manifest.json',
                'face_landmark_68_model-weights_manifest.json'
            ];
            
            for (const file of rootFiles) {
                try {
                    const fileUrl = chrome.runtime.getURL(file);
                    console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Testing root model file: ${fileUrl}`);
                    
                    const response = await fetch(fileUrl, { 
                        method: 'HEAD',
                        cache: 'no-cache'
                    });
                    
                    if (response.ok) {
                        console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Found model file in root: ${file}`);
                        return true;
                    }
                } catch (error) {
                    console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Failed to check root model file ${file}: ${error.message}`);
                }
            }
            
            console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Could not find any model files`);
            return false;
        }
        
        // For other directories, use a simpler approach - check a known file
        const testFiles = {
            'models/FaceAPI': 'models/FaceAPI/ssd_mobilenetv1/ssd_mobilenetv1_model-weights_manifest.json',
            'models/FaceNet': 'models/FaceNet/Facenet512_tfjs_graph_model/model.json',
            'lib': 'lib/face-api.min.js',
            'js': 'js/content.js',
            'css': 'css/styles.css'
        };
        
        const testFile = testFiles[path];
        if (testFile) {
            try {
                const fileUrl = chrome.runtime.getURL(testFile);
                console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Testing directory via file: ${fileUrl}`);
                
                const response = await fetch(fileUrl, { 
                    method: 'HEAD',
                    cache: 'no-cache'
                });
                
                if (response.ok) {
                    console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Directory exists (verified via file): ${path}`);
                    return true;
                }
            } catch (error) {
                console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Failed to verify directory via file: ${error.message}`);
            }
        }
        
        console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Directory does not exist: ${path}`);
        return false;
    } catch (error) {
        console.warn(`[FaceOne ${new Date().toLocaleTimeString()}] Error checking if directory exists: ${path}`, error);
        // Return false on error, but don't fail completely
        return false;
    }
}

// Add helper function to get model loading status
function getModelLoadingStatus() {
    return {
        faceApi: {
            loaded: typeof state !== 'undefined' && state.faceApiLoaded,
            loading: typeof state !== 'undefined' && state.faceApiLoading,
            error: typeof state !== 'undefined' ? state.faceApiError : null
        },
        faceNet: {
            loaded: typeof state !== 'undefined' && state.faceNetLoaded,
            loading: typeof state !== 'undefined' && state.faceNetLoading,
            error: typeof state !== 'undefined' ? state.faceNetError : null
        },
        myModel: {
            loaded: typeof state !== 'undefined' && state.myModelLoaded,
            loading: typeof state !== 'undefined' && state.myModelLoading,
            error: typeof state !== 'undefined' ? state.myModelError : null
        }
    };
}

/**
 * Attempt to recover from model loading failures by using direct paths and explicit loading
 */
async function recoverModelLoading() {
    console.log("[FaceOne " + new Date().toLocaleTimeString() + "] Attempting to recover model loading...");
    
    try {
        // First, check if the model files exist in root or models directory
        const modelLocations = [
            // Root directory files
            {
                path: '',
                files: [
                    'face_recognition_model-weights_manifest.json',
                    'face_recognition_model-shard1',
                    'face_landmark_68_model-weights_manifest.json',
                    'face_landmark_68_model-shard1'
                ]
            },
            // Models directory files
            {
                path: 'models/',
                files: [
                    'face_recognition_model-weights_manifest.json',
                    'face_recognition_model-shard1',
                    'face_landmark_68_model-weights_manifest.json',
                    'face_landmark_68_model-shard1'
                ]
            },
            // FaceAPI directory files
            {
                path: 'models/FaceAPI/',
                files: [
                    'ssd_mobilenetv1/ssd_mobilenetv1_model-weights_manifest.json',
                    'ssd_mobilenetv1/ssd_mobilenetv1_model-shard1',
                    'ssd_mobilenetv1/ssd_mobilenetv1_model-shard2'
                ]
            }
        ];
        
        // Check each location to find where model files are stored
        let modelPath = '';
        let filesFound = false;
        
        for (const location of modelLocations) {
            console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Checking for models in: ${location.path || 'root directory'}`);
            
            let allFilesExist = true;
            for (const file of location.files) {
                try {
                    const url = chrome.runtime.getURL(location.path + file);
                    console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Checking: ${url}`);
                    
                    const response = await fetch(url, { 
                        method: 'HEAD', 
                        cache: 'no-cache' 
                    });
                    
                    if (!response.ok) {
                        allFilesExist = false;
                        console.log(`[FaceOne ${new Date().toLocaleTimeString()}] File not found: ${file}`);
                        break;
                    }
                } catch (error) {
                    allFilesExist = false;
                    console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Error checking file: ${error.message}`);
                    break;
                }
            }
            
            if (allFilesExist) {
                modelPath = location.path;
                filesFound = true;
                console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Found all model files in: ${modelPath || 'root directory'}`);
                break;
            }
        }
        
        if (!filesFound) {
            console.error(`[FaceOne ${new Date().toLocaleTimeString()}] Could not find model files in any location`);
            throw new Error("Could not find model files");
        }
        
        // Now try loading models with direct paths
        console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Loading models from: ${modelPath || 'root directory'}`);
        
        try {
            // Use explicit model loading with the correct path
            const baseModelPath = chrome.runtime.getURL(modelPath);
            
            // Modify the way we load the models to use direct file URLs
            console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Loading face detection model...`);
            await faceapi.nets.ssdMobilenetv1.load(baseModelPath);
            
            console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Loading face landmark model...`);
            await faceapi.nets.faceLandmark68Net.load(baseModelPath);
            
            console.log(`[FaceOne ${new Date().toLocaleTimeString()}] Loading face recognition model...`);
            await faceapi.nets.faceRecognitionNet.load(baseModelPath);
            
            console.log(`[FaceOne ${new Date().toLocaleTimeString()}] All models loaded successfully!`);
            state.faceApiLoaded = true;
            state.faceApiLoading = false;
            return true;
        } catch (error) {
            console.error(`[FaceOne ${new Date().toLocaleTimeString()}] Error loading models from ${modelPath}:`, error);
            throw error;
        }
    } catch (error) {
        console.error(`[FaceOne ${new Date().toLocaleTimeString()}] Model recovery failed:`, error);
        throw error;
    }
}

// Add to the initialize function to call recovery if normal loading fails
async function initialize() {
// ... existing code ...
}

// Add top-level variables for tracking
let imagesProcessed = 0;

// Add function to increment processed images count
function incrementProcessedImagesCount() {
    imagesProcessed++;
}
