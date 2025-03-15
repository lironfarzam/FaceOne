/**
 * @fileoverview TensorFlow initialization and model management for FaceOne extension
 * 
 * @author Liron Farzam
 * @version 1.0.0
 * 
 * Handles the initialization of TensorFlow.js within a sandboxed environment.
 * Manages model loading, warmup, and memory cleanup to ensure optimal performance
 * while preventing memory leaks. This module is designed to run in an isolated frame
 * to prevent TensorFlow.js from interfering with the main page's JavaScript execution.
 */

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * Global configuration settings for TensorFlow and model management
 * @type {Object}
 * @constant
 */
const CONFIG = {
    /** @type {number} - Maximum time to wait for operations in milliseconds */
    TIMEOUT: 30000,
    /** @type {number} - Maximum number of retries for failed operations */
    MAX_RETRIES: 3,
    /** @type {number} - Maximum number of embeddings to cache */
    CACHE_SIZE: 50,
    /** @type {number} - Time in milliseconds before cache entries expire */
    CACHE_EXPIRY: 5 * 60 * 1000,
    /** @type {number} - Size of tensor used for model warmup */
    WARMUP_SIZE: 4
};

//==============================================================================
// STATE MANAGEMENT
//==============================================================================

/**
 * Global state object for tracking initialization status and loaded models
 * @type {Object}
 */
const state = {
    /** @type {Object} - References to loaded models and their status */
    models: {
        /** @type {tf.GraphModel|null} - The FaceNet model for face embeddings */
        faceNet: null,
        /** @type {boolean} - Whether models have been warmed up */
        warmedUp: false
    },
    /** @type {Object} - Initialization state tracking */
    initialization: {
        /** @type {number} - Number of initialization attempts */
        attempts: 0,
        /** @type {boolean} - Whether TensorFlow.js is ready */
        tfReady: false,
        /** @type {boolean} - Whether the backend is ready */
        backendReady: false
    },
    /** @type {Map<string, Object>} - Cache for generated embeddings */
    cache: new Map()
};

//==============================================================================
// TENSORFLOW INITIALIZATION
//==============================================================================

/**
 * Initializes TensorFlow.js with the WebGL backend
 * 
 * Sets up TensorFlow.js environment, configures the WebGL backend, 
 * and loads required models. Handles memory management setup.
 * 
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If TensorFlow initialization fails
 */
async function initializeTensorFlow() {
    if (state.initialization.tfReady) return;

    try {
        // Wait for TF to be available
        await waitForTensorFlow();
        
        // Configure backend
        await tf.setBackend('webgl');
        await tf.ready();
        
        // Set up memory management
        tf.engine().startScope();
        setupMemoryManagement();

        state.initialization.tfReady = true;
        state.initialization.backendReady = true;
        
        logWithEmoji('success', 'initializeTensorFlow', 'TensorFlow initialized successfully', 
            { backend: tf.getBackend() });
        
        // Initialize models
        await initializeModels();
        
    } catch (error) {
        logWithEmoji('error', 'initializeTensorFlow', 'TF initialization failed', error);
        throw error;
    }
}

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

/**
 * Waits for TensorFlow.js to be available in the global scope
 * 
 * Polls for the global tf object until it's available or times out.
 * 
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If TensorFlow isn't loaded within the timeout period
 */
async function waitForTensorFlow() {
    const startTime = Date.now();
    
    while (Date.now() - startTime < CONFIG.TIMEOUT) {
        if (typeof tf !== 'undefined') {
            return;
        }
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    throw new Error('TensorFlow not loaded');
}

/**
 * Sets up automatic memory management for TensorFlow.js
 * 
 * Creates an interval that periodically ends and starts scopes
 * to release unused tensors and triggers garbage collection.
 * 
 * @returns {void}
 */
function setupMemoryManagement() {
    // Automatic memory cleanup
    setInterval(() => {
        tf.engine().endScope();
        tf.engine().startScope();
        if (global.gc) global.gc();
    }, 10000);
}

//==============================================================================
// MODEL MANAGEMENT
//==============================================================================

/**
 * Initializes machine learning models required by the extension
 * 
 * Loads the FaceNet model, performs warmup operations to ensure
 * fast first inference, and logs performance metrics.
 * 
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If model initialization fails
 */
async function initializeModels() {
    try {
        const modelPath = chrome.runtime.getURL('models/facenet/model.json');
        logWithEmoji('loading', 'initializeModels', 'Loading FaceNet model', { path: modelPath });
        
        const startTime = performance.now();
        state.models.faceNet = await tf.loadGraphModel(modelPath);
        const loadTime = Math.round(performance.now() - startTime);
        
        logWithEmoji('success', 'initializeModels', 'FaceNet model loaded successfully', 
            { loadTimeMs: loadTime });
        
        logWithEmoji('loading', 'initializeModels', 'Warming up model');
        await warmupModel(state.models.faceNet);
        state.models.warmedUp = true;
        
        logWithEmoji('success', 'initializeModels', 'Model warmup complete');
    } catch (error) {
        logWithEmoji('error', 'initializeModels', 'Model initialization failed', error);
        throw error;
    }
}

/**
 * Performs model warmup to optimize first inference performance
 * 
 * Runs a dummy prediction through the model to initialize internal
 * WebGL buffers and compile shaders, reducing the latency of the
 * first real prediction.
 * 
 * @async
 * @param {tf.GraphModel} model - The model to warm up
 * @returns {Promise<void>}
 */
async function warmupModel(model) {
    const dummyInput = tf.zeros([1, 224, 224, 3]);
    try {
        await model.predict(dummyInput);
    } finally {
        dummyInput.dispose();
    }
}

//==============================================================================
// INITIALIZATION TRIGGER
//==============================================================================

/**
 * Initializes TensorFlow sandbox when the page loads
 * 
 * Event handler that triggers TensorFlow initialization
 * and notifies the parent frame of any initialization errors.
 * 
 * @returns {void}
 */
window.addEventListener('load', () => {
    logWithEmoji('start', 'sandboxInit', 'Initializing TensorFlow sandbox');
    initializeTensorFlow().catch(error => {
        logWithEmoji('error', 'sandboxInit', 'Initialization error', error);
        // Notify parent of error
        window.parent.postMessage({
            type: 'INIT_ERROR',
            error: error.message
        }, '*');
    });
}); 