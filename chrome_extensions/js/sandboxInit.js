/**
 * @fileoverview TensorFlow initialization and model management
 */

//=============================================================================
// Configuration
//=============================================================================
const CONFIG = {
    TIMEOUT: 30000,
    MAX_RETRIES: 3,
    CACHE_SIZE: 50,
    CACHE_EXPIRY: 5 * 60 * 1000,
    WARMUP_SIZE: 4
};

//=============================================================================
// State Management
//=============================================================================
const state = {
    models: {
        faceNet: null,
        warmedUp: false
    },
    initialization: {
        attempts: 0,
        tfReady: false,
        backendReady: false
    },
    cache: new Map()
};

//=============================================================================
// TensorFlow Initialization
//=============================================================================
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
        
        // Initialize models
        await initializeModels();
        
    } catch (error) {
        console.error('TF initialization failed:', error);
        throw error;
    }
}

//=============================================================================
// Helper Functions
//=============================================================================
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

function setupMemoryManagement() {
    // Automatic memory cleanup
    setInterval(() => {
        tf.engine().endScope();
        tf.engine().startScope();
        if (global.gc) global.gc();
    }, 10000);
}

//=============================================================================
// Model Management
//=============================================================================
async function initializeModels() {
    try {
        const modelPath = chrome.runtime.getURL('models/facenet/model.json');
        state.models.faceNet = await tf.loadGraphModel(modelPath);
        await warmupModel(state.models.faceNet);
        state.models.warmedUp = true;
    } catch (error) {
        console.error('Model initialization failed:', error);
        throw error;
    }
}

async function warmupModel(model) {
    const dummyInput = tf.zeros([1, 224, 224, 3]);
    try {
        await model.predict(dummyInput);
    } finally {
        dummyInput.dispose();
    }
}

//=============================================================================
// Initialize on Load
//=============================================================================
window.addEventListener('load', () => {
    initializeTensorFlow().catch(error => {
        console.error('Initialization error:', error);
        // Notify parent of error
        window.parent.postMessage({
            type: 'INIT_ERROR',
            error: error.message
        }, '*');
    });
}); 