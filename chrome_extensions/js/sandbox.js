/**
 * sandbox.js - TensorFlow.js sandbox for FaceOne extension
 * Handles isolated model loading and inference operations
 */

// Fallback logging utilities in case utils.js is not yet loaded
if (typeof logWithEmoji !== 'function') {
    window.logWithEmoji = function(type, functionName, message) {
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
        }
        
        console.log(`${emoji} ${functionName}: ${message}`);
    };
}

if (typeof logFunctionEntry !== 'function') {
    window.logFunctionEntry = function(functionName) {
        if (typeof logWithEmoji === 'function') {
            logWithEmoji('setup', functionName, 'Function started');
        } else {
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

let faceNetModel = null;
let modelWarmedUp = false;
let isInitialized = false;
let isModelLoading = false;
const modelCache = new Map();
const MAX_CACHE_SIZE = 50;
const CACHE_EXPIRY_TIME = 5 * 60 * 1000; // 5 minutes in milliseconds
const INITIALIZATION_TIMEOUT = 20000; // Increased to 20 seconds
const MAX_RETRY_ATTEMPTS = 3;
const MODEL_LOAD_TIMEOUT = 30000;
let tfBackendInitialized = false;
let tensorsToDispose = new Set();
let initializationAttempts = 0;
const MAX_INIT_ATTEMPTS = 3;

// Add initialization status tracking
const initStatus = {
    tfReady: false,
    backendReady: false,
    error: null
};

// Add cache for processed images
const imageCache = {
    images: new Map(), // Map to store image data
    maxSize: 1000,
    
    add(src, isProcessed = true, canDelete = false) {
        if (this.images.size >= this.maxSize) {
            // Remove oldest entry by converting to array and removing first element
            const sources = Array.from(this.images.keys());
            this.images.delete(sources[0]);
        }
        this.images.set(src, {
            isProcessed,
            canDelete,
            timestamp: Date.now()
        });
    },
    
    has(src) {
        return this.images.has(src);
    },

    isProcessed(src) {
        const entry = this.images.get(src);
        return entry ? entry.isProcessed : false;
    },

    canDelete(src) {
        const entry = this.images.get(src);
        return entry ? entry.canDelete : false;
    },

    setProcessed(src, value = true) {
        const entry = this.images.get(src);
        if (entry) {
            entry.isProcessed = value;
            entry.timestamp = Date.now();
        }
    },

    setDeletable(src, value = true) {
        const entry = this.images.get(src);
        if (entry) {
            entry.canDelete = value;
            entry.timestamp = Date.now();
        }
    },

    clear() {
        this.images.clear();
    }
};

// Modify waitForBackend to be more robust
async function waitForBackend(timeout = INITIALIZATION_TIMEOUT) {
    const startTime = Date.now();
    
    while (!tf.backend() && (Date.now() - startTime) < timeout) {
        try {
            await tf.ready();
            if (tf.backend()) {
                initStatus.backendReady = true;
                return true;
            }
        } catch (e) {
            logWithEmoji('error', 'waitForBackend', 'Backend initialization attempt failed: ' + e.message);
        }
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    if (!tf.backend()) {
        throw new Error('Backend initialization timeout after ' + timeout + 'ms');
    }
    return true;
}

// Add function to safely check engine
function isTfEngineAvailable() {
    try {
        return tf && tf.engine && tf.engine() && tf.engine().backend != null;
    } catch (e) {
        return false;
    }
}

// Modify safeDisposeTensors to be more defensive
function safeDisposeTensors() {
    try {
        if (!isTfEngineAvailable()) return;
        
        const tensorsArray = Array.from(tensorsToDispose);
        tensorsToDispose.clear(); // Clear first to prevent circular issues
        
        tf.tidy(() => {
            tensorsArray.forEach(tensor => {
                try {
                    if (tensor && !tensor.isDisposed && tensor.dispose) {
                        tensor.dispose();
                    }
                } catch (e) {
                    // Silently ignore disposal errors
                }
            });
        });
    } catch (e) {
        // Silently fail if cleanup isn't possible
    }
}

// Modify isTfBackendReady to use the new check
function isTfBackendReady() {
    try {
        return isInitialized && isTfEngineAvailable() && tfBackendInitialized;
    } catch (e) {
        return false;
    }
}

// Modify initTensorFlow to handle initialization better
async function initTensorFlow(retryCount = 0) {
    if (isTfBackendReady()) {
        logWithEmoji('success', 'initTensorFlow', 'TensorFlow already initialized');
        return;
    }

    if (retryCount >= MAX_INIT_ATTEMPTS) {
        throw new Error('Max initialization attempts reached');
    }

    try {
        logWithEmoji('loading', 'initTensorFlow', 'Initializing TensorFlow...');
        initializationAttempts++;
        
        // Reset status
        isInitialized = false;
        tfBackendInitialized = false;
        initStatus.tfReady = false;
        initStatus.backendReady = false;
        initStatus.error = null;

        // Wait for TF to be ready
        await tf.ready();
        initStatus.tfReady = true;
        logWithEmoji('success', 'initTensorFlow', 'TensorFlow core ready');

        // Try to initialize backend
        await waitForBackend();
        logWithEmoji('success', 'initTensorFlow', 'Backend initialized');

        // Set up WebGL backend if available
        if (!tf.findBackend('webgl')) {
            logWithEmoji('warning', 'initTensorFlow', 'WebGL not available, using CPU backend');
            await tf.setBackend('cpu');
        } else {
            await tf.setBackend('webgl');
            const backend = tf.backend();
            
            if (backend && backend.setWebGLFlag) {
                backend.setWebGLFlag('WEBGL_FORCE_F16_TEXTURES', true);
                backend.setWebGLFlag('WEBGL_VERSION', 2);
                backend.setWebGLFlag('WEBGL_PACK', true);
            }
        }

        // Verify initialization
        if (!tf.backend()) {
            throw new Error('Backend not properly initialized');
        }

        isInitialized = true;
        tfBackendInitialized = true;

        // Send success message
        window.parent.postMessage({
            type: 'TF_INITIALIZED',
            success: true,
            info: {
                version: tf.version.tfjs,
                backend: tf.getBackend(),
                isWebGL: tf.getBackend() === 'webgl',
                memoryInfo: tf.memory()
            }
        }, '*');

        logWithEmoji('success', 'initTensorFlow', 'TensorFlow initialization complete');
        
    } catch (error) {
        logError('initTensorFlow', 'TensorFlow initialization error', error);
        initStatus.error = error;
        
        // Clean up on error
        try {
            if (tf.backend()) {
                await tf.disposeVariables();
                tf.engine().endScope();
            }
        } catch (e) {
            logWithEmoji('warning', 'initTensorFlow', 'Cleanup error: ' + e.message);
        }

        isInitialized = false;
        tfBackendInitialized = false;

        if (retryCount < MAX_INIT_ATTEMPTS - 1) {
            logWithEmoji('loading', 'initTensorFlow', `Retrying initialization (attempt ${retryCount + 2}/${MAX_INIT_ATTEMPTS})`);
            await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1)));
            return initTensorFlow(retryCount + 1);
        }

        window.parent.postMessage({
            type: 'TF_INITIALIZED',
            success: false,
            error: error.message,
            initStatus: initStatus
        }, '*');
        
        throw error;
    }
}

// Add function to check TF status
function getTfStatus() {
    return {
        isInitialized,
        tfBackendInitialized,
        backend: tf.getBackend(),
        initStatus: { ...initStatus },
        memory: tf.memory(),
        attempts: initializationAttempts
    };
}

async function loadModel(modelPath) {
    if (isModelLoading) {
        logWithEmoji('warning', 'loadModel', 'Model load already in progress');
        return;
    }

    try {
        isModelLoading = true;
        modelWarmedUp = false;

        if (!isInitialized || !tf.backend()) {
            await initTensorFlow();
        }

        // Dispose existing model if any
        if (faceNetModel) {
            try {
                await faceNetModel.dispose();
                faceNetModel = null;
            } catch (e) {
                logWithEmoji('warning', 'loadModel', 'Error disposing existing model: ' + e.message);
            }
        }

        // Load model with timeout
        const modelLoadPromise = tf.loadGraphModel(modelPath);
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Model load timeout')), MODEL_LOAD_TIMEOUT)
        );

        faceNetModel = await Promise.race([modelLoadPromise, timeoutPromise]);
        
        // Determine model type and create appropriate warmup inputs
        let dummyInputs;
        if (modelPath.includes('myModel')) {
            // For similarity model that takes two 512-dim vectors
            dummyInputs = tf.tidy(() => [
                tf.zeros([1, 512]),
                tf.zeros([1, 512])
            ]);
        } else {
            // For FaceNet model
            dummyInputs = tf.tidy(() => tf.zeros([1, 160, 160, 3]));
        }
        
        // Warm up with appropriate inputs
        const warmupResult = await faceNetModel.predict(dummyInputs);
        await (Array.isArray(warmupResult) ? Promise.all(warmupResult.map(t => t.data())) : warmupResult.data());
        
        // Dispose warmup tensors
        if (Array.isArray(dummyInputs)) {
            dummyInputs.forEach(tensor => tensor.dispose());
        } else {
            dummyInputs.dispose();
        }
        if (Array.isArray(warmupResult)) {
            warmupResult.forEach(tensor => tensor.dispose());
        } else {
            warmupResult.dispose();
        }
        
        modelWarmedUp = true;
        
        window.parent.postMessage({
            type: 'MODEL_LOADED',
            success: true,
            modelInfo: {
                inputShape: Array.isArray(faceNetModel.inputs) ? 
                    faceNetModel.inputs.map(input => input.shape) :
                    faceNetModel.inputs[0].shape,
                outputShape: Array.isArray(faceNetModel.outputs) ?
                    faceNetModel.outputs.map(output => output.shape) :
                    faceNetModel.outputs[0].shape,
                warmedUp: true
            }
        }, '*');
    } catch (error) {
        logError('loadModel', 'Model loading error', error);
        modelWarmedUp = false;
        faceNetModel = null;
        window.parent.postMessage({
            type: 'MODEL_LOADED',
            success: false,
            error: error.message
        }, '*');
        throw error;
    } finally {
        isModelLoading = false;
    }
}

// Add cache entry with timestamp
function addToCache(key, embedding) {
    if (modelCache.size >= MAX_CACHE_SIZE) {
        // Remove oldest entry
        const firstKey = modelCache.keys().next().value;
        modelCache.delete(firstKey);
    }
    modelCache.set(key, {
        embedding: Array.from(embedding),
        timestamp: Date.now()
    });
}

// Get cache entry if not expired
function getFromCache(key) {
    const entry = modelCache.get(key);
    if (!entry) return null;
    
    // Check if entry has expired
    if (Date.now() - entry.timestamp > CACHE_EXPIRY_TIME) {
        modelCache.delete(key);
        return null;
    }
    
    return entry.embedding;
}

// Modify memory cleanup
const memoryCleanupInterval = setInterval(() => {
    try {
        if (!isTfEngineAvailable()) return;

        tf.tidy(() => {
            try {
                const memoryInfo = tf.memory();
                if (memoryInfo.numTensors > 100 || memoryInfo.numBytes > 50000000) {
                    logWithEmoji('warning', 'memoryCleanup', `Memory usage: ${memoryInfo.numTensors} tensors, ${Math.round(memoryInfo.numBytes / (1024 * 1024))}MB GPU: ${Math.round((memoryInfo.numBytesInGPU || 0) / (1024 * 1024))}MB`);
                    
                    if (memoryInfo.numTensors > 1000) {
                        safeDisposeTensors();
                        if (isTfEngineAvailable()) {
                            try {
                                tf.engine().endScope();
                                tf.engine().startScope();
                            } catch (e) {
                                // Silently ignore scope errors
                            }
                        }
                    }
                }
            } catch (e) {
                // Silently ignore memory info errors
            }
        });
    } catch (error) {
        logWithEmoji('warning', 'memoryCleanup', 'Error: ' + error.message);
    }
}, 30000);

// Generate embedding from image data
async function generateEmbedding(imageData) {
    if (!isTfBackendReady()) {
        throw new Error('TensorFlow not initialized or ready');
    }

    if (!faceNetModel) {
        throw new Error('Model not loaded');
    }

    if (!modelWarmedUp) {
        throw new Error('Model not warmed up');
    }

    if (isModelLoading) {
        throw new Error('Model loading in progress');
    }

    return tf.tidy(() => {
        try {
            const img = tf.tensor(imageData, [160, 160, 4]);
            tensorsToDispose.add(img);
            
            const rgb = img.slice([0, 0, 0], [-1, -1, 3]);
            tensorsToDispose.add(rgb);
            
            const processed = rgb.expandDims(0).toFloat().div(127.5).sub(1);
            tensorsToDispose.add(processed);
            
            const embedding = faceNetModel.predict(processed);
            tensorsToDispose.add(embedding);
            
            const embeddingData = embedding.squeeze();
            tensorsToDispose.add(embeddingData);
            
            const normalizedEmbedding = tf.div(embeddingData, tf.norm(embeddingData));
            tensorsToDispose.add(normalizedEmbedding);
            
            const finalEmbedding = normalizedEmbedding.dataSync();
            
            safeDisposeTensors();
            
            // Mark the image as processed in the cache if a source URL is provided
            if (imageData.sourceUrl) {
                imageCache.add(imageData.sourceUrl, true, false);
            }
            
            return {
                type: 'EMBEDDING_GENERATED',
                success: true,
                embedding: Array.from(finalEmbedding),
                fromCache: false
            };
        } catch (error) {
            // Mark the image as failed in the cache if a source URL is provided
            if (imageData.sourceUrl) {
                imageCache.add(imageData.sourceUrl, false, false);
            }
            
            safeDisposeTensors();
            return {
                type: 'EMBEDDING_GENERATED',
                success: false,
                error: error.message
            };
        }
    });
}

// Handle model loading message
async function handleModelLoad(data) {
    try {
        const { modelName, modelPath, waitForWarmup } = data;
        logWithEmoji('loading', 'loadModel', `Loading ${modelName} model from ${modelPath}...`);
        
        if (!modelManager.state.isInitialized) {
            await modelManager.initialize();
        }
        
        const result = await modelManager.loadModel(modelName, modelPath);
        
        if (waitForWarmup && !modelManager.state.warmedUp[modelName]) {
            await modelManager.warmupModel(modelName);
        }
        
        window.parent.postMessage({
            type: 'MODEL_LOADED',
            modelName: modelName,
            success: true,
            modelInfo: {
                ...result.modelInfo,
                warmedUp: modelManager.state.warmedUp[modelName]
            }
        }, '*');
        
    } catch (error) {
        logError(`handleModelLoad`, `Error loading ${data.modelName} model`, error);
        window.parent.postMessage({
            type: 'MODEL_LOADED',
            modelName: data.modelName,
            success: false,
            error: error.message
        }, '*');
    }
}

// ModelManager class definition
class ModelManager {
    constructor() {
        this.models = {
            faceNet: null,
            myModel: null  // Add the new model
        };
        
        this.state = {
            isInitialized: false,
            tfBackendInitialized: false,
            modelLoadAttempts: 0,
            isLoading: false,
            warmedUp: {
                faceNet: false,
                myModel: false  // Add warmup state for new model
            }
        };
        
        this.constants = {
            MAX_RETRY_ATTEMPTS: 3,
            MODEL_LOAD_TIMEOUT: 30000,
            INITIALIZATION_TIMEOUT: 10000
        };
    }

    async initialize() {
        if (this.state.isInitialized && this.state.tfBackendInitialized) return;

        try {
            this.state.isInitialized = false;
            this.state.tfBackendInitialized = false;
            
            await tf.ready();
            await this.waitForBackend();
            
            if (!tf.findBackend('webgl')) {
                logWithEmoji('warning', 'ModelManager.initialize', 'WebGL not available, using CPU backend');
                await tf.setBackend('cpu');
            } else {
                await tf.setBackend('webgl');
                const backend = tf.backend();
                
                if (backend && backend.setWebGLFlag) {
                    backend.setWebGLFlag('WEBGL_FORCE_F16_TEXTURES', true);
                    backend.setWebGLFlag('WEBGL_VERSION', 2);
                    backend.setWebGLFlag('WEBGL_PACK', true);
                }
            }

            this.state.isInitialized = true;
            this.state.tfBackendInitialized = true;
            
            return {
                success: true,
                info: {
                    version: tf.version.tfjs,
                    backend: tf.getBackend(),
                    isWebGL: tf.getBackend() === 'webgl'
                }
            };
        } catch (error) {
            this.state.isInitialized = false;
            this.state.tfBackendInitialized = false;
            throw error;
        }
    }

    async waitForBackend() {
        const startTime = Date.now();
        while (!tf.backend() && (Date.now() - startTime) < this.constants.INITIALIZATION_TIMEOUT) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        if (!tf.backend()) {
            throw new Error('Backend initialization timeout');
        }
    }

    async loadModel(modelName, modelPath) {
        const LOAD_TIMEOUT = 30000;
        
        if (this.state.isLoading) {
            const startTime = Date.now();
            while (this.state.isLoading && (Date.now() - startTime) < LOAD_TIMEOUT) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            if (this.state.isLoading) {
                throw new Error('Model load timeout');
            }
        }

        try {
            this.state.isLoading = true;
            this.state.warmedUp[modelName] = false;

            if (!this.state.isInitialized) {
                await this.initialize();
            }

            // Cleanup existing model
            await this.disposeModel(modelName);

            // Load model with timeout
            const modelLoadPromise = tf.loadGraphModel(modelPath);
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Model load timeout')), this.constants.MODEL_LOAD_TIMEOUT)
            );

            this.models[modelName] = await Promise.race([modelLoadPromise, timeoutPromise]);
            
            // Warm up the model
            await this.warmupModel(modelName);
            
            return {
                success: true,
                modelInfo: {
                    name: modelName,
                    inputShape: this.models[modelName].inputs[0].shape,
                    outputShape: this.models[modelName].outputs[0].shape,
                    warmedUp: true
                }
            };
        } catch (error) {
            this.state.warmedUp[modelName] = false;
            this.models[modelName] = null;
            throw error;
        } finally {
            this.state.isLoading = false;
        }
    }

    async warmupModel(modelName) {
        if (!this.models[modelName]) throw new Error(`${modelName} model not loaded`);

        // Get input shapes from model
        const inputShapes = this.models[modelName].inputs.map(input => input.shape);
        
        // Create dummy inputs based on model type
        const dummyInputs = tf.tidy(() => {
            switch(modelName) {
                case 'faceNet':
                    return tf.zeros([1, 160, 160, 3]);
                case 'myModel':
                    // For similarity model that takes two 512-dim vectors
                    return [
                        tf.zeros([1, 512]),
                        tf.zeros([1, 512])
                    ];
                default:
                    return inputShapes.map(shape => tf.zeros([1, ...shape.slice(1)]));
            }
        });
        
        try {
            const warmupResult = await this.models[modelName].predict(dummyInputs);
            await (Array.isArray(warmupResult) ? Promise.all(warmupResult.map(t => t.data())) : warmupResult.data());
            
            // Dispose results
            if (Array.isArray(warmupResult)) {
                warmupResult.forEach(tensor => tensor.dispose());
            } else {
                warmupResult.dispose();
            }
            
            // Dispose inputs
            if (Array.isArray(dummyInputs)) {
                dummyInputs.forEach(tensor => tensor.dispose());
            } else {
                dummyInputs.dispose();
            }
            
            this.state.warmedUp[modelName] = true;
        } catch (error) {
            logError(`warmupModel-${modelName}`, 'Error warming up model', error);
            throw error;
        }
    }

    async disposeModel(modelName) {
        if (this.models[modelName]) {
            try {
                await this.models[modelName].dispose();
                this.models[modelName] = null;
                this.state.warmedUp[modelName] = false;
            } catch (e) {
                logWithEmoji('warning', `disposeModel-${modelName}`, 'Error disposing model: ' + e.message);
            }
        }
    }

    async generateEmbedding(imageData) {
        if (!this.isReady('faceNet')) {
            throw new Error('FaceNet model not ready');
        }

        return tf.tidy(() => {
            try {
                const img = tf.tensor(imageData, [160, 160, 4]);
                tensorsToDispose.add(img);
                
                const rgb = img.slice([0, 0, 0], [-1, -1, 3]);
                tensorsToDispose.add(rgb);
                
                const processed = rgb.expandDims(0).toFloat().div(127.5).sub(1);
                tensorsToDispose.add(processed);
                
                const embedding = this.models.faceNet.predict(processed);
                tensorsToDispose.add(embedding);
                
                const embeddingData = embedding.squeeze();
                tensorsToDispose.add(embeddingData);
                
                const normalizedEmbedding = tf.div(embeddingData, tf.norm(embeddingData));
                tensorsToDispose.add(normalizedEmbedding);
                
                const finalEmbedding = normalizedEmbedding.dataSync();
                
                safeDisposeTensors();
                
                // Mark the image as processed in the cache if a source URL is provided
                if (imageData.sourceUrl) {
                    imageCache.add(imageData.sourceUrl, true, false);
                }
                
                return {
                    type: 'EMBEDDING_GENERATED',
                    success: true,
                    embedding: Array.from(finalEmbedding),
                    fromCache: false
                };
            } catch (error) {
                // Mark the image as failed in the cache if a source URL is provided
                if (imageData.sourceUrl) {
                    imageCache.add(imageData.sourceUrl, false, false);
                }
                
                safeDisposeTensors();
                return {
                    type: 'EMBEDDING_GENERATED',
                    success: false,
                    error: error.message
                };
            }
        });
    }

    isReady(modelName) {
        return this.state.isInitialized && 
               this.state.tfBackendInitialized && 
               this.models[modelName] && 
               this.state.warmedUp[modelName];
    }

    async cleanup() {
        try {
            // First, clear all caches and reset flags
            imageCache.clear();
            modelCache.clear();
            tensorsToDispose.clear();

            // Reset all state flags
            this.state.isInitialized = false;
            this.state.tfBackendInitialized = false;
            this.state.isLoading = false;
            Object.keys(this.state.warmedUp).forEach(key => {
                this.state.warmedUp[key] = false;
            });

            // Safely dispose models first
            if (this.models.faceNet) {
                try {
                    await this.models.faceNet.dispose();
                } catch (e) {
                    logWithEmoji('warning', 'ModelManager.cleanup', 'Error disposing FaceNet model: ' + e.message);
                }
                this.models.faceNet = null;
            }

            // Safely dispose any remaining tensors
            try {
                const tensors = tf.memory().numTensors;
                if (tensors > 0) {
                    tf.disposeVariables();
                    tf.engine().endScope();
                    tf.engine().startScope();
                }
            } catch (e) {
                logWithEmoji('warning', 'ModelManager.cleanup', 'Error during tensor cleanup: ' + e.message);
            }

            // Reset global flags
            isInitialized = false;
            tfBackendInitialized = false;
            modelWarmedUp = false;
            isModelLoading = false;
            faceNetModel = null;

        } catch (error) {
            logWithEmoji('warning', 'ModelManager.cleanup', 'Model cleanup error: ' + error.message);
            // Even if there's an error, try to reset the state
            this.state.isInitialized = false;
            this.state.tfBackendInitialized = false;
        }
    }

    // Add method to compare two embeddings using myModel
    async compareFaceEmbeddings(embedding1, embedding2) {
        if (!this.isReady('myModel')) {
            throw new Error('Similarity model not ready');
        }

        return tf.tidy(() => {
            try {
                // Convert embeddings to tensors
                const tensor1 = tf.tensor2d([embedding1], [1, 512]);
                const tensor2 = tf.tensor2d([embedding2], [1, 512]);
                
                // Run inference
                const similarity = this.models.myModel.predict([tensor1, tensor2]);
                const result = similarity.dataSync()[0];
                
                return {
                    type: 'SIMILARITY_COMPUTED',
                    success: true,
                    similarity: result
                };
            } catch (error) {
                return {
                    type: 'SIMILARITY_COMPUTED',
                    success: false,
                    error: error.message
                };
            }
        });
    }
}

// Create model manager instance
const modelManager = new ModelManager();

// Main message handler
function setupMessageHandlers() {
    window.addEventListener('message', async (event) => {
        try {
            const data = event.data;
            if (!data || !data.type) return;
            
            let response = { type: data.type, success: false };
            
            switch (data.type) {
                case 'INIT':
                    await modelManager.initialize();
                    response.success = true;
                    response.info = {
                        version: tf.version.tfjs,
                        backend: tf.getBackend(),
                        isWebGL: tf.getBackend() === 'webgl'
                    };
                    break;
                    
                case 'LOAD_MODEL':
                    await handleModelLoad(data);
                    return; // handleModelLoad sends its own response
                    
                case 'GENERATE_EMBEDDING':
                    if (!modelManager.isReady('faceNet')) {
                        throw new Error('FaceNet model not ready');
                    }
                    const result = await modelManager.generateEmbedding(data.imageData);
                    window.parent.postMessage(result, '*');
                    return;
                    
                case 'COMPUTE_SIMILARITY':
                    const similarityResult = await modelManager.compareFaceEmbeddings(
                        data.embedding1,
                        data.embedding2
                    );
                    window.parent.postMessage(similarityResult, '*');
                    return;
                    
                case 'CLEAR_CACHE':
                    imageCache.clear();
                    if (modelManager.state.isInitialized) {
                        await modelManager.cleanup();
                    }
                    response.success = true;
                    break;

                case 'CLEANUP':
                    await modelManager.cleanup();
                    response.success = true;
                    break;

                case 'GET_TF_STATUS':
                    window.parent.postMessage({
                        type: 'TF_STATUS',
                        status: getTfStatus()
                    }, '*');
                    return;
                    
                case 'CHECK_MODEL_STATUS':
                    const modelName = data.modelName;
                    window.parent.postMessage({
                        type: 'MODEL_STATUS',
                        isReady: modelManager.isReady(modelName),
                        reason: !modelManager.models[modelName] ? 'Model not loaded' : 
                               !modelManager.state.warmedUp[modelName] ? 'Model not warmed up' : null
                    }, '*');
                    return;
                    
                default:
                    logWithEmoji('warning', 'messageHandler', 'Unknown message type: ' + data.type);
                    response.error = 'Unknown message type';
            }
            
            window.parent.postMessage(response, '*');
        } catch (error) {
            logError('messageHandler', 'Error handling message', error);
            window.parent.postMessage({
                type: event.data?.type || 'ERROR',
                success: false,
                error: error.message
            }, '*');
        }
    });
}

// Cleanup handlers
function setupCleanupHandlers() {
    // Modify visibility change handler to be more defensive
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') {
            try {
                if (isTfEngineAvailable()) {
                    safeDisposeTensors();
                }
                imageCache.clear();
            } catch (error) {
                logWithEmoji('warning', 'visibilitychange', 'Cleanup error: ' + error.message);
            }
        }
    });

    // Modify cleanup before unload
    window.addEventListener('beforeunload', () => {
        try {
            // Clear interval first
            if (memoryCleanupInterval) {
                clearInterval(memoryCleanupInterval);
            }

            // Safe cleanup of tensors
            safeDisposeTensors();

            // Only try engine cleanup if it's available
            if (isTfEngineAvailable()) {
                try {
                    tf.engine().endScope();
                } catch (e) {
                    // Silently ignore engine cleanup errors
                }
            }

            // Clear caches and reset flags
            imageCache.clear();
            tensorsToDispose.clear();
            isInitialized = false;
            tfBackendInitialized = false;
            modelWarmedUp = false;
            
            // Clear model reference
            if (faceNetModel) {
                try {
                    faceNetModel.dispose();
                } catch (e) {
                    // Silently ignore model disposal errors
                }
                faceNetModel = null;
            }
        } catch (error) {
            // Log warning but don't throw
            logWithEmoji('warning', 'beforeunload', 'Cleanup error: ' + error.message);
        }
    }, { once: true });
}

// Initialize on load
function initOnLoad() {
    window.addEventListener('load', () => {
        logWithEmoji('loading', 'sandboxInit', 'Sandbox frame loaded, initializing TensorFlow...');
        
        // Set up message handlers
        setupMessageHandlers();
        
        // Set up cleanup handlers
        setupCleanupHandlers();
        
        // Initialize the model manager
        modelManager.initialize().catch(error => {
            logError('initialization', 'ModelManager initialization failed', error);
        });
        
        // Initialize TensorFlow
        initTensorFlow().catch(error => {
            logError('initialization', 'TensorFlow initialization failed', error);
            window.parent.postMessage({
                type: 'TF_INITIALIZED',
                success: false,
                error: error.message,
                status: getTfStatus()
            }, '*');
        });
    });
}

// Start initialization
initOnLoad(); 