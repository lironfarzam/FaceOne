/**
 * @fileoverview Optimized Web Worker for image processing operations.
 * Handles image preprocessing tasks in a separate thread.
 * 
 * @author Liron Farzam
 * @version 1.0.0
 * 
 * This worker provides an isolated execution environment for computationally 
 * intensive image processing tasks. It implements resource management, 
 * performance optimization, memory monitoring, and error handling to ensure
 * efficient and reliable image processing operations without impacting the
 * main thread's performance.
 */

//==============================================================================
// LOGGING SYSTEM
//==============================================================================

/**
 * Simplified logWithEmoji function for worker context
 * Provides consistent logging across the extension with emoji indicators.
 * 
 * @param {string} type - Type of message: 'info', 'success', 'warning', 'error', etc.
 * @param {string} functionName - Name of the function generating the log
 * @param {string} message - The message to log
 * @param {Object} [details] - Optional detailed information
 * @returns {void}
 */
function logWithEmoji(type, functionName, message, details = null) {
    // Workers don't share window object, so we need our own DEBUG flag
    const DEBUG = self.DEBUG || false;
    
    // Critical messages are always shown
    const isCritical = ['error', 'warning'].includes(type);
    
    // Only log if debugging is enabled or it's a critical message
    if (!DEBUG && !isCritical) {
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
        case 'performance': emoji = '📊'; break;
        case 'memory': emoji = '🧠'; break;
        case 'worker': emoji = '👷'; break;
    }
    
    // Basic logging
    if (details) {
        console.log(`Worker - ${emoji} ${functionName}: ${message}`, details);
    } else {
        console.log(`Worker - ${emoji} ${functionName}: ${message}`);
    }
}

//==============================================================================
// GLOBAL STATE AND CONFIGURATION
//==============================================================================

/**
 * Worker state object that tracks runtime statistics and operational status
 * @type {Object}
 */
const state = {
    /** @type {boolean} - Whether the worker has been properly initialized */
    initialized: false,
    /** @type {number} - Count of images processed since worker start */
    processingCount: 0,
    /** @type {number} - Duration of the most recent processing operation in ms */
    lastProcessingTime: 0,
    /** @type {number} - Timestamp when worker was started */
    startTimestamp: Date.now(),
    /** @type {number} - Current JavaScript heap size in bytes */
    memoryUsage: 0,
    /** @type {number} - Peak JavaScript heap size observed in bytes */
    peakMemoryUsage: 0
};

/**
 * Configuration settings for the worker's behavior
 * @type {Object}
 */
const config = {
    /** @type {number} - Maximum image dimension to process in pixels */
    maxImageSize: 1024,
    /** @type {number} - Maximum time allowed for processing a single image in ms */
    processingTimeout: 30000,
    /** @type {number} - Maximum retry attempts for failed operations */
    maxRetries: 2,
    /** @type {number} - Interval between memory usage checks in ms */
    memoryCheckInterval: 30000 // 30 seconds
};

//==============================================================================
// RESOURCE MANAGEMENT AND PERFORMANCE OPTIMIZATION
//==============================================================================

/**
 * Shared canvas and context for reuse across operations to reduce memory allocation
 * @type {OffscreenCanvas|null}
 */
let sharedCanvas = null;

/**
 * Shared 2D rendering context for the shared canvas
 * @type {OffscreenCanvasRenderingContext2D|null}
 */
let sharedCtx = null;

/** 
 * Resource management - interval for memory checking
 * @type {number|null}
 */
let memoryCheckInterval = null;

/**
 * Pool of reusable canvases to reduce memory allocation and garbage collection
 * @type {Array<OffscreenCanvas>}
 */
let canvasPool = [];

/**
 * Maximum number of canvases to keep in the pool to limit memory usage
 * @type {number}
 * @constant
 */
const MAX_CANVAS_POOL_SIZE = 3;

/**
 * Initializes shared resources used by the worker
 * Creates a shared canvas and context, and starts memory monitoring
 * 
 * @returns {void}
 */
function initializeSharedResources() {
    if (!sharedCanvas) {
        sharedCanvas = new OffscreenCanvas(1, 1);
        sharedCtx = sharedCanvas.getContext('2d', {
            alpha: false,  // Optimize by disabling alpha channel if not needed
            willReadFrequently: true  // Optimize for pixel manipulation operations
        });
    }
    
    // Start memory monitoring
    startMemoryMonitoring();
}

/**
 * Sets up periodic memory usage monitoring
 * Monitors JavaScript heap size to prevent memory leaks and excessive usage
 * 
 * @returns {void}
 */
function startMemoryMonitoring() {
    if (memoryCheckInterval) {
        clearInterval(memoryCheckInterval);
    }
    
    // Check memory usage periodically
    memoryCheckInterval = setInterval(() => {
        checkMemoryUsage();
    }, config.memoryCheckInterval);
}

/**
 * Checks current memory usage and performs cleanup if necessary
 * Uses performance.memory API if available to monitor heap size
 * 
 * @returns {void}
 */
function checkMemoryUsage() {
    try {
        // Get current memory stats using performance API
        if (self.performance && performance.memory) {
            state.memoryUsage = performance.memory.usedJSHeapSize;
            
            if (state.memoryUsage > state.peakMemoryUsage) {
                state.peakMemoryUsage = state.memoryUsage;
            }
            
            // If memory usage is too high, clean up resources
            if (state.memoryUsage > 100 * 1024 * 1024) { // 100MB threshold
                logWithEmoji('memory', 'checkMemoryUsage', 'Memory usage high, cleaning resources', 
                    { memoryUsage: Math.round(state.memoryUsage / (1024 * 1024)) + 'MB' });
                cleanupResources();
            }
        }
    } catch (error) {
        // Ignore errors accessing memory API
    }
}

/**
 * Cleans up resources when memory usage is high
 * Releases canvas pool and attempts to trigger garbage collection
 * 
 * @returns {void}
 */
function cleanupResources() {
    // Release canvas pool
    while (canvasPool.length > 0) {
        canvasPool.pop();
    }
    
    logWithEmoji('memory', 'cleanupResources', 'Canvas pool cleared');
    
    // Run garbage collection if available
    if (typeof gc === 'function') {
        try {
            gc();
            logWithEmoji('memory', 'cleanupResources', 'Garbage collection triggered');
        } catch (e) {
            // Ignore errors
        }
    }
}

/**
 * Gets or creates a canvas from the pool for efficient reuse
 * 
 * @param {number} width - Required canvas width in pixels
 * @param {number} height - Required canvas height in pixels
 * @returns {OffscreenCanvas} A canvas of the requested size
 */
function getCanvasFromPool(width, height) {
    // Try to find a canvas in the pool that is at least the requested size
    for (let i = 0; i < canvasPool.length; i++) {
        const canvas = canvasPool[i];
        
        if (canvas.width >= width && canvas.height >= height) {
            // Remove from pool and return
            canvasPool.splice(i, 1);
            return canvas;
        }
    }
    
    // Create a new canvas if none found
    return new OffscreenCanvas(width, height);
}

/**
 * Returns a canvas to the pool for reuse when no longer needed
 * 
 * @param {OffscreenCanvas} canvas - The canvas to return to the pool
 * @returns {void}
 */
function returnCanvasToPool(canvas) {
    // Don't add if pool is full
    if (canvasPool.length >= MAX_CANVAS_POOL_SIZE) return;
    
    // Clear the canvas
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Add to pool
    canvasPool.push(canvas);
}

//==============================================================================
// MESSAGE HANDLING
//==============================================================================

/**
 * Main message handler for the worker
 * Dispatches incoming messages to appropriate handlers based on message type
 * 
 * @param {MessageEvent} e - The message event containing the task data
 * @param {Object} e.data - The message data payload
 * @param {string} e.data.type - The type of operation to perform
 * @param {ImageData} [e.data.data] - The image data to process (for image operations)
 * @param {number} [e.data.width] - The width of the image
 * @param {number} [e.data.height] - The height of the image
 * @param {Object} [e.data.config] - Optional configuration settings
 * @returns {void}
 */
self.onmessage = async function(e) {
    const { type, data, width, height } = e.data;
    
    try {
        switch (type) {
            case 'INIT':
                // Store worker ID and config if provided
                if (e.data.workerId !== undefined) {
                    state.workerId = e.data.workerId;
                }
                
                if (e.data.config) {
                    Object.assign(config, e.data.config);
                }
                
                // Set debug flag if provided
                if (e.data.config && e.data.config.debug !== undefined) {
                    self.DEBUG = e.data.config.debug;
                }
                
                await handleInit();
                break;

            case 'PROCESS_IMAGE':
                await handleImageProcessing(data, width, height);
                break;
                
            case 'GET_MEMORY_STATS':
                // Update memory stats
                checkMemoryUsage();
                
                // Send back the memory stats
                self.postMessage({
                    type: 'MEMORY_STATS',
                    success: true,
                    memoryUsage: state.memoryUsage,
                    peakMemoryUsage: state.peakMemoryUsage,
                    processingCount: state.processingCount,
                    uptime: Date.now() - state.startTimestamp
                });
                break;

            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        logWithEmoji('error', 'messageHandler', `Error handling message type: ${type}`, error);
        self.postMessage({
            type: `${type}_FAILED`,
            success: false,
            error: error.message
        });
    }
};

//==============================================================================
// OPERATION HANDLERS
//==============================================================================

/**
 * Initializes the worker with optimized settings
 * Sets up shared resources and reports ready status
 * 
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If initialization fails
 */
async function handleInit() {
    try {
        logWithEmoji('setup', 'handleInit', 'Initializing worker', { workerId: state.workerId });
        initializeSharedResources();
        state.initialized = true;
        logWithEmoji('success', 'handleInit', 'Worker initialized successfully');
        
        self.postMessage({
            type: 'WORKER_READY',
            success: true,
            workerId: state.workerId,
            config: { ...config }
        });
    } catch (error) {
        state.initialized = false;
        logWithEmoji('error', 'handleInit', 'Worker initialization failed', error);
        throw error;
    }
}

/**
 * Handles image processing requests with performance optimizations
 * Processes the image and returns the result to the main thread
 * 
 * @async
 * @param {ImageData} imageData - The image data to process
 * @param {number} width - The width of the image
 * @param {number} height - The height of the image
 * @returns {Promise<void>}
 * @throws {Error} If processing fails or times out
 */
async function handleImageProcessing(imageData, width, height) {
    if (!state.initialized) {
        throw new Error('Worker not initialized');
    }

    const startTime = performance.now();
    state.processingCount++;

    logWithEmoji('image', 'handleImageProcessing', 'Processing image', 
        { width, height, count: state.processingCount });

    try {
        // Resize canvas if needed
        if (sharedCanvas.width < width || sharedCanvas.height < height) {
            sharedCanvas.width = width;
            sharedCanvas.height = height;
        }

        // Process image
        const processedData = await processImageOptimized(imageData, width, height);
        
        // Track performance
        state.lastProcessingTime = performance.now() - startTime;
        
        logWithEmoji('success', 'handleImageProcessing', 'Image processed successfully', 
            { processingTime: Math.round(state.lastProcessingTime) + 'ms' });

        self.postMessage({
            type: 'IMAGE_PROCESSED',
            success: true,
            data: processedData,
            stats: {
                processingTime: state.lastProcessingTime,
                totalProcessed: state.processingCount
            }
        });
    } catch (error) {
        logWithEmoji('error', 'handleImageProcessing', 'Image processing failed', error);
        throw error;
    }
}

//==============================================================================
// IMAGE PROCESSING CORE FUNCTIONS
//==============================================================================

/**
 * Optimized image processing with caching, timeouts, and performance improvements
 * 
 * @async
 * @param {ImageData} imageData - The image data to process
 * @param {number} width - The width of the image
 * @param {number} height - The height of the image
 * @returns {Promise<ImageData>} The processed image data
 * @throws {Error} If processing times out or fails
 */
async function processImageOptimized(imageData, width, height) {
    // Put image data on canvas
    sharedCtx.putImageData(imageData, 0, 0);

    // Apply optimized processing with timeout protection
    const processed = await Promise.race([
        applyImageProcessing(imageData),
        new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Processing timeout')), 
            config.processingTimeout)
        )
    ]);

    // Check memory usage after processing
    checkMemoryUsage();

    return processed;
}

/**
 * Applies actual image processing with performance optimizations
 * This is the core function where specific image processing algorithms are implemented
 * 
 * @async
 * @param {ImageData} imageData - The image data to process
 * @returns {Promise<ImageData>} The processed image data
 */
async function applyImageProcessing(imageData) {
    // Get a canvas to work with
    const canvas = getCanvasFromPool(imageData.width, imageData.height);
    const ctx = canvas.getContext('2d');
    
    try {
        // Example processing - replace with actual implementation
        // Here we just create a copy of the image data
        ctx.putImageData(imageData, 0, 0);
        
        // Create a new ImageData from the canvas
        const processed = ctx.getImageData(0, 0, imageData.width, imageData.height);
        
        // Add your image processing logic here
        // This is where you'd implement face detection, etc.
        
        return processed;
    } finally {
        // Return the canvas to the pool for reuse
        returnCanvasToPool(canvas);
    }
}

//==============================================================================
// WORKER LIFECYCLE MANAGEMENT
//==============================================================================

/**
 * Clean up resources when worker is terminating
 * Ensures proper cleanup of intervals and object references
 */
self.addEventListener('close', () => {
    logWithEmoji('setup', 'workerCleanup', 'Worker is terminating, cleaning up resources');
    
    if (memoryCheckInterval) {
        clearInterval(memoryCheckInterval);
    }
    
    // Clean up canvas pool
    canvasPool.length = 0;
    
    // Clean up shared resources
    sharedCanvas = null;
    sharedCtx = null;
}); 