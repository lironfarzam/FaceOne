/**
 * @fileoverview Optimized Web Worker for image processing operations.
 * Handles image preprocessing tasks in a separate thread.
 * @author Liron Farzam
 * @version 1.0.0
 */

//=============================================================================
// Global Variables and Constants
//=============================================================================
/**
 * Worker state and configuration
 */
const state = {
    initialized: false,
    processingCount: 0,
    lastProcessingTime: 0,
    startTimestamp: Date.now(),
    memoryUsage: 0,
    peakMemoryUsage: 0
};

const config = {
    maxImageSize: 1024,
    processingTimeout: 30000,
    maxRetries: 2,
    memoryCheckInterval: 30000 // 30 seconds
};

//=============================================================================
// Performance Optimization
//=============================================================================
/**
 * Reusable canvas and context for image processing
 */
let sharedCanvas = null;
let sharedCtx = null;

// Resource management
let memoryCheckInterval = null;
let canvasPool = [];
const MAX_CANVAS_POOL_SIZE = 3;

/**
 * Initialize shared resources
 */
function initializeSharedResources() {
    if (!sharedCanvas) {
        sharedCanvas = new OffscreenCanvas(1, 1);
        sharedCtx = sharedCanvas.getContext('2d', {
            alpha: false,
            willReadFrequently: true
        });
    }
    
    // Start memory monitoring
    startMemoryMonitoring();
}

/**
 * Start monitoring memory usage
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
 * Check current memory usage
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
                cleanupResources();
            }
        }
    } catch (error) {
        // Ignore errors accessing memory API
    }
}

/**
 * Clean up resources when memory is high
 */
function cleanupResources() {
    // Release canvas pool
    while (canvasPool.length > 0) {
        canvasPool.pop();
    }
    
    // Run garbage collection if available
    if (typeof gc === 'function') {
        try {
            gc();
        } catch (e) {
            // Ignore errors
        }
    }
}

/**
 * Get or create a canvas from the pool
 * @param {number} width - Canvas width
 * @param {number} height - Canvas height
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
 * Return a canvas to the pool
 * @param {OffscreenCanvas} canvas - The canvas to return
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

//=============================================================================
// Message Handler
//=============================================================================
/**
 * Main message handler for the worker
 * @param {MessageEvent} e - The message event containing the task data
 * @param {string} e.data.type - The type of operation to perform
 * @param {ImageData} e.data.data - The image data to process
 * @param {number} e.data.width - The width of the image
 * @param {number} e.data.height - The height of the image
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
        self.postMessage({
            type: `${type}_FAILED`,
            success: false,
            error: error.message
        });
    }
};

//=============================================================================
// Handler Functions
//=============================================================================
/**
 * Initialize worker with optimized settings
 */
async function handleInit() {
    try {
        initializeSharedResources();
        state.initialized = true;
        self.postMessage({
            type: 'WORKER_READY',
            success: true,
            workerId: state.workerId,
            config: { ...config }
        });
    } catch (error) {
        state.initialized = false;
        throw error;
    }
}

/**
 * Process image with performance optimizations
 */
async function handleImageProcessing(imageData, width, height) {
    if (!state.initialized) {
        throw new Error('Worker not initialized');
    }

    const startTime = performance.now();
    state.processingCount++;

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
        throw error;
    }
}

//=============================================================================
// Image Processing Functions
//=============================================================================
/**
 * Optimized image processing with caching and performance improvements
 */
async function processImageOptimized(imageData, width, height) {
    // Put image data on canvas
    sharedCtx.putImageData(imageData, 0, 0);

    // Apply optimized processing
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
 * Apply actual image processing with optimizations
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

//=============================================================================
// Cleanup Function
//=============================================================================
// Clean up resources when worker is terminating
self.addEventListener('close', () => {
    if (memoryCheckInterval) {
        clearInterval(memoryCheckInterval);
    }
    
    // Clean up canvas pool
    canvasPool.length = 0;
    
    // Clean up shared resources
    sharedCanvas = null;
    sharedCtx = null;
}); 