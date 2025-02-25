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
    lastProcessingTime: 0
};

const config = {
    maxImageSize: 1024,
    processingTimeout: 30000,
    maxRetries: 2
};

//=============================================================================
// Performance Optimization
//=============================================================================
/**
 * Reusable canvas and context for image processing
 */
let sharedCanvas = null;
let sharedCtx = null;

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
}

//=============================================================================
// Message Handler
//=============================================================================
/**
 * Main message handler for the worker
 * @param {MessageEvent} e - The message event containing the task data
 * @param {string} e.data.type - The type of operation to perform
 * @param {ImageData} e.data.imageData - The image data to process
 * @param {number} e.data.width - The width of the image
 * @param {number} e.data.height - The height of the image
 */
self.onmessage = async function(e) {
    const { type, imageData, width, height } = e.data;
    
    try {
        switch (type) {
            case 'INIT':
                await handleInit();
                break;

            case 'PROCESS_IMAGE':
                await handleImageProcessing(imageData, width, height);
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
            success: true
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

    return processed;
}

/**
 * Apply actual image processing with optimizations
 */
async function applyImageProcessing(imageData) {
    // Example processing - replace with actual implementation
    const processed = new ImageData(
        new Uint8ClampedArray(imageData.data),
        imageData.width,
        imageData.height
    );

    // Add your image processing logic here
    // This is where you'd implement face detection, etc.

    return processed;
} 