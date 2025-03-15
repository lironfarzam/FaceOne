/**
 * Utility functions for FaceOne extension
 */

/**
 * Logs a message with an appropriate emoji based on message type
 * Only logs if DEBUG is enabled or log type is critical, or if message is related to model/embedding loading in normal mode
 * 
 * @param {string} type - Type of message: 'info', 'success', 'warning', 'error', 'model', 'image', etc.
 * @param {string} functionName - Name of the function generating the log
 * @param {string} message - The message to log
 * @param {Object} [details] - Optional detailed information for debug mode only
 */
function logWithEmoji(type, functionName, message, details = null) {
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
        case 'info':
            emoji = '📋';
            break;
        case 'success':
            emoji = '✅';
            break;
        case 'warning':
            emoji = '⚠️';
            break;
        case 'error':
            emoji = '❌';
            break;
        case 'model':
            emoji = '🧠';
            break;
        case 'image':
            emoji = '🖼️';
            break;
        case 'loading':
            emoji = '🔄';
            break;
        case 'setup':
            emoji = '🔧';
            break;
        case 'timer':
            emoji = '⏱️';
            break;
        case 'search':
            emoji = '🔍';
            break;
        case 'lock':
            emoji = '🔒';
            break;
        case 'unlock':
            emoji = '🔓';
            break;
        case 'start':
            emoji = '🚀';
            break;
        case 'draw':
            emoji = '🎨';
            break;
        case 'performance':
            emoji = '📊';
            break;
        case 'stats':
            emoji = '📈';
            break;
        case 'network':
            emoji = '🌐';
            break;
        case 'processing':
            emoji = '⚙️';
            break;
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
}

/**
 * Logs the entry point to a function with the setup emoji
 * Only logs if debugging is enabled
 * 
 * @param {string} functionName - Name of the function being entered 
 */
function logFunctionEntry(functionName) {
    if (window.DEBUG) {
        logWithEmoji('setup', functionName, 'Function started');
    }
}

/**
 * Logs an error with detailed information
 * 
 * @param {string} functionName - Name of the function where the error occurred
 * @param {string} message - Error message
 * @param {Error|null} error - Optional Error object for stack traces
 */
function logError(functionName, message, error = null) {
    logWithEmoji('error', functionName, message);
    if (error && error.stack) {
        console.error(`${functionName} error stack:`, error.stack);
    } else if (error) {
        console.error(`${functionName} error details:`, error);
    }
}

/**
 * Creates a message handler with timeout and cleanup
 * 
 * @param {string} expectedType - Expected message type to listen for
 * @param {number} timeout - Timeout in milliseconds
 * @param {Function} onSuccess - Success callback function
 * @param {Function} onError - Error callback function
 * @returns {Promise} Promise resolving to the message data
 */
function createMessageHandler(expectedType, timeout, onSuccess, onError) {
    return new Promise((resolve, reject) => {
        let messageListener = null;
        let timeoutId = null;
        
        const cleanup = () => {
            if (timeoutId) clearTimeout(timeoutId);
            if (messageListener) window.removeEventListener('message', messageListener);
        };
        
        messageListener = (event) => {
            if (event.data && event.data.type === expectedType) {
                cleanup();
                if (onSuccess) {
                    try {
                        const result = onSuccess(event.data);
                        resolve(result);
                    } catch (error) {
                        logError('messageHandler', `Error handling successful ${expectedType} message:`, error);
                        reject(error);
                    }
                } else {
                    resolve(event.data);
                }
            }
        };
        
        window.addEventListener('message', messageListener);
        
        timeoutId = setTimeout(() => {
            cleanup();
            const error = new Error(`Timeout waiting for ${expectedType} message (${timeout}ms)`);
            if (onError) {
                try {
                    onError(error);
                } catch (callbackError) {
                    logError('messageHandler', `Error in timeout handler for ${expectedType}:`, callbackError);
                }
            }
            reject(error);
        }, timeout);
        
        return { cleanup };
    });
}

/**
 * Gets the number of initialization attempts from session storage
 * 
 * @returns {number} Number of attempts
 */
function getInitAttempts() {
    logFunctionEntry('getInitAttempts');
    logWithEmoji('info', 'getInitAttempts', 'Getting initialization attempts count');
    const attempts = sessionStorage.getItem('initAttempts') || 0;
    return parseInt(attempts, 10);
}

/**
 * Increments the initialization attempts counter in session storage
 * 
 * @returns {number} Updated attempts count
 */
function incrementInitAttempts() {
    logFunctionEntry('incrementInitAttempts');
    logWithEmoji('info', 'incrementInitAttempts', 'Incrementing initialization attempts count');
    const attempts = getInitAttempts() + 1;
    sessionStorage.setItem('initAttempts', attempts);
    return attempts;
}

/**
 * Resets the initialization attempts counter in session storage
 */
function resetInitAttempts() {
    logFunctionEntry('resetInitAttempts');
    logWithEmoji('info', 'resetInitAttempts', 'Resetting initialization attempts count');
    sessionStorage.removeItem('initAttempts');
}

/**
 * Debounces a function call
 * 
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    logFunctionEntry('debounce');
    logWithEmoji('info', 'debounce', 'Creating debounced function');
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

/**
 * Rounds a number to the nearest multiple of 32
 * 
 * @param {number} num - Number to round
 * @returns {number} Rounded number
 */
function roundToMultipleOf32(num) {
    return Math.ceil(num / 32) * 32;
}

/**
 * Calculates Intersection over Union for two bounding boxes
 * 
 * @param {Object} box1 - First bounding box with x, y, width, height
 * @param {Object} box2 - Second bounding box with x, y, width, height
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
 * Detects if image needs rotation based on aspect ratio
 * 
 * @param {Object} imageData - The image data with width and height
 * @returns {boolean} True if the image likely needs rotation
 */
function detectImageRotation(imageData) {
    // Simple heuristic: check if height is significantly larger than width
    // This assumes portrait photos are more likely to need rotation
    const aspectRatio = imageData.width / imageData.height;
    return aspectRatio < 0.7; // Arbitrary threshold for portrait orientation
}

/**
 * Checks if an element is a Facebook profile image
 * 
 * @param {HTMLElement} element - The element to check
 * @returns {boolean} True if the element is a Facebook profile image
 */
function isFacebookProfileImage(element) {
    // Check if element is within Facebook's profile picture container
    return element.closest('[data-visualcompletion="media-vc-image"]') !== null ||
           element.closest('[data-type="profile_picture"]') !== null ||
           element.closest('.profile-photo-container') !== null;
}

/**
 * Adjusts face detection coordinates after image rotation
 * 
 * @param {Array} detections - Array of face detections
 * @param {number} angle - Rotation angle in degrees
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @returns {Array} Adjusted detections
 */
function adjustDetectionCoordinates(detections, angle, canvas) {
    logFunctionEntry('adjustDetectionCoordinates');
    logWithEmoji('image', 'adjustDetectionCoordinates', `Adjusting coordinates for ${angle} degree rotation`);
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
                x: rotatedX - width/2 + centerX,
                y: rotatedY - height/2 + centerY,
                width,
                height
            }
        };
    });
}

/**
 * Attempts to recover a failed image by restoring original styles
 * 
 * @param {HTMLImageElement} img - The image element to recover
 */
function recoverFailedImage(img) {
    logFunctionEntry('recoverFailedImage');
    logWithEmoji('setup', 'recoverFailedImage', 'Attempting to recover failed image');
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

/**
 * Prevents text selection when clicking on images
 * 
 * @param {Event} e - The event object
 */
function preventTextSelection(e) {
    if (e.target.tagName === 'IMG') {
        e.preventDefault();
        window.getSelection().removeAllRanges();
    }
}

/**
 * Creates a canvas for drawing face detections
 * 
 * @param {HTMLImageElement} img - The image element to create canvas for
 * @returns {HTMLCanvasElement} The created canvas element
 */
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

/**
 * Scales face detection coordinates by the given factor
 * 
 * @param {Array} detections - Array of face detections
 * @param {number} scaleFactor - Factor to scale by
 * @returns {Array} Scaled detections
 */
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

/**
 * Rotates an image on a canvas by the specified angle
 * 
 * @param {HTMLCanvasElement} canvas - The canvas containing the image
 * @param {number} angle - Rotation angle in degrees
 * @returns {HTMLCanvasElement} New canvas with rotated image
 */
function rotateImage(canvas, angle) {
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
    
    // Move to center of new canvas
    ctx.translate(rotatedCanvas.width / 2, rotatedCanvas.height / 2);
    ctx.rotate(radians);
    ctx.drawImage(canvas, -width / 2, -height / 2, width, height);
    
    return rotatedCanvas;
}

/**
 * Applies or removes a blur effect on an element
 * 
 * @param {HTMLElement} element - The element to apply blur to
 * @param {boolean} shouldBlur - Whether to apply (true) or remove (false) blur
 */
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
            const svgId = wrapper.getAttribute('data-original-svg-id');
            const svg = document.getElementById(svgId);
            if (svg) {
                svg.style.filter = `blur(${blurAmount})`;
                svg.style.transform = transform;
                svg.style.willChange = shouldBlur ? 'filter' : 'auto';
            }
            break;
        case 'background':
            const bgElement = wrapper.querySelector('[data-original-background]');
            if (bgElement) {
                bgElement.style.filter = `blur(${blurAmount})`;
                bgElement.style.transform = transform;
                bgElement.style.willChange = shouldBlur ? 'filter' : 'auto';
            }
            break;
    }
}

/**
 * PHASE 1: PERFORMANCE OPTIMIZATIONS
 */

/**
 * Enhanced unified tensor memory management to prevent memory leaks and improve performance
 * This is a singleton that can be imported and reused throughout the application
 */
const TensorMemoryManager = (() => {
    // Configuration with adaptive memory limits
    const config = {
        // Default memory budget (will be adjusted based on device)
        MAX_BYTES_MB: 200,
        // Memory check interval in ms
        MEMORY_CHECK_INTERVAL: 30000,
        // Memory threshold percentage to trigger cleanup
        MEMORY_THRESHOLD: 0.8,
        // Debug mode for detailed logging
        DEBUG: false
    };
    
    // Set global DEBUG flag for extension-wide access
    window.DEBUG = config.DEBUG;
    
    // Adapt memory limits based on device capabilities
    try {
        if (navigator && navigator.deviceMemory) {
            // Adjust based on device memory (if available)
            const deviceMemoryGB = navigator.deviceMemory;
            // Scale budget proportionally, but keep within reasonable limits
            config.MAX_BYTES_MB = Math.min(Math.max(100, deviceMemoryGB * 100), 500);
            logWithEmoji('setup', 'TensorMemoryManager', 
                `Adaptive memory limit set to ${config.MAX_BYTES_MB}MB based on ${deviceMemoryGB}GB device memory`);
        }
    } catch (e) {
        // Fallback to default if detection fails
        logWithEmoji('warning', 'TensorMemoryManager', 'Failed to detect device memory, using default limits');
    }
    
    // Keep track of tensors with LRU behavior
    class LRUTensorTracker {
        constructor() {
            this.tensors = new Map(); // Maps tensor to its last access time
        }
        
        track(tensor) {
            if (tensor && !tensor.isDisposed) {
                this.tensors.set(tensor, Date.now());
            }
            return tensor;
        }
        
        untrack(tensor) {
            this.tensors.delete(tensor);
        }
        
        /**
         * Get tensors ordered by least recently used
         */
        getLRUTensors() {
            return Array.from(this.tensors.entries())
                .sort((a, b) => a[1] - b[1]) // Sort by timestamp (oldest first)
                .map(entry => entry[0]);     // Extract just the tensors
        }
        
        /**
         * Update timestamp for a tensor to mark it as recently used
         */
        touch(tensor) {
            if (this.tensors.has(tensor)) {
                this.tensors.set(tensor, Date.now());
            }
        }
        
        clear() {
            this.tensors.clear();
        }
        
        get size() {
            return this.tensors.size;
        }
    }
    
    // Create LRU tracker instance
    const trackedTensors = new LRUTensorTracker();
    
    // Setup periodic memory check
    let memoryCheckInterval = null;
    
    // Last known memory metrics
    let lastMemoryMetrics = {
        timestamp: 0,
        numTensors: 0,
        numBytes: 0,
        issuedWarning: false
    };
    
    // Flag to indicate we're in a cleanup cycle
    let isPerformingCleanup = false;
    
    /**
     * Conditionally log based on debug setting
     */
    function debugLog(type, message) {
        if (config.DEBUG) {
            logWithEmoji(type, 'TensorMemoryManager', message);
        }
    }
    
    return {
        /**
         * Initialize the memory manager
         * @param {Object} options - Optional configuration overrides
         * @returns {Object} - This instance for chaining
         */
        initialize(options = {}) {
            logFunctionEntry('TensorMemoryManager.initialize');
            
            // Apply custom configuration
            Object.assign(config, options);
            
            logWithEmoji('setup', 'TensorMemoryManager', 
                `Initializing tensor memory management with ${config.MAX_BYTES_MB}MB limit`);
            
            // Start periodic memory checks
            this.startPeriodicChecks();
            
            // Listen for tab visibility changes to force GC when tab is hidden
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    logWithEmoji('info', 'TensorMemoryManager', 'Tab hidden, performing garbage collection');
                    this.garbageCollect(true); // Force aggressive collection when tab hidden
                }
            });
            
            // Make available globally for the sandbox to access
            window.TensorMemoryManager = this;
            
            return this;
        },
        
        /**
         * Get the current configuration
         * @returns {Object} Current configuration
         */
        getConfig() {
            return {...config};
        },
        
        /**
         * Update configuration settings
         * @param {Object} options - New configuration options
         */
        configure(options = {}) {
            Object.assign(config, options);
            
            // Restart checks if interval changed
            if (options.MEMORY_CHECK_INTERVAL && memoryCheckInterval) {
                this.stopPeriodicChecks();
                this.startPeriodicChecks();
            }
            
            logWithEmoji('setup', 'TensorMemoryManager', 'Configuration updated');
        },
        
        /**
         * Track a tensor to ensure it gets disposed
         * @param {tf.Tensor} tensor - The tensor to track
         * @returns {tf.Tensor} The same tensor for chaining
         */
        track(tensor) {
            if (!tensor) return null;
            
            if (Array.isArray(tensor)) {
                // Handle arrays of tensors
                tensor.forEach(t => trackedTensors.track(t));
                return tensor;
            }
            
            return trackedTensors.track(tensor);
        },
        
        /**
         * Mark a tensor as recently used to prevent early disposal
         * @param {tf.Tensor} tensor - The tensor to mark
         */
        markAsUsed(tensor) {
            if (!tensor) return;
            
            if (Array.isArray(tensor)) {
                tensor.forEach(t => trackedTensors.touch(t));
                return;
            }
            
            trackedTensors.touch(tensor);
        },
        
        /**
         * Dispose a tensor and remove it from tracking
         * @param {tf.Tensor} tensor - The tensor to dispose
         */
        dispose(tensor) {
            if (!tensor) return;
            
            if (Array.isArray(tensor)) {
                tensor.forEach(t => this.dispose(t));
                return;
            }
            
            if (tensor && !tensor.isDisposed && tensor.dispose) {
                try {
                    tensor.dispose();
                    trackedTensors.untrack(tensor);
                } catch (e) {
                    logError('TensorMemoryManager', 'Error disposing tensor', e);
                }
            }
        },
        
        /**
         * Run an operation within a memory-managed context
         * Similar to tf.tidy but with our own tracking
         * @param {Function} fn - Function to execute
         * @returns {any} - Result of the function
         */
        tidy(fn) {
            if (!window.tf) {
                return fn();
            }
            
            return tf.tidy(() => {
                const result = fn();
                
                // If result is a tensor or array of tensors, track it
                if (result && (result instanceof tf.Tensor || 
                    (Array.isArray(result) && result[0] instanceof tf.Tensor))) {
                    this.track(result);
                }
                
                return result;
            });
        },
        
        /**
         * Start periodic memory checks
         */
        startPeriodicChecks() {
            if (memoryCheckInterval) {
                clearInterval(memoryCheckInterval);
            }
            
            memoryCheckInterval = setInterval(() => {
                this.checkMemory();
            }, config.MEMORY_CHECK_INTERVAL);
            
            debugLog('setup', 'Started periodic memory checks');
        },
        
        /**
         * Stop periodic memory checks
         */
        stopPeriodicChecks() {
            if (memoryCheckInterval) {
                clearInterval(memoryCheckInterval);
                memoryCheckInterval = null;
                debugLog('setup', 'Stopped periodic memory checks');
            }
        },
        
        /**
         * Check current memory usage and collect garbage if needed
         */
        checkMemory() {
            if (!window.tf || !tf.memory) return;
            
            // Skip if already in cleanup
            if (isPerformingCleanup) return;
            
            try {
                const memInfo = tf.memory();
                const memUsageMB = Math.round(memInfo.numBytes / (1024 * 1024));
                const memoryUsagePercent = memInfo.numBytes / (config.MAX_BYTES_MB * 1024 * 1024);
                
                // Update memory metrics
                lastMemoryMetrics = {
                    timestamp: Date.now(),
                    numTensors: memInfo.numTensors,
                    numBytes: memInfo.numBytes,
                    issuedWarning: lastMemoryMetrics.issuedWarning
                };
                
                // Log memory status periodically even without warnings
                debugLog('info', `Memory: ${memUsageMB}MB (${Math.round(memoryUsagePercent * 100)}%), Tensors: ${memInfo.numTensors}`);
                
                // Only log warnings if we exceed threshold
                if (memoryUsagePercent > config.MEMORY_THRESHOLD) {
                    // Don't spam warnings - only log if we haven't recently
                    if (!lastMemoryMetrics.issuedWarning) {
                        logWithEmoji('warning', 'TensorMemoryManager', 
                            `Memory usage at ${Math.round(memoryUsagePercent * 100)}% of ${config.MAX_BYTES_MB}MB limit, running garbage collection`);
                        lastMemoryMetrics.issuedWarning = true;
                    }
                    
                    // Progressive cleanup based on severity
                    if (memoryUsagePercent > 0.95) {
                        // Critical: Aggressive cleanup
                        this.garbageCollect(true);
                    } else {
                        // Standard cleanup
                        this.garbageCollect(false);
                    }
                } else {
                    // Reset warning flag when below threshold
                    lastMemoryMetrics.issuedWarning = false;
                }
            } catch (error) {
                logError('TensorMemoryManager', 'Error checking memory', error);
            }
        },
        
        /**
         * Run a garbage collection cycle to free memory
         * @param {boolean} aggressive - Whether to perform aggressive cleanup
         */
        garbageCollect(aggressive = false) {
            // Prevent reentrancy
            if (isPerformingCleanup) return;
            isPerformingCleanup = true;
            
            try {
                debugLog('loading', 'Running garbage collection' + (aggressive ? ' (aggressive)' : ''));
                
                if (!window.tf || !tf.memory) {
                    isPerformingCleanup = false;
                    return;
                }
                
                // Get tensors by LRU order (oldest first)
                const lruTensors = trackedTensors.getLRUTensors();
                let disposedCount = 0;
                
                if (aggressive) {
                    // In aggressive mode, dispose all except very recent tensors
                    lruTensors.forEach(tensor => {
                        try {
                            if (tensor && !tensor.isDisposed) {
                                tensor.dispose();
                                disposedCount++;
                            }
                        } catch (e) {
                            // Ignore errors during disposal
                        }
                    });
                    
                    // Clear tracking completely
                    trackedTensors.clear();
                } else {
                    // In normal mode, dispose oldest 50% of tensors
                    const disposeCount = Math.floor(lruTensors.length * 0.5);
                    
                    for (let i = 0; i < disposeCount; i++) {
                        const tensor = lruTensors[i];
                        if (tensor && !tensor.isDisposed) {
                            try {
                                tensor.dispose();
                                trackedTensors.untrack(tensor);
                                disposedCount++;
                            } catch (e) {
                                // Ignore errors during disposal
                            }
                        }
                    }
                }
                
                // Force TensorFlow garbage collection
                if (window.tf && tf.engine) {
                    this.forceTfGarbageCollection();
                }
                
                // Only log substantial cleanups to reduce console noise
                if (disposedCount > 0 || aggressive) {
                    logWithEmoji('success', 'TensorMemoryManager', 
                        `Garbage collection complete. Disposed ${disposedCount} tensors.`);
                }
                
                // Update memory metrics after cleanup
                if (window.tf && tf.memory) {
                    const memInfo = tf.memory();
                    const memUsageMB = Math.round(memInfo.numBytes / (1024 * 1024));
                    debugLog('info', `Post-GC Memory: ${memUsageMB}MB, Tensors: ${memInfo.numTensors}`);
                }
            } catch (error) {
                logError('TensorMemoryManager', 'Error during garbage collection', error);
            } finally {
                isPerformingCleanup = false;
            }
        },
        
        /**
         * Force TensorFlow's internal garbage collection
         */
        forceTfGarbageCollection() {
            if (!window.tf || !tf.engine) return;
            
            try {
                // Run empty tidy to trigger disposal
                tf.tidy(() => {});
                
                // Handle any dangling scopes
                if (tf.engine().state && 
                    tf.engine().state.numDataMovesStack && 
                    tf.engine().state.numDataMovesStack.length > 0) {
                    
                    // End any existing scopes and start a fresh one
                    tf.engine().endScope();
                    tf.engine().startScope();
                }
            } catch (e) {
                debugLog('warning', `Error during TensorFlow GC: ${e.message}`);
            }
        },
        
        /**
         * Get current memory statistics
         * @returns {Object} Memory statistics
         */
        getMemoryStats() {
            if (!window.tf || !tf.memory) {
                return {
                    available: false,
                    timestamp: Date.now()
                };
            }
            
            try {
                const memInfo = tf.memory();
                return {
                    available: true,
                    timestamp: Date.now(),
                    numTensors: memInfo.numTensors,
                    numBytes: memInfo.numBytes,
                    numBytesInGPU: memInfo.numBytesInGPU || 0,
                    unreliable: memInfo.unreliable,
                    maxMemoryLimit: config.MAX_BYTES_MB * 1024 * 1024,
                    trackedTensors: trackedTensors.size
                };
            } catch (e) {
                return {
                    available: false,
                    error: e.message,
                    timestamp: Date.now()
                };
            }
        },
        
        /**
         * Clean up resources when shutting down
         */
        cleanup() {
            this.stopPeriodicChecks();
            this.garbageCollect(true);
            
            // Remove event listener
            document.removeEventListener('visibilitychange', this.handleVisibilityChange);
        }
    };
})();

/**
 * Image queue for progressive image processing
 * Processes images in order of visibility and priority
 */
class ImageQueue {
    constructor(options = {}) {
        logFunctionEntry('ImageQueue.constructor');
        logWithEmoji('setup', 'ImageQueue', 'Creating new image queue');
        
        // Queue of pending images to process
        this.queue = [];
        
        // Set of images already in queue or processed
        this.processed = new Set();
        
        // Currently processing flag
        this.isProcessing = false;
        
        // Maximum batch size to process at once
        this.batchSize = options.batchSize || 5;
        
        // Processing interval
        this.processingInterval = options.processingInterval || 300;
        
        // Interval timer reference
        this.timer = null;
        
        // Start processing loop
        this.startProcessing();
    }
    
    /**
     * Add an image to the processing queue
     * @param {HTMLElement} element - The image element to process
     * @param {number} priority - Priority level (lower means higher priority)
     */
    add(element, priority = 100) {
        // Skip if already processed or in queue
        if (this.processed.has(element)) {
            return;
        }
        
        // Mark as processed to avoid duplicates
        this.processed.add(element);
        
        // Create queue item with priority
        const queueItem = {
            element,
            priority,
            timestamp: Date.now()
        };
        
        // Add to queue
        this.queue.push(queueItem);
        
        // Log the addition to queue if not too many items (to avoid console spam)
        if (this.queue.length < 20) {
            logWithEmoji('info', 'ImageQueue', `Added element to queue. Queue size: ${this.queue.length}`);
        } else if (this.queue.length % 50 === 0) {
            logWithEmoji('info', 'ImageQueue', `Queue size reached ${this.queue.length} items`);
        }
    }
    
    /**
     * Start the processing loop
     */
    startProcessing() {
        if (this.timer) {
            clearInterval(this.timer);
        }
        
        this.timer = setInterval(() => {
            this.processNext();
        }, this.processingInterval);
    }
    
    /**
     * Stop the processing loop
     */
    stopProcessing() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
    }
    
    /**
     * Process the next batch of images in the queue
     */
    async processNext() {
        // Skip if already processing or queue is empty
        if (this.isProcessing || this.queue.length === 0) {
            return;
        }
        
        // Skip processing if tab is not active
        if (typeof TabResourceManager !== 'undefined' && !TabResourceManager.isActive()) {
            return;
        }
        
        // Set processing flag
        this.isProcessing = true;
        
        try {
            // Sort queue by priority (lower number = higher priority)
            this.queue.sort((a, b) => {
                // First by priority
                if (a.priority !== b.priority) {
                    return a.priority - b.priority;
                }
                // Then by timestamp (older first)
                return a.timestamp - b.timestamp;
            });
            
            // Take a batch of items to process
            const batch = this.queue.splice(0, this.batchSize);
            
            if (batch.length > 0) {
                logWithEmoji('info', 'ImageQueue', `Processing batch of ${batch.length} images. Remaining: ${this.queue.length}`);
            }
            
            // Process each item in the batch
            for (const item of batch) {
                const { element } = item;
                
                // Skip if element is no longer in the DOM
                if (!element.isConnected) {
                    continue;
                }
                
                // Check if image is visible
                const isVisible = this.isElementVisible(element);
                
                // If not visible, add back to queue with lower priority
                if (!isVisible) {
                    item.priority += 50; // Reduce priority for invisible elements
                    this.queue.push(item);
                    continue;
                }
                
                // Process the element (this should call the actual image processing function)
                try {
                    if (typeof handleVisibleElement === 'function') {
                        await handleVisibleElement(element);
                    } else {
                        logWithEmoji('warning', 'ImageQueue', 'handleVisibleElement function not available');
                    }
                } catch (error) {
                    logError('ImageQueue', `Error processing element: ${error.message}`, error);
                }
            }
        } catch (error) {
            logError('ImageQueue', 'Error in processNext', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    /**
     * Clear the queue and processed set
     */
    clear() {
        this.queue = [];
        this.processed.clear();
        this.isProcessing = false;
    }
    
    /**
     * Pause the queue processing
     */
    pause() {
        this.stopProcessing();
    }
    
    /**
     * Resume the queue processing
     */
    resume() {
        this.startProcessing();
    }
    
    /**
     * Check if an element is currently visible in the viewport
     * @param {HTMLElement} element - The element to check
     * @returns {boolean} True if the element is visible
     */
    isElementVisible(element) {
        // Check if element exists and is connected to DOM
        if (!element || !element.isConnected) {
            return false;
        }
        
        // Get element boundaries
        const rect = element.getBoundingClientRect();
        
        // Check if element has size
        if (rect.width === 0 || rect.height === 0) {
            return false;
        }
        
        // Check if element is in viewport
        const viewportWidth = window.innerWidth || document.documentElement.clientWidth;
        const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
        
        // Element must be at least partially visible in the viewport
        const isInViewport = (
            rect.top < viewportHeight &&
            rect.bottom > 0 &&
            rect.left < viewportWidth &&
            rect.right > 0
        );
        
        // Check if element is not hidden with CSS
        const style = window.getComputedStyle(element);
        const isStyleVisible = (
            style.display !== 'none' &&
            style.visibility !== 'hidden' &&
            parseFloat(style.opacity) > 0
        );
        
        return isInViewport && isStyleVisible;
    }
    
    /**
     * Get the current queue size
     * @returns {number} Number of items in queue
     */
    size() {
        return this.queue.length;
    }
    
    /**
     * Configure the queue settings
     * @param {Object} options - Configuration options
     */
    configure(options = {}) {
        if (options.batchSize) {
            this.batchSize = options.batchSize;
        }
        if (options.processingInterval) {
            this.processingInterval = options.processingInterval;
            // Restart processing with new interval
            this.stopProcessing();
            this.startProcessing();
        }
    }
}

/**
 * Tab-based resource manager that reduces processing when tab is inactive
 * This is a singleton that can be imported and reused throughout the application
 */
const TabResourceManager = (() => {
    // Track state and timers
    let isTabActive = !document.hidden;
    let documentObserver = null;
    let suspensionTimer = null;
    let resumptionTimer = null;
    let statusCheckInterval = null;
    
    // Configuration settings
    const CONFIG = {
        // How long to wait after tab becomes inactive before suspending resources (ms)
        suspensionDelay: 5000,
        
        // How long to wait after tab becomes active before fully resuming (ms)
        resumptionDelay: 500,
        
        // How often to check processing status (ms)
        statusCheckInterval: 10000,
        
        // Function to call when checking extension status (will be defined by user)
        statusCheckCallback: null
    };
    
    return {
        /**
         * Initialize the tab resource manager
         * @param {Object} options - Configuration options
         * @param {Function} options.statusCheckCallback - Function to call to check extension status
         * @param {Object} options.documentObserver - MutationObserver instance to manage
         * @param {number} options.suspensionDelay - Custom suspension delay (ms)
         * @param {number} options.resumptionDelay - Custom resumption delay (ms)
         */
        initialize(options = {}) {
            logFunctionEntry('TabResourceManager.initialize');
            logWithEmoji('setup', 'TabResourceManager', 'Initializing tab resource management');
            
            // Apply custom options
            if (options.statusCheckCallback) {
                CONFIG.statusCheckCallback = options.statusCheckCallback;
            }
            if (options.documentObserver) {
                documentObserver = options.documentObserver;
            }
            if (typeof options.suspensionDelay === 'number') {
                CONFIG.suspensionDelay = options.suspensionDelay;
            }
            if (typeof options.resumptionDelay === 'number') {
                CONFIG.resumptionDelay = options.resumptionDelay;
            }
            
            // Set initial state based on tab visibility
            isTabActive = !document.hidden;
            logWithEmoji('info', 'TabResourceManager', `Initial tab state: ${isTabActive ? 'active' : 'inactive'}`);
            
            // Listen for visibility changes
            document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));
            
            // Start periodic status checks
            this.startStatusChecks();
            
            // Return this instance for chaining
            return this;
        },
        
        /**
         * Handle tab visibility changes
         */
        handleVisibilityChange() {
            const wasActive = isTabActive;
            isTabActive = !document.hidden;
            
            if (wasActive && !isTabActive) {
                // Tab became inactive
                logWithEmoji('info', 'TabResourceManager', 'Tab became inactive');
                
                // Clear any pending resumption
                if (resumptionTimer) {
                    clearTimeout(resumptionTimer);
                    resumptionTimer = null;
                }
                
                // Schedule suspension after delay
                suspensionTimer = setTimeout(() => {
                    this.suspendProcessing();
                }, CONFIG.suspensionDelay);
                
            } else if (!wasActive && isTabActive) {
                // Tab became active
                logWithEmoji('info', 'TabResourceManager', 'Tab became active');
                
                // Clear any pending suspension
                if (suspensionTimer) {
                    clearTimeout(suspensionTimer);
                    suspensionTimer = null;
                }
                
                // Schedule resumption after delay
                resumptionTimer = setTimeout(() => {
                    this.resumeProcessing();
                }, CONFIG.resumptionDelay);
            }
        },
        
        /**
         * Suspend resource-intensive operations
         */
        suspendProcessing() {
            logWithEmoji('lock', 'TabResourceManager', 'Suspending resource-intensive operations');
            
            // Pause mutation observers
            if (documentObserver) {
                try {
                    documentObserver.disconnect();
                    logWithEmoji('success', 'TabResourceManager', 'Suspended mutation observer');
                } catch (error) {
                    logError('TabResourceManager', 'Error suspending observer', error);
                }
            }
            
            // Release GPU resources when possible
            if (window.tf && tf.engine) {
                try {
                    // Keep model in memory but release temporary tensors
                    TensorMemoryManager.garbageCollect();
                    logWithEmoji('success', 'TabResourceManager', 'Released TensorFlow resources');
                } catch (error) {
                    logError('TabResourceManager', 'Error releasing TensorFlow resources', error);
                }
            }
            
            // Emit a suspension event
            window.dispatchEvent(new CustomEvent('faceone:suspended', {
                detail: { timestamp: Date.now() }
            }));
        },
        
        /**
         * Resume normal processing operations
         */
        resumeProcessing() {
            logWithEmoji('unlock', 'TabResourceManager', 'Resuming normal operations');
            
            // Restart observers
            if (documentObserver) {
                try {
                    documentObserver.observe(document.body, {
                        childList: true,
                        subtree: true,
                        attributes: true,
                        attributeFilter: ['src', 'xlink:href']
                    });
                    logWithEmoji('success', 'TabResourceManager', 'Resumed mutation observer');
                } catch (error) {
                    logError('TabResourceManager', 'Error resuming observer', error);
                }
            }
            
            // Check extension status
            if (CONFIG.statusCheckCallback && typeof CONFIG.statusCheckCallback === 'function') {
                try {
                    CONFIG.statusCheckCallback();
                } catch (error) {
                    logError('TabResourceManager', 'Error in status check callback', error);
                }
            }
            
            // Emit a resumption event
            window.dispatchEvent(new CustomEvent('faceone:resumed', {
                detail: { timestamp: Date.now() }
            }));
        },
        
        /**
         * Start periodic status checks
         */
        startStatusChecks() {
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
            }
            
            statusCheckInterval = setInterval(() => {
                // Only run status checks when the tab is active
                if (isTabActive && CONFIG.statusCheckCallback && typeof CONFIG.statusCheckCallback === 'function') {
                    try {
                        CONFIG.statusCheckCallback();
                    } catch (error) {
                        logError('TabResourceManager', 'Error in periodic status check', error);
                    }
                }
            }, CONFIG.statusCheckInterval);
        },
        
        /**
         * Stop periodic status checks
         */
        stopStatusChecks() {
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
                statusCheckInterval = null;
            }
        },
        
        /**
         * Update the document observer reference
         * @param {MutationObserver} observer - The new observer instance
         */
        setDocumentObserver(observer) {
            documentObserver = observer;
        },
        
        /**
         * Check if the tab is currently active
         * @returns {boolean} True if tab is active, false otherwise
         */
        isActive() {
            return isTabActive;
        },
        
        /**
         * Configure the tab resource manager
         * @param {Object} options - Configuration options
         */
        configure(options = {}) {
            Object.assign(CONFIG, options);
        },
        
        /**
         * Clean up resources when shutting down
         */
        cleanup() {
            this.stopStatusChecks();
            
            if (suspensionTimer) {
                clearTimeout(suspensionTimer);
                suspensionTimer = null;
            }
            
            if (resumptionTimer) {
                clearTimeout(resumptionTimer);
                resumptionTimer = null;
            }
            
            document.removeEventListener('visibilitychange', this.handleVisibilityChange);
        }
    };
})(); 