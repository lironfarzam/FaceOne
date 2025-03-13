/**
 * Utility functions for FaceOne extension
 */

/**
 * Logs a message with an appropriate emoji based on message type
 * 
 * @param {string} type - Type of message: 'info', 'success', 'warning', 'error', 'model', 'image', etc.
 * @param {string} functionName - Name of the function generating the log
 * @param {string} message - The message to log
 */
function logWithEmoji(type, functionName, message) {
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
    }
    
    // Log with the selected emoji and function name
    console.log(`${emoji} ${functionName}: ${message}`);
}

/**
 * Logs the entry point to a function with the setup emoji
 * 
 * @param {string} functionName - Name of the function being entered 
 */
function logFunctionEntry(functionName) {
    logWithEmoji('setup', functionName, 'Function started');
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