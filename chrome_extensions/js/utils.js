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