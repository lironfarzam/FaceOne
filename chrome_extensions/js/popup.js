/**
 * @fileoverview popup.js - Extension popup UI controller
 * 
 * @author Liron Farzam
 * @version 1.0.0
 * 
 * Controls the extension's popup interface, manages user settings, and handles
 * communication with the content script. Provides UI for switching between face
 * detection and blur modes, adjusting settings, and performing actions like
 * reprocessing images or clearing the blur list.
 */

//==============================================================================
// INITIALIZATION AND SETUP
//==============================================================================

/**
 * Initializes the popup UI when the DOM content is loaded.
 * Sets up all event handlers and loads saved settings.
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Popup script loaded');

    //--------------------------------------------------------------------------
    // UI ELEMENT REFERENCES
    //--------------------------------------------------------------------------
    
    // Get all UI elements
    const faceDetectionButton = document.getElementById('faceDetectionMode');
    const blurButton = document.getElementById('blurMode');
    const autoProcessCheckbox = document.getElementById('autoProcessImages');
    const addLabelCheckbox = document.getElementById('addLabel');
    const frameFaceCheckbox = document.getElementById('frameFaceDetected');
    const confidenceSlider = document.getElementById('confidenceThreshold');
    const confidenceValue = document.getElementById('confidenceValue');
    const debugModeCheckbox = document.getElementById('debugMode');
    const reprocessButton = document.getElementById('reprocessButton');
    const clearBlurListButton = document.getElementById('clearBlurListButton');
    const blurredCount = document.getElementById('blurredCount');
    const statusIndicator = document.getElementById('statusIndicator');

    // Verify elements are found
    if (!faceDetectionButton || !blurButton) {
        console.error('Mode buttons not found!');
        return;
    }

    // Get all mode-specific settings
    const modeSpecificSettings = document.querySelectorAll('[data-mode]');

    //--------------------------------------------------------------------------
    // LOAD SAVED SETTINGS
    //--------------------------------------------------------------------------
    
    /**
     * Loads previously saved settings from Chrome storage and initializes the UI.
     * Sets up the initial state of the popup interface based on stored preferences.
     */
    chrome.storage.sync.get({
        // Default values if settings aren't found
        processingMode: 'face_detection',
        autoProcessImages: true,
        addLabel: true,
        frameFaceDetected: true,
        confidenceThreshold: 70,
        debugMode: false
    }, function(items) {
        console.log('Loaded settings:', items);
        
        // Set initial UI state
        updateUIForMode(items.processingMode);
        
        autoProcessCheckbox.checked = items.autoProcessImages;
        addLabelCheckbox.checked = items.addLabel;
        frameFaceCheckbox.checked = items.frameFaceDetected;
        debugModeCheckbox.checked = items.debugMode;
        confidenceSlider.value = items.confidenceThreshold;
        confidenceValue.textContent = `${items.confidenceThreshold}%`;
        
        // Check for blurred images count
        updateBlurredImagesCount();
    });

    //--------------------------------------------------------------------------
    // MODE SWITCHING EVENT HANDLERS
    //--------------------------------------------------------------------------
    
    /**
     * Event handler for face detection mode button.
     * Switches the extension to face detection mode and updates the UI.
     */
    faceDetectionButton.addEventListener('click', function() {
        console.log('Face Detection button clicked');
        if (this.classList.contains('active')) {
            console.log('Already in face detection mode');
            return;
        }
        updateUIForMode('face_detection');
        saveSettings('processingMode', 'face_detection');
        showStatus('Switched to Face Detection mode');
    });

    /**
     * Event handler for blur mode button.
     * Switches the extension to direct blur mode and updates the UI.
     */
    blurButton.addEventListener('click', function() {
        console.log('Blur button clicked');
        if (this.classList.contains('active')) {
            console.log('Already in blur mode');
            return;
        }
        updateUIForMode('blur');
        saveSettings('processingMode', 'blur');
        showStatus('Switched to Blur mode');
    });

    //--------------------------------------------------------------------------
    // SETTINGS EVENT HANDLERS
    //--------------------------------------------------------------------------
    
    /**
     * Event handler for auto-process checkbox.
     * Enables or disables automatic processing of images.
     */
    autoProcessCheckbox.addEventListener('change', function() {
        console.log('Auto process changed:', this.checked);
        saveSettings('autoProcessImages', this.checked);
        showStatus(this.checked ? 'Auto-processing enabled' : 'Auto-processing disabled');
    });

    /**
     * Event handler for label checkbox.
     * Enables or disables showing labels on processed images.
     */
    addLabelCheckbox.addEventListener('change', function() {
        console.log('Show labels changed:', this.checked);
        saveSettings('addLabel', this.checked);
        showStatus(this.checked ? 'Labels enabled' : 'Labels disabled');
    });

    /**
     * Event handler for frame face detection checkbox.
     * Enables or disables drawing frames around detected faces.
     */
    frameFaceCheckbox.addEventListener('change', function() {
        console.log('Show frames changed:', this.checked);
        saveSettings('frameFaceDetected', this.checked);
        showStatus(this.checked ? 'Face frames enabled' : 'Face frames disabled');
    });

    /**
     * Event handler for debug mode checkbox.
     * Enables or disables verbose logging for debugging.
     * Also sends an immediate update to the content script.
     */
    debugModeCheckbox.addEventListener('change', function() {
        console.log('Debug mode changed:', this.checked);
        saveSettings('debugMode', this.checked);
        
        if (this.checked) {
            showStatus('Debug mode enabled - Verbose logging activated');
            addDiagnosticMessage('info', 'Debug mode ON - Check browser console for detailed logs');
        } else {
            showStatus('Debug mode disabled - Normal logging activated');
            addDiagnosticMessage('info', 'Debug mode OFF - Reduced console logging');
        }
        
        // Notify content script to update debug mode immediately
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs && tabs.length > 0) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    action: 'updateSettings',
                    settings: { debugMode: debugModeCheckbox.checked }
                }, function(response) {
                    if (response && response.success) {
                        console.log('Debug mode updated in content script:', response.debug);
                    }
                });
            }
        });
    });

    /**
     * Event handler for confidence slider input (real-time updates).
     * Updates the displayed confidence value as the slider moves.
     */
    confidenceSlider.addEventListener('input', function() {
        confidenceValue.textContent = `${this.value}%`;
    });

    /**
     * Event handler for confidence slider change (after release).
     * Saves the new confidence threshold setting.
     */
    confidenceSlider.addEventListener('change', function() {
        console.log('Confidence threshold changed:', this.value);
        saveSettings('confidenceThreshold', parseInt(this.value));
        showStatus(`Confidence threshold set to ${this.value}%`);
    });

    //--------------------------------------------------------------------------
    // ACTION BUTTON HANDLERS
    //--------------------------------------------------------------------------
    
    /**
     * Event handler for reprocess button.
     * Sends a message to the content script to reprocess all images
     * on the current page using current settings.
     */
    reprocessButton.addEventListener('click', function() {
        if (this.disabled) return;
        
        this.disabled = true;
        showStatus('Reprocessing images...');

        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    type: 'REPROCESS_IMAGES',
                    settings: {
                        processingMode: document.querySelector('.mode-button.active').id === 'faceDetectionMode' ? 'face_detection' : 'blur',
                        autoProcessImages: autoProcessCheckbox.checked,
                        addLabel: addLabelCheckbox.checked,
                        frameFaceDetected: frameFaceCheckbox.checked,
                        confidenceThreshold: parseInt(confidenceSlider.value)
                    }
                }, function(response) {
                    reprocessButton.disabled = false;
                    if (response && response.success) {
                        showStatus('Images reprocessed successfully');
                    } else {
                        showStatus('Error reprocessing images');
                    }
                });

                // Fallback in case content script doesn't respond
                setTimeout(() => {
                    if (reprocessButton.disabled) {
                        reprocessButton.disabled = false;
                        showStatus('Reprocess timeout - please try again');
                    }
                }, 5000);
            }
        });
    });

    /**
     * Event handler for clear blur list button.
     * Sends a message to the content script to clear all
     * blurred images from storage.
     */
    clearBlurListButton.addEventListener('click', function() {
        if (this.disabled) return;
        
        this.disabled = true;
        showStatus('Clearing blurred images list...');

        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    type: 'CLEAR_BLUR_LIST'
                }, function(response) {
                    clearBlurListButton.disabled = false;
                    if (response && response.success) {
                        showStatus(`Cleared ${response.count} blurred images`);
                        updateBlurredImagesCount();
                    } else {
                        showStatus('Error clearing blur list');
                    }
                });

                // Fallback in case content script doesn't respond
                setTimeout(() => {
                    if (clearBlurListButton.disabled) {
                        clearBlurListButton.disabled = false;
                        showStatus('Clear timeout - please try again');
                    }
                }, 5000);
            }
        });
    });
    
    //--------------------------------------------------------------------------
    // UTILITY FUNCTIONS
    //--------------------------------------------------------------------------
    
    /**
     * Updates the count of blurred images displayed in the popup.
     * Queries the content script for current blur statistics.
     */
    function updateBlurredImagesCount() {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    type: 'GET_BLUR_COUNT'
                }, function(response) {
                    if (response && response.success) {
                        const blurredCount = document.getElementById('blurredCount');
                        if (blurredCount) {
                            if (response.historicalCount !== undefined) {
                                blurredCount.textContent = `Active: ${response.count} | Auto-renewed: ${response.autoRenewedCount || 0} | Historical: ${response.historicalCount}`;
                                blurredCount.title = 'Active: Currently blurred images\nAuto-renewed: Images that were automatically reblurred\nHistorical: Previously blurred images';
                            } else {
                                blurredCount.textContent = `Blurred images: ${response.count}`;
                            }
                        }
                    } else {
                        const blurredCount = document.getElementById('blurredCount');
                        if (blurredCount) {
                            blurredCount.textContent = 'Blurred images: unavailable';
                        }
                    }
                });
            }
        });
    }

    /**
     * Updates the UI based on the selected processing mode.
     * Shows or hides mode-specific settings and updates button states.
     * 
     * @param {string} mode - The processing mode to switch to ('face_detection' or 'blur')
     */
    function updateUIForMode(mode) {
        console.log('Updating UI for mode:', mode);
        
        // Update mode buttons
        faceDetectionButton.classList.toggle('active', mode === 'face_detection');
        blurButton.classList.toggle('active', mode === 'blur');

        // Update visibility of mode-specific settings
        modeSpecificSettings.forEach(setting => {
            const settingMode = setting.getAttribute('data-mode');
            if (settingMode === 'both') {
                setting.classList.remove('disabled');
            } else {
                setting.classList.toggle('disabled', settingMode !== mode);
            }
        });

        // Send immediate update to content script
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    type: 'MODE_CHANGED',
                    mode: mode,
                    settings: {
                        processingMode: mode,
                        autoProcessImages: autoProcessCheckbox.checked,
                        addLabel: addLabelCheckbox.checked,
                        frameFaceDetected: frameFaceCheckbox.checked,
                        confidenceThreshold: parseInt(confidenceSlider.value)
                    }
                });
            }
        });
    }

    /**
     * Saves a setting to Chrome storage and notifies the content script.
     * 
     * @param {string} key - The setting key to save
     * @param {any} value - The value to save for the setting
     */
    function saveSettings(key, value) {
        console.log('Saving setting:', key, value);
        const settings = {};
        settings[key] = value;
        
        chrome.storage.sync.set(settings, function() {
            if (chrome.runtime.lastError) {
                console.error('Error saving settings:', chrome.runtime.lastError);
                return;
            }
            
            // Get current mode
            const currentMode = document.querySelector('.mode-button.active').id === 'faceDetectionMode' ? 'face_detection' : 'blur';
            
            // Notify content script of settings change with full settings context
            chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                if (tabs[0]) {
                    chrome.tabs.sendMessage(tabs[0].id, {
                        type: 'SETTINGS_UPDATED',
                        settings: {
                            processingMode: currentMode,
                            autoProcessImages: autoProcessCheckbox.checked,
                            addLabel: addLabelCheckbox.checked,
                            frameFaceDetected: frameFaceCheckbox.checked,
                            confidenceThreshold: parseInt(confidenceSlider.value),
                            ...settings
                        }
                    }, function(response) {
                        if (chrome.runtime.lastError) {
                            console.error('Error sending message:', chrome.runtime.lastError);
                        } else {
                            console.log('Settings update sent to content script');
                        }
                    });
                }
            });
        });
    }

    /**
     * Displays a status message to the user.
     * Shows a temporary notification that fades after 2 seconds.
     * 
     * @param {string} message - The status message to display
     */
    function showStatus(message) {
        console.log('Status:', message);
        if (statusIndicator) {
            statusIndicator.textContent = message;
            statusIndicator.classList.add('show');
            
            setTimeout(() => {
                statusIndicator.classList.remove('show');
            }, 2000);
        }
    }

    //--------------------------------------------------------------------------
    // DIAGNOSTIC DIALOG FUNCTIONALITY
    //--------------------------------------------------------------------------
    
    const diagnosticDialog = document.getElementById('diagnosticDialog');
    const diagnosticMessages = document.getElementById('diagnosticMessages');
    const closeDiagnosticDialog = document.getElementById('closeDiagnosticDialog');
    
    // Initialize diagnostic message storage
    let diagnosticMessageHistory = [];
    const MAX_MESSAGES = 50;
    
    /**
     * Adds a diagnostic message to the diagnostic dialog.
     * Creates a timestamped entry with appropriate styling based on message level.
     * 
     * @param {string} level - Message level ('info', 'warning', 'error')
     * @param {string} text - The message text to display
     */
    function addDiagnosticMessage(level, text) {
        // Make sure diagnostic dialog elements exist
        if (!diagnosticDialog || !diagnosticMessages) {
            console.error('Diagnostic dialog elements not found');
            return;
        }
        
        // Create timestamp
        const now = new Date();
        const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
        
        // Add to history
        diagnosticMessageHistory.push({ timestamp, level, text });
        
        // Trim history if needed
        if (diagnosticMessageHistory.length > MAX_MESSAGES) {
            diagnosticMessageHistory = diagnosticMessageHistory.slice(-MAX_MESSAGES);
        }
        
        // Add to display
        const messageElement = document.createElement('div');
        messageElement.className = `diagnostic-message ${level}`;
        
        const timestampSpan = document.createElement('span');
        timestampSpan.className = 'timestamp';
        timestampSpan.textContent = timestamp;
        
        messageElement.appendChild(timestampSpan);
        messageElement.appendChild(document.createTextNode(text));
        
        diagnosticMessages.appendChild(messageElement);
        
        // Auto-scroll
        diagnosticMessages.scrollTop = diagnosticMessages.scrollHeight;
        
        // Show the dialog if it's not already visible
        diagnosticDialog.classList.add('show');
    }
    
    /**
     * Event handler for the diagnostic dialog close button.
     * Hides the diagnostic dialog when clicked.
     */
    if (closeDiagnosticDialog) {
        closeDiagnosticDialog.addEventListener('click', function() {
            if (diagnosticDialog) {
                diagnosticDialog.classList.remove('show');
            }
        });
    }
    
    //--------------------------------------------------------------------------
    // DEVELOPMENT TOOLS - Can be removed in production
    //--------------------------------------------------------------------------
    
    /**
     * Adds a test diagnostic button for development purposes.
     * This can be removed in production builds.
     */
    const actionsSection = document.querySelector('.actions-section');
    if (actionsSection) {
        const testDiagnosticButton = document.createElement('button');
        testDiagnosticButton.textContent = 'Test Diagnostic';
        testDiagnosticButton.className = 'action-button';
        testDiagnosticButton.style.marginTop = '8px';
        testDiagnosticButton.addEventListener('click', function() {
            const levels = ['info', 'warning', 'error'];
            const level = levels[Math.floor(Math.random() * levels.length)];
            addDiagnosticMessage(level, `Test diagnostic message with ${level} level`);
        });
        actionsSection.appendChild(testDiagnosticButton);
    }
}); 