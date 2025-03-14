document.addEventListener('DOMContentLoaded', function() {
    console.log('Popup script loaded');

    // Get all UI elements
    const faceDetectionButton = document.getElementById('faceDetectionMode');
    const blurButton = document.getElementById('blurMode');
    const autoProcessCheckbox = document.getElementById('autoProcessImages');
    const addLabelCheckbox = document.getElementById('addLabel');
    const frameFaceCheckbox = document.getElementById('frameFaceDetected');
    const confidenceSlider = document.getElementById('confidenceThreshold');
    const confidenceValue = document.getElementById('confidenceValue');
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

    // Load saved settings
    chrome.storage.sync.get({
        processingMode: 'face_detection',
        autoProcessImages: true,
        addLabel: true,
        frameFaceDetected: true,
        confidenceThreshold: 70
    }, function(items) {
        console.log('Loaded settings:', items);
        
        // Set initial UI state
        updateUIForMode(items.processingMode);
        
        autoProcessCheckbox.checked = items.autoProcessImages;
        addLabelCheckbox.checked = items.addLabel;
        frameFaceCheckbox.checked = items.frameFaceDetected;
        confidenceSlider.value = items.confidenceThreshold;
        confidenceValue.textContent = `${items.confidenceThreshold}%`;
        
        // Check for blurred images count
        updateBlurredImagesCount();
    });

    // Mode switching
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

    // Settings changes
    autoProcessCheckbox.addEventListener('change', function() {
        console.log('Auto process changed:', this.checked);
        saveSettings('autoProcessImages', this.checked);
        showStatus(this.checked ? 'Auto-processing enabled' : 'Auto-processing disabled');
    });

    addLabelCheckbox.addEventListener('change', function() {
        console.log('Show labels changed:', this.checked);
        saveSettings('addLabel', this.checked);
        showStatus(this.checked ? 'Labels enabled' : 'Labels disabled');
    });

    frameFaceCheckbox.addEventListener('change', function() {
        console.log('Show frames changed:', this.checked);
        saveSettings('frameFaceDetected', this.checked);
        showStatus(this.checked ? 'Face frames enabled' : 'Face frames disabled');
    });

    confidenceSlider.addEventListener('input', function() {
        confidenceValue.textContent = `${this.value}%`;
    });

    confidenceSlider.addEventListener('change', function() {
        console.log('Confidence threshold changed:', this.value);
        saveSettings('confidenceThreshold', parseInt(this.value));
        showStatus(`Confidence threshold set to ${this.value}%`);
    });

    // Reprocess button
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

    // Clear blur list button
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
    
    // Function to update the number of blurred images
    function updateBlurredImagesCount() {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    type: 'GET_BLUR_COUNT'
                }, function(response) {
                    if (response && response.success) {
                        const blurredCount = document.getElementById('blurredCount');
                        if (response.historicalCount !== undefined) {
                            blurredCount.textContent = `Active: ${response.count} | Auto-renewed: ${response.autoRenewedCount || 0} | Historical: ${response.historicalCount}`;
                            blurredCount.title = 'Active: Currently blurred images\nAuto-renewed: Images that were automatically reblurred\nHistorical: Previously blurred images';
                        } else {
                            blurredCount.textContent = `Blurred images: ${response.count}`;
                        }
                    } else {
                        const blurredCount = document.getElementById('blurredCount');
                        blurredCount.textContent = 'Blurred images: unavailable';
                    }
                });
            }
        });
    }

    // Helper function to update UI based on mode
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

    // Helper function to save settings
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

    // Helper function to show status messages
    function showStatus(message) {
        console.log('Status:', message);
        statusIndicator.textContent = message;
        statusIndicator.classList.add('show');
        
        setTimeout(() => {
            statusIndicator.classList.remove('show');
        }, 2000);
    }
}); 