/**
 * @fileoverview Manages the extension's popup UI and settings.
 * Handles user interactions and synchronizes settings with Chrome storage.
 * @author Liron Farzam
 * @version 1.0.0
 */

//=============================================================================
// Global Variables and Constants
//=============================================================================
/**
 * UI element references
 * @type {Object.<string, HTMLElement>}
 */
const settings = {
    frameProsessedImage: null,
    frameFaceDetected: null,
    addLabel: null,
    autoProcessImages: null,
    minimumImageSizeRange: null,
    minimumImageSizeNumber: null,
    confidenceThreshold: null
};

/**
 * Default settings values
 * @type {Object}
 */
const DEFAULT_SETTINGS = {
    processingMode: 'face_detection',
    autoProcessImages: true,
    addLabel: true,
    frameFaceDetected: true,
    confidenceThreshold: 70
};

//=============================================================================
// Initialization
//=============================================================================
/**
 * Initializes the popup UI and functionality
 */
async function initializePopup() {
    try {
        await loadSavedSettings();
        initializeEventListeners();
        console.log('Popup initialized successfully');
    } catch (error) {
        console.error('Error initializing popup:', error);
        showStatus('Error initializing popup', 'error');
    }
}

//=============================================================================
// Settings Management
//=============================================================================
/**
 * Gets current settings from UI elements
 * @returns {Object} Current settings object
 */
function getCurrentSettings() {
    return {
        processingMode: document.querySelector('.mode-button.active').id === 'faceDetectionMode' ? 
            'face_detection' : 'blur',
        frameProsessedImage: settings.frameProsessedImage.checked,
        frameFaceDetected: settings.frameFaceDetected.checked,
        addLabel: settings.addLabel.checked,
        autoProcessImages: settings.autoProcessImages.checked,
        minimumImageSize: parseInt(settings.minimumImageSizeNumber.value),
        confidenceThreshold: parseInt(settings.confidenceThreshold.value)
    };
}

/**
 * Updates settings in Chrome storage and notifies content script
 * @param {Object} newSettings - Settings to update
 * @param {boolean} showStatus - Whether to show the save status message
 * @returns {Promise} Resolves when settings are saved and applied
 */
async function updateSettings(newSettings, showStatus = true) {
    console.log('Saving settings:', newSettings);

    try {
        // Save to Chrome storage
        await new Promise((resolve, reject) => {
            chrome.storage.sync.set(newSettings, () => {
                if (chrome.runtime.lastError) {
                    reject(chrome.runtime.lastError);
                } else {
                    resolve();
                }
            });
        });

        // Show status if requested
        if (showStatus) {
            showSaveStatus();
        }

        // Notify content script
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tabs[0]?.id) {
            await chrome.tabs.sendMessage(tabs[0].id, {
                type: 'SETTINGS_UPDATED',
                settings: newSettings
            });
        }

        return true;
    } catch (error) {
        console.error('Error updating settings:', error);
        showStatus('Error saving settings', 'error');
        return false;
    }
}

/**
 * Loads saved settings from Chrome storage
 */
async function loadSavedSettings() {
    const defaults = {
        processingMode: 'face_detection',
        frameProsessedImage: true,
        frameFaceDetected: true,
        addLabel: true,
        autoProcessImages: true,
        minimumImageSize: 100,
        confidenceThreshold: 70
    };

    try {
        const items = await new Promise(resolve => {
            chrome.storage.sync.get(defaults, resolve);
        });

        // Update UI elements
        settings.frameProsessedImage.checked = items.frameProsessedImage;
        settings.frameFaceDetected.checked = items.frameFaceDetected;
        settings.addLabel.checked = items.addLabel;
        settings.autoProcessImages.checked = items.autoProcessImages;
        settings.minimumImageSizeRange.value = items.minimumImageSize;
        settings.minimumImageSizeNumber.value = items.minimumImageSize;
        settings.confidenceThreshold.value = items.confidenceThreshold;

        // Update mode buttons
        const faceDetectionButton = document.getElementById('faceDetectionMode');
        const blurButton = document.getElementById('blurMode');
        if (items.processingMode === 'face_detection') {
            faceDetectionButton.classList.add('active');
            blurButton.classList.remove('active');
        } else {
            blurButton.classList.add('active');
            faceDetectionButton.classList.remove('active');
        }

        updateThresholdValue(items.confidenceThreshold);
    } catch (error) {
        console.error('Error loading settings:', error);
        showStatus('Error loading settings', 'error');
    }
}

/**
 * Saves settings to Chrome storage and notifies content script
 * @param {string} key - Setting key to update
 * @param {any} value - New value for the setting
 */
function saveSettings(key, value) {
    // ... existing saveSettings code ...
}

//=============================================================================
// UI Management
//=============================================================================
/**
 * Updates UI based on selected mode
 * @param {string} mode - The selected processing mode
 */
function updateUIForMode(mode) {
    // ... existing updateUIForMode code ...
}

/**
 * Shows status message in the UI
 * @param {string} message - Message to display
 * @param {string} [type='success'] - Type of message
 */
function showStatus(message, type = 'success') {
    const status = document.querySelector('.status');
    status.textContent = message;
    status.className = `status show ${type}`;
    setTimeout(() => {
        status.classList.remove('show');
    }, 2000);
}

/**
 * Updates threshold value display
 * @param {number} value - New threshold value
 */
function updateThresholdValue(value) {
    const thresholdValue = document.querySelector('.threshold-value');
    if (thresholdValue) {
        thresholdValue.textContent = `${value}%`;
    }
}

//=============================================================================
// Event Listeners
//=============================================================================
/**
 * Initializes all event listeners for UI elements
 */
function initializeEventListeners() {
    // Mode switching
    ['faceDetectionMode', 'blurMode'].forEach(id => {
        const button = document.getElementById(id);
        button.addEventListener('click', async () => {
            if (button.classList.contains('active')) return;
            
            const newSettings = getCurrentSettings();
            newSettings.processingMode = id === 'faceDetectionMode' ? 'face_detection' : 'blur';
            await updateSettings(newSettings, true);
        });
    });

    // Checkbox changes
    ['frameProsessedImage', 'frameFaceDetected', 'addLabel', 'autoProcessImages'].forEach(id => {
        settings[id].addEventListener('change', async () => {
            await updateSettings(getCurrentSettings(), false);
        });
    });

    // Range and number input sync
    settings.minimumImageSizeRange.addEventListener('input', (e) => {
        settings.minimumImageSizeNumber.value = e.target.value;
    });

    settings.minimumImageSizeNumber.addEventListener('input', async (e) => {
        let value = parseInt(e.target.value);
        value = Math.max(0, Math.min(value, 10000));
        settings.minimumImageSizeRange.value = value;
        e.target.value = value;
        await updateSettings(getCurrentSettings(), false);
    });

    // Confidence threshold changes
    settings.confidenceThreshold.addEventListener('input', (e) => {
        updateThresholdValue(e.target.value);
    });

    settings.confidenceThreshold.addEventListener('change', async () => {
        await updateSettings(getCurrentSettings(), false);
    });

    // Reprocess button
    document.querySelector('.reprocess-button')?.addEventListener('click', async () => {
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tabs[0]?.id) {
            try {
                await chrome.tabs.sendMessage(tabs[0].id, { type: 'REPROCESS_ALL' });
                showStatus('Reprocessing images...', 'success');
            } catch (error) {
                console.error('Error requesting reprocess:', error);
                showStatus('Error reprocessing images', 'error');
            }
        }
    });
}

//=============================================================================
// Initialization Call
//=============================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Initialize DOM elements
    Object.keys(settings).forEach(key => {
        const element = document.getElementById(key);
        if (!element) {
            console.warn(`Element not found: ${key}`);
        }
        settings[key] = element;
    });

    // Initialize popup
    initializePopup().catch(error => {
        console.error('Failed to initialize popup:', error);
        showStatus('Failed to initialize popup', 'error');
    });
}); 