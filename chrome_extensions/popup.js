/**
 * @fileoverview Manages the extension's popup UI and settings.
 * Handles user interactions and synchronizes settings with Chrome storage.
 * @author Liron Farzam
 * @version 1.0.0
 */

// DOM Elements
const settings = {
    frameProsessedImage: document.getElementById('frameProsessedImage'),
    frameFaceDetected: document.getElementById('frameFaceDetected'),
    addLabel: document.getElementById('addLabel'),
    autoProcessImages: document.getElementById('autoProcessImages'),
    minimumImageSizeRange: document.getElementById('minimumImageSizeRange'),
    minimumImageSizeNumber: document.getElementById('minimumImageSizeNumber'),
    confidenceThreshold: document.getElementById('confidenceThreshold')
};

/**
 * Saves and applies settings to Chrome storage and active tabs
 * @param {boolean} showStatus - Whether to show the save status message
 */
function saveAndApplySettings(showStatus = true) {
    const newSettings = {
        frameProsessedImage: settings.frameProsessedImage.checked,
        frameFaceDetected: settings.frameFaceDetected.checked,
        addLabel: settings.addLabel.checked,
        autoProcessImages: settings.autoProcessImages.checked,
        minimumImageSize: parseInt(settings.minimumImageSizeNumber.value),
        confidenceThreshold: parseInt(settings.confidenceThreshold.value)
    };

    console.log('Saving new settings:', newSettings);

    // Save to Chrome storage
    chrome.storage.sync.set(newSettings, () => {
        if (chrome.runtime.lastError) {
            console.error('Error saving settings:', chrome.runtime.lastError);
            return;
        }
        
        if (showStatus) {
            showSaveStatus();
        }
        
        // Update content script settings
        updateContentScriptSettings(newSettings);
    });
}

/**
 * Shows the save status message briefly
 */
function showSaveStatus() {
    const status = document.querySelector('.status');
    status.classList.add('show');
    setTimeout(() => {
        status.classList.remove('show');
    }, 2000);
}

/**
 * Updates settings in active content script
 * @param {Object} settings - The new settings to apply
 */
function updateContentScriptSettings(settings) {
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
        if (!tabs[0]?.id) {
            console.error('No active tab found');
            return;
        }
        
        console.log('Sending settings update:', settings);
        chrome.tabs.sendMessage(tabs[0].id, {
            type: 'UPDATE_SETTINGS',
            settings: settings
        }, (response) => {
            if (chrome.runtime.lastError) {
                console.error('Error sending settings:', chrome.runtime.lastError);
                return;
            }
            console.log('Settings update response:', response);
        });
    });
}

/**
 * Loads saved settings from Chrome storage
 */
function loadSavedSettings() {
    chrome.storage.sync.get({
        // Default values
        frameProsessedImage: true,
        frameFaceDetected: true,
        addLabel: true,
        autoProcessImages: true,
        minimumImageSize: 100,
        confidenceThreshold: 70
    }, (items) => {
        // Set checkbox states
        settings.frameProsessedImage.checked = items.frameProsessedImage;
        settings.frameFaceDetected.checked = items.frameFaceDetected;
        settings.addLabel.checked = items.addLabel;
        settings.autoProcessImages.checked = items.autoProcessImages;
        
        // Set range and number input values
        settings.minimumImageSizeRange.value = items.minimumImageSize;
        settings.minimumImageSizeNumber.value = items.minimumImageSize;

        // Set confidence threshold slider
        settings.confidenceThreshold.value = items.confidenceThreshold;
        updateThresholdValue(items.confidenceThreshold);
    });
}

function updateThresholdValue(value) {
    const thresholdValue = document.querySelector('.threshold-value');
    if (thresholdValue) {
        thresholdValue.textContent = `${value}%`;
    }
}

// Initialize event listeners
function initializeEventListeners() {
    // Add change listeners to checkboxes for immediate effect
    ['frameProsessedImage', 'frameFaceDetected', 'addLabel', 'autoProcessImages'].forEach(setting => {
        settings[setting].addEventListener('change', () => {
            saveAndApplySettings(false);
        });
    });

    // Sync range and number inputs
    settings.minimumImageSizeRange.addEventListener('input', (e) => {
        settings.minimumImageSizeNumber.value = e.target.value;
    });

    settings.minimumImageSizeNumber.addEventListener('input', (e) => {
        let value = parseInt(e.target.value);
        value = Math.max(0, Math.min(value, 10000));
        settings.minimumImageSizeRange.value = value;
        e.target.value = value;
        saveAndApplySettings(false);
    });

    // Save button handler
    document.querySelector('.save-button').addEventListener('click', () => {
        saveAndApplySettings(true);
    });

    // Add reprocess button handler
    document.querySelector('.reprocess-button').addEventListener('click', () => {
        chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
            if (!tabs[0]?.id) {
                console.error('No active tab found');
                return;
            }
            
            console.log('Requesting reprocess of all images');
            chrome.tabs.sendMessage(tabs[0].id, {
                type: 'REPROCESS_ALL'
            }, (response) => {
                if (chrome.runtime.lastError) {
                    console.error('Error requesting reprocess:', chrome.runtime.lastError);
                    return;
                }
                console.log('Reprocess response:', response);
            });
        });
    });

    // Save settings when changed
    document.querySelectorAll('input[type="checkbox"], input[type="radio"]').forEach(input => {
        input.addEventListener('change', saveSettings);
    });

    // Add confidence threshold slider listener
    settings.confidenceThreshold.addEventListener('input', (e) => {
        updateThresholdValue(e.target.value);
        saveAndApplySettings(false);
    });
}

// Initialize popup
document.addEventListener('DOMContentLoaded', () => {
    loadSavedSettings();
    initializeEventListeners();
});

function saveSettings() {
    // Get current settings
    const settings = {
        frameProsessedImage: document.getElementById('frameProsessedImage').checked,
        frameFaceDetected: document.getElementById('frameFaceDetected').checked,
        addLabel: document.getElementById('addLabel').checked,
        autoProcessImages: document.getElementById('autoProcessImages').checked,
        confidenceThreshold: parseInt(document.getElementById('confidenceThreshold').value)
    };

    // Save to storage
    chrome.storage.sync.set(settings, function() {
        // Show save confirmation
        const status = document.querySelector('.status');
        status.classList.add('show');
        setTimeout(() => status.classList.remove('show'), 2000);

        // Notify content script
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs[0]?.id) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    type: 'SETTINGS_UPDATED',
                    settings: settings
                });
            }
        });
    });
} 