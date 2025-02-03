// Get all the settings elements
const settings = {
    frameProsessedImage: document.getElementById('frameProsessedImage'),
    frameFaceDetected: document.getElementById('frameFaceDetected'),
    addLabel: document.getElementById('addLabel'),
    autoProcessImages: document.getElementById('autoProcessImages'),
    minimumImageSizeRange: document.getElementById('minimumImageSizeRange'),
    minimumImageSizeNumber: document.getElementById('minimumImageSizeNumber')
};

// Function to save and apply settings
function saveAndApplySettings(showStatus = true) {
    const newSettings = {
        frameProsessedImage: settings.frameProsessedImage.checked,
        frameFaceDetected: settings.frameFaceDetected.checked,
        addLabel: settings.addLabel.checked,
        autoProcessImages: settings.autoProcessImages.checked,
        minimumImageSize: parseInt(settings.minimumImageSizeNumber.value)
    };

    // Save to Chrome storage
    chrome.storage.sync.set(newSettings, () => {
        if (showStatus) {
            // Show saved message
            const status = document.querySelector('.status');
            status.classList.add('show');
            
            // Hide message after 2 seconds
            setTimeout(() => {
                status.classList.remove('show');
            }, 2000);
        }
        
        // Send message to content script to update settings
        chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
            chrome.tabs.sendMessage(tabs[0].id, {
                type: 'UPDATE_SETTINGS',
                settings: newSettings
            });
        });
    });
}

// Load saved settings when popup opens
document.addEventListener('DOMContentLoaded', () => {
    chrome.storage.sync.get({
        // Default values
        frameProsessedImage: true,
        frameFaceDetected: true,
        addLabel: true,
        autoProcessImages: true,
        minimumImageSize: 100
    }, (items) => {
        // Set checkbox states
        settings.frameProsessedImage.checked = items.frameProsessedImage;
        settings.frameFaceDetected.checked = items.frameFaceDetected;
        settings.addLabel.checked = items.addLabel;
        settings.autoProcessImages.checked = items.autoProcessImages;
        
        // Set range and number input values
        settings.minimumImageSizeRange.value = items.minimumImageSize;
        settings.minimumImageSizeNumber.value = items.minimumImageSize;
    });
});

// Add change listeners to checkboxes for immediate effect
['frameProsessedImage', 'frameFaceDetected', 'addLabel', 'autoProcessImages'].forEach(setting => {
    settings[setting].addEventListener('change', () => {
        saveAndApplySettings(false); // Don't show the status message for immediate changes
    });
});

// Sync range and number inputs
settings.minimumImageSizeRange.addEventListener('input', (e) => {
    settings.minimumImageSizeNumber.value = e.target.value;
});

settings.minimumImageSizeNumber.addEventListener('input', (e) => {
    let value = parseInt(e.target.value);
    if (value < 0) value = 0;
    if (value > 10000) value = 10000;
    settings.minimumImageSizeRange.value = value;
    e.target.value = value;
});

// Save button now uses the same function but shows the status
document.querySelector('.save-button').addEventListener('click', () => {
    saveAndApplySettings(true); // Show the status message
}); 