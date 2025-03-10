/**
 * @fileoverview Optimized popup UI manager
 * @author Liron Farzam
 * @version 1.0.0
 */

//=============================================================================
// Performance Optimizations
//=============================================================================
/**
 * Cache DOM elements and use a single mutation observer
 */
const UI = {
    elements: new Map(),
    observer: new MutationObserver(handleDOMChanges),
    cache: new Map(),
    
    init() {
        // Cache all important elements
        const selectors = {
            modes: '.mode-button',
            settings: '#settings-container',
            status: '.status-indicator',
            sliders: 'input[type="range"]',
            checkboxes: 'input[type="checkbox"]',
            advancedToggle: '#showAdvancedSettings',
            advancedSection: '#advancedSettings',
            memoryStats: '#memoryStats'
        };

        Object.entries(selectors).forEach(([key, selector]) => {
            this.elements.set(key, document.querySelectorAll(selector));
        });

        // Set up observer
        this.observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['class', 'disabled']
        });
    }
};

//=============================================================================
// Event Handling
//=============================================================================
/**
 * Optimized event delegation
 */
const EventHandler = {
    init() {
        document.body.addEventListener('click', this.handleClick);
        document.body.addEventListener('change', this.handleChange);
        document.body.addEventListener('input', this.handleInput);
    },

    handleClick(e) {
        const target = e.target.closest('[data-action]');
        if (!target) return;

        const action = target.dataset.action;
        Actions[action]?.(target);
    },

    handleChange: debounce((e) => {
        const target = e.target;
        if (!target.dataset.setting) return;

        Settings.update(target.dataset.setting, target.type === 'checkbox' ? target.checked : target.value);
    }, 250),

    handleInput(e) {
        const target = e.target;
        if (target.type === 'range') {
            UI.elements.get('sliderValues')?.forEach(el => {
                if (el.dataset.for === target.id) {
                    el.textContent = `${target.value}%`;
                }
            });
        }
    }
};

//=============================================================================
// Settings Management
//=============================================================================
/**
 * Optimized settings manager with caching
 */
const Settings = {
    async init() {
        const defaults = {
            processingMode: 'face_detection',
            autoProcessImages: true,
            addLabel: true,
            frameFaceDetected: true,
            confidenceThreshold: 70,
            progressiveProcessing: true,
            maxImagesPerPage: 100,
            memoryManagement: {
                enabled: true,
                maxTensors: 1000,
                maxBytes: 200 * 1024 * 1024, // 200MB
                cleanupInterval: 60000 // 1 minute
            }
        };

        try {
            const saved = await chrome.storage.sync.get(defaults);
            this.applySettings(saved);
            UI.cache.set('settings', saved);
            
            // Initialize advanced settings visibility
            this.toggleAdvancedSettings(false);
            
            // Initialize memory stats
            this.updateMemoryStats();
        } catch (error) {
            console.error('Settings initialization failed:', error);
            this.applySettings(defaults);
        }
    },

    async update(key, value) {
        const settings = UI.cache.get('settings') || {};
        
        // Handle nested settings
        if (key.includes('.')) {
            const [parent, child] = key.split('.');
            settings[parent] = settings[parent] || {};
            settings[parent][child] = value;
        } else {
            settings[key] = value;
        }

        try {
            await chrome.storage.sync.set(settings);
            await this.notifyContentScript(settings);
            UI.cache.set('settings', settings);
        } catch (error) {
            console.error('Settings update failed:', error);
            StatusManager.show('Settings update failed', 'error');
        }
    },
    
    async notifyContentScript(settings) {
        try {
            const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
            if (tabs.length > 0) {
                await chrome.tabs.sendMessage(tabs[0].id, { 
                    type: 'SETTINGS_UPDATED', 
                    settings 
                });
                StatusManager.show('Settings updated', 'success');
            }
        } catch (error) {
            console.error('Failed to notify content script:', error);
        }
    },
    
    applySettings(settings) {
        // Update mode selection
        const modeButtons = document.querySelectorAll('.mode-button');
        modeButtons.forEach(button => {
            button.classList.toggle('active', 
                button.id === `${settings.processingMode}Mode`);
        });
        
        // Update checkboxes
        document.getElementById('autoProcessImages').checked = 
            settings.autoProcessImages;
        document.getElementById('addLabel').checked = 
            settings.addLabel;
        document.getElementById('frameFaceDetected').checked = 
            settings.frameFaceDetected;
        
        // Update advanced settings
        if (document.getElementById('progressiveProcessing')) {
            document.getElementById('progressiveProcessing').checked = 
                settings.progressiveProcessing;
        }
        
        if (document.getElementById('maxImagesPerPage')) {
            document.getElementById('maxImagesPerPage').value = 
                settings.maxImagesPerPage;
        }
        
        if (document.getElementById('enableMemoryManagement')) {
            document.getElementById('enableMemoryManagement').checked = 
                settings.memoryManagement?.enabled;
        }
            
        // Update slider
        const slider = document.getElementById('confidenceThreshold');
        slider.value = settings.confidenceThreshold;
        document.getElementById('confidenceValue').textContent = 
            `${settings.confidenceThreshold}%`;
            
        // Update UI visibility based on mode
        updateUIForMode(settings.processingMode);
    },
    
    toggleAdvancedSettings(show) {
        const advancedSection = document.getElementById('advancedSettings');
        const showAdvancedToggle = document.getElementById('showAdvancedSettings');
        
        if (advancedSection) {
            if (show === undefined) {
                // Toggle current state
                show = advancedSection.classList.contains('hidden');
            }
            
            advancedSection.classList.toggle('hidden', !show);
            
            if (showAdvancedToggle) {
                showAdvancedToggle.textContent = show ? 'Hide Advanced Settings' : 'Show Advanced Settings';
            }
        }
    },
    
    async updateMemoryStats() {
        const memoryStatsElement = document.getElementById('memoryStats');
        if (!memoryStatsElement) return;
        
        try {
            const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
            if (tabs.length > 0) {
                chrome.tabs.sendMessage(tabs[0].id, { type: 'GET_MEMORY_STATS' }, (response) => {
                    if (chrome.runtime.lastError) {
                        console.error('Error getting memory stats:', chrome.runtime.lastError);
                        memoryStatsElement.textContent = 'Memory stats unavailable';
                        return;
                    }
                    
                    if (response && response.stats) {
                        const stats = response.stats;
                        memoryStatsElement.innerHTML = `
                            <div class="memory-stat">
                                <span>Tensors:</span> 
                                <span>${stats.numTensors || 'N/A'}</span>
                            </div>
                            <div class="memory-stat">
                                <span>Memory:</span> 
                                <span>${stats.memoryMB || 'N/A'} MB</span>
                            </div>
                            <div class="memory-stat">
                                <span>Images Processed:</span> 
                                <span>${stats.imagesProcessed || 'N/A'}</span>
                            </div>
                            <div class="memory-stat">
                                <span>Cache Size:</span> 
                                <span>${stats.cacheSize || 'N/A'} items</span>
                            </div>
                        `;
                    } else {
                        memoryStatsElement.textContent = 'Memory stats unavailable';
                    }
                });
            }
        } catch (error) {
            console.error('Failed to get memory stats:', error);
            memoryStatsElement.textContent = 'Error retrieving memory stats';
        }
        
        // Schedule next update
        setTimeout(() => this.updateMemoryStats(), 5000);
    }
};

//=============================================================================
// Status Management
//=============================================================================
/**
 * Improved status manager with animation handling
 */
const StatusManager = {
    queue: [],
    isShowing: false,

    show(message, type = 'success', duration = 2000) {
        this.queue.push({ message, type, duration });
        if (!this.isShowing) {
            this.processQueue();
        }
    },

    async processQueue() {
        if (this.queue.length === 0) {
            this.isShowing = false;
            return;
        }

        this.isShowing = true;
        const { message, type, duration } = this.queue.shift();
        const statusElement = document.querySelector('.status-indicator');

        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status-indicator show ${type}`;

            await new Promise(resolve => setTimeout(resolve, duration));
            statusElement.classList.remove('show');

            // Wait for animation
            await new Promise(resolve => setTimeout(resolve, 300));
        }
        
        this.processQueue();
    }
};

//=============================================================================
// Actions
//=============================================================================
/**
 * Action handlers for UI interactions
 */
const Actions = {
    setMode(element) {
        const mode = element.id.replace('Mode', '');
        Settings.update('processingMode', mode);
        
        // Update UI
        document.querySelectorAll('.mode-button').forEach(btn => {
            btn.classList.toggle('active', btn === element);
        });
        
        updateUIForMode(mode);
    },
    
    reprocessImages() {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs.length > 0) {
                chrome.tabs.sendMessage(tabs[0].id, { type: 'REPROCESS_IMAGES' });
                StatusManager.show('Reprocessing images...', 'info');
            }
        });
    },
    
    toggleAdvancedSettings() {
        Settings.toggleAdvancedSettings();
    },
    
    cleanupMemory() {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs.length > 0) {
                chrome.tabs.sendMessage(tabs[0].id, { type: 'CLEANUP_MEMORY' });
                StatusManager.show('Memory cleanup initiated', 'info');
            }
        });
    }
};

//=============================================================================
// Helper Functions
//=============================================================================
/**
 * Update UI based on selected mode
 */
function updateUIForMode(mode) {
    document.querySelectorAll('[data-mode]').forEach(el => {
        const modes = el.dataset.mode.split(',');
        el.classList.toggle('hidden', 
            !modes.includes(mode) && !modes.includes('both'));
    });
}

/**
 * Handle DOM changes for dynamic UI updates
 */
function handleDOMChanges(mutations) {
    // Implementation depends on specific UI needs
}

/**
 * Debounce function to limit rapid calls
 */
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

//=============================================================================
// Initialization
//=============================================================================
document.addEventListener('DOMContentLoaded', function() {
    // Check if extension is disabled due to refresh loop
    checkRefreshLoopDisabled();
    
    // Set up event listeners
    setupEventListeners();
    
    // Load settings
    loadSettings();
});

// Function to check if extension is disabled due to refresh loop
function checkRefreshLoopDisabled() {
    const refreshLoopAlert = document.getElementById('refreshLoopAlert');
    
    try {
        // Check local storage for disabled flags
        const disabledUntil = localStorage.getItem('faceone_disabled_until');
        const isPermanentlyDisabled = localStorage.getItem('faceone_permanently_disabled') === 'true';
        
        if (disabledUntil || isPermanentlyDisabled) {
            refreshLoopAlert.style.display = 'block';
            
            // If it's temporarily disabled, show the remaining time
            if (disabledUntil) {
                const now = Date.now();
                const disabledTime = parseInt(disabledUntil);
                
                if (disabledTime > now) {
                    const remainingMinutes = Math.ceil((disabledTime - now) / 60000);
                    const timeMessage = document.createElement('p');
                    timeMessage.textContent = `Time remaining: ${remainingMinutes} minutes`;
                    refreshLoopAlert.appendChild(timeMessage);
                }
            }
            
            // If it's permanently disabled, show a more serious message
            if (isPermanentlyDisabled) {
                const permanentMessage = document.createElement('p');
                permanentMessage.innerHTML = '<strong>Permanently disabled!</strong> This is due to excessive page refreshing.';
                refreshLoopAlert.appendChild(permanentMessage);
            }
        }
    } catch (error) {
        console.error('Error checking refresh loop disabled status:', error);
    }
}

// Function to set up event listeners
function setupEventListeners() {
    // Enable toggle
    const enableToggle = document.getElementById('enableToggle');
    enableToggle.addEventListener('change', function() {
        saveSettings({ enabled: this.checked });
        showStatus(this.checked ? 'Extension enabled' : 'Extension disabled', this.checked ? 'success' : 'warning');
    });
    
    // Min size slider
    const minSizeSlider = document.getElementById('minSizeSlider');
    const minSizeValue = document.getElementById('minSizeValue');
    
    minSizeSlider.addEventListener('input', function() {
        minSizeValue.textContent = this.value + 'px';
    });
    
    minSizeSlider.addEventListener('change', function() {
        saveSettings({ minimumImageSize: parseInt(this.value) });
        showStatus('Settings saved', 'success');
    });
    
    // Reset settings button
    const resetSettingsBtn = document.getElementById('resetSettingsBtn');
    resetSettingsBtn.addEventListener('click', function() {
        resetSettings();
        showStatus('Settings reset to defaults', 'success');
    });
    
    // Clear cache button
    const clearCacheBtn = document.getElementById('clearCacheBtn');
    clearCacheBtn.addEventListener('click', function() {
        clearCache();
        showStatus('Cache cleared successfully', 'success');
    });
    
    // Reset refresh loop button
    const resetRefreshLoopBtn = document.getElementById('resetRefreshLoopBtn');
    if (resetRefreshLoopBtn) {
        resetRefreshLoopBtn.addEventListener('click', function() {
            resetRefreshLoop();
            showStatus('Refresh loop detection reset. Reloading...', 'success');
            setTimeout(() => {
                chrome.tabs.reload();
            }, 1500);
        });
    }
}

// Function to show status message
function showStatus(message, type) {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;
    statusElement.className = 'status ' + type;
    statusElement.style.display = 'block';
    
    // Hide after 3 seconds
    setTimeout(() => {
        statusElement.style.display = 'none';
    }, 3000);
}

// Function to load settings
function loadSettings() {
    chrome.storage.sync.get({
        // Default settings
        enabled: true,
        minimumImageSize: 40,
        processingMode: 'face_detection',
        showLabel: true,
        showFaceFrame: true,
        confidenceThreshold: 70,
        autoProcessImages: true
    }, function(settings) {
        // Apply settings to UI
        document.getElementById('enableToggle').checked = settings.enabled;
        
        const minSizeSlider = document.getElementById('minSizeSlider');
        const minSizeValue = document.getElementById('minSizeValue');
        
        if (minSizeSlider && minSizeValue) {
            minSizeSlider.value = settings.minimumImageSize;
            minSizeValue.textContent = settings.minimumImageSize + 'px';
        }
    });
}

// Function to save settings
function saveSettings(settings) {
    chrome.storage.sync.get({
        // Default settings
        enabled: true,
        minimumImageSize: 40,
        processingMode: 'face_detection',
        showLabel: true,
        showFaceFrame: true,
        confidenceThreshold: 70,
        autoProcessImages: true
    }, function(existingSettings) {
        // Merge new settings with existing settings
        const updatedSettings = { ...existingSettings, ...settings };
        
        // Save to storage
        chrome.storage.sync.set(updatedSettings, function() {
            console.log('Settings saved:', updatedSettings);
            
            // Send settings to content script if we're on an active tab
            chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
                if (tabs.length > 0) {
                    chrome.tabs.sendMessage(tabs[0].id, { 
                        action: 'updateSettings', 
                        settings: updatedSettings 
                    });
                }
            });
        });
    });
}

// Function to reset settings
function resetSettings() {
    const defaultSettings = {
        enabled: true,
        minimumImageSize: 40,
        processingMode: 'face_detection',
        showLabel: true,
        showFaceFrame: true,
        confidenceThreshold: 70,
        autoProcessImages: true
    };
    
    // Save default settings
    chrome.storage.sync.set(defaultSettings, function() {
        console.log('Settings reset to defaults');
        
        // Reload settings in UI
        loadSettings();
        
        // Send reset to content script
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            if (tabs.length > 0) {
                chrome.tabs.sendMessage(tabs[0].id, { 
                    action: 'updateSettings', 
                    settings: defaultSettings 
                });
            }
        });
    });
}

// Function to clear cache
function clearCache() {
    chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
        if (tabs.length > 0) {
            chrome.tabs.sendMessage(tabs[0].id, { action: 'clearCache' });
        }
    });
}

// Function to reset refresh loop detection
function resetRefreshLoop() {
    try {
        // Clear session storage refresh counter
        sessionStorage.removeItem('faceone_page_refresh');
        
        // Clear local storage keys
        localStorage.removeItem('faceone_disabled_until');
        localStorage.removeItem('faceone_permanently_disabled');
        localStorage.removeItem('faceone_permanent_disable'); // Old key for backward compatibility
        
        // Hide the warning
        const refreshLoopAlert = document.getElementById('refreshLoopAlert');
        if (refreshLoopAlert) {
            refreshLoopAlert.style.display = 'none';
        }
        
        console.log('Refresh loop detection reset successfully');
        
        // Send reset message to content script
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            if (tabs.length > 0) {
                chrome.tabs.sendMessage(tabs[0].id, { action: 'resetRefreshLoop' });
            }
        });
    } catch (error) {
        console.error('Error resetting refresh loop detection:', error);
        showStatus('Error resetting refresh loop: ' + error.message, 'error');
    }
} 