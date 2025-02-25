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
            checkboxes: 'input[type="checkbox"]'
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
            confidenceThreshold: 70
        };

        try {
            const saved = await chrome.storage.sync.get(defaults);
            this.applySettings(saved);
            UI.cache.set('settings', saved);
        } catch (error) {
            console.error('Settings initialization failed:', error);
            this.applySettings(defaults);
        }
    },

    async update(key, value) {
        const settings = UI.cache.get('settings') || {};
        settings[key] = value;

        try {
            await chrome.storage.sync.set({ [key]: value });
            await this.notifyContentScript(settings);
            UI.cache.set('settings', settings);
        } catch (error) {
            console.error('Settings update failed:', error);
            StatusManager.show('Settings update failed', 'error');
        }
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
        const status = UI.elements.get('status');

        status.textContent = message;
        status.className = `status show ${type}`;

        await new Promise(resolve => setTimeout(resolve, duration));
        status.classList.remove('show');

        // Wait for animation
        await new Promise(resolve => setTimeout(resolve, 300));
        this.processQueue();
    }
};

//=============================================================================
// Initialization
//=============================================================================
document.addEventListener('DOMContentLoaded', async () => {
    UI.init();
    EventHandler.init();
    await Settings.init();
}); 