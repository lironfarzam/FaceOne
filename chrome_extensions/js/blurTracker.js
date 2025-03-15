/**
 * @fileoverview blurTracker.js - Lightweight URL-based image blurring system
 * 
 * @author Liron Farzam
 * @version 1.0.0
 * @license MIT
 * 
 * A lightweight system for tracking and managing image URLs that need to be blurred.
 * This module only stores URLs of images that require blurring, without keeping embeddings,
 * making it memory and storage efficient. It handles DOM mutations to catch dynamically 
 * added images, normalizes URLs (especially for Facebook images), and manages the 
 * lifecycle of blurred image URLs with automatic expiration and historical tracking.
 */

//==============================================================================
// MAIN BLUR TRACKER OBJECT
//==============================================================================

const blurTracker = {
    //--------------------------------------------------------------------------
    // PROPERTIES
    //--------------------------------------------------------------------------
    
    /**
     * Map that stores active image URLs with their timestamp of when they were marked
     * @type {Map<string, number>} - Map of normalized URL to timestamp
     */
    blurredImages: new Map(),
    
    /**
     * Map that stores historical records of previously blurred images
     * @type {Map<string, Object>} - Map of normalized URL to historical data object
     */
    historicalBlurs: new Map(),
    
    /**
     * Minimum age to keep entries in active list (30 minutes in milliseconds)
     * @type {number}
     * @constant
     */
    MIN_AGE_MS: 30 * 60 * 1000,
    
    /**
     * Maximum age to keep historical entries (7 days in milliseconds)
     * @type {number}
     * @constant
     */
    HISTORY_MAX_AGE: 7 * 24 * 60 * 60 * 1000,
    
    /**
     * Flag indicating whether data needs to be saved to storage
     * @type {boolean}
     */
    needsSave: false,
    
    //--------------------------------------------------------------------------
    // INITIALIZATION METHODS
    //--------------------------------------------------------------------------
    
    /**
     * Initializes the blur tracker by loading stored data and setting up observers
     * 
     * This method:
     * 1. Loads previously saved blur data from chrome.storage.local
     * 2. Sets up periodic save timers
     * 3. Initializes the mutation observer for DOM changes
     * 
     * @returns {void}
     */
    init() {
        logWithEmoji('setup', 'blurTracker.init', 'Initializing blurTracker');
        chrome.storage.local.get(['blurredImageData', 'historicalBlurData'], (result) => {
            // Load current blur data
            if (result.blurredImageData) {
                try {
                    // Convert stored array back to Map
                    const storedData = result.blurredImageData;
                    this.blurredImages = new Map();
                    
                    // Restore the data with validity check
                    for (const [url, timestamp] of storedData) {
                        // Only restore if URL is valid
                        if (url && typeof url === 'string') {
                            this.blurredImages.set(url, timestamp);
                        }
                    }
                    
                    logWithEmoji('success', 'blurTracker.init', `Loaded ${this.blurredImages.size} previously blurred image URLs`, 
                        { count: this.blurredImages.size });
                } catch (error) {
                    logWithEmoji('error', 'blurTracker.init', 'Error loading blurred image data', error);
                    // Reset to empty Map on error
                    this.blurredImages = new Map();
                }
            }
            
            // Load historical blur data
            if (result.historicalBlurData) {
                try {
                    // Convert stored array back to Map
                    const storedHistorical = result.historicalBlurData;
                    this.historicalBlurs = new Map();
                    
                    // Restore the historical data with validity check
                    for (const [url, data] of storedHistorical) {
                        // Only restore if URL is valid
                        if (url && typeof url === 'string') {
                            this.historicalBlurs.set(url, data);
                        }
                    }
                    
                    logWithEmoji('success', 'blurTracker.init', `Loaded ${this.historicalBlurs.size} historical blurred image URLs`,
                        { count: this.historicalBlurs.size });
                } catch (error) {
                    logWithEmoji('error', 'blurTracker.init', 'Error loading historical blur data', error);
                    // Reset to empty Map on error
                    this.historicalBlurs = new Map();
                }
            }
        });
        
        // Set up periodic save timer
        this.startSaveTimer();
        
        // Set up MutationObserver to catch newly added images
        this.setupMutationObserver();
    },
    
    /**
     * Sets up timers to periodically save blur data to storage
     * 
     * Sets up three mechanisms for saving:
     * 1. Every minute if changes are detected
     * 2. Every 5 minutes regardless of changes
     * 3. On page unload if changes are detected
     * 
     * @returns {void}
     */
    startSaveTimer() {
        // Save every minute if needed
        setInterval(() => {
            if (this.needsSave) {
                this.save(true);
            }
        }, 60 * 1000);
        
        // Force save every 5 minutes regardless
        setInterval(() => {
            this.save(true);
        }, 5 * 60 * 1000);
        
        // Save on page unload
        window.addEventListener('beforeunload', () => {
            if (this.needsSave) {
                this.save(true);
            }
        });
    },
    
    /**
     * Sets up a MutationObserver to detect and process newly added images in the DOM
     * 
     * This method is critical for catching dynamically loaded images, especially
     * in infinite scrolling scenarios or when content is loaded after initial page load.
     * 
     * @returns {void}
     * @throws Will silently fail if MutationObserver is not supported or document is not available
     */
    setupMutationObserver() {
        // Only set up if we're in a browser environment with MutationObserver
        if (typeof MutationObserver !== 'undefined' && document && document.body) {
            const observer = new MutationObserver((mutations) => {
                for (const mutation of mutations) {
                    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                        this.processAddedNodes(mutation.addedNodes);
                    }
                }
            });
            
            // Start observing the document with the configured parameters
            observer.observe(document.body, {
                childList: true,  // Watch for changes to the direct children
                subtree: true,    // Watch for changes to the entire subtree
                attributes: false // Don't track attribute changes (optimized for performance)
            });
            
            logWithEmoji('setup', 'blurTracker.setupMutationObserver', 'MutationObserver setup complete');
        }
    },
    
    /**
     * Processes nodes added to the DOM, identifying and blurring images as needed
     * 
     * Recursively checks all newly added DOM nodes for images and processes them
     * for potential blurring based on stored URLs.
     * 
     * @param {NodeList} nodes - Collection of DOM nodes to process
     * @returns {void}
     */
    processAddedNodes(nodes) {
        for (const node of nodes) {
            // Skip non-element nodes
            if (node.nodeType !== Node.ELEMENT_NODE) continue;
            
            // Check if this node is an image
            if (node.tagName === 'IMG') {
                this.applyBlurIfNeeded(node);
            }
            
            // Recursively check child nodes
            if (node.querySelectorAll) {
                const images = node.querySelectorAll('img');
                for (const img of images) {
                    this.applyBlurIfNeeded(img);
                }
            }
        }
    },
    
    //--------------------------------------------------------------------------
    // IMAGE PROCESSING METHODS
    //--------------------------------------------------------------------------
    
    /**
     * Applies blur to an image element if its URL is in the blur list
     * 
     * Checks if an image should be blurred based on its URL, and applies
     * CSS blur effect if necessary. Also handles auto-renewal of previously
     * blurred images.
     * 
     * @param {HTMLImageElement} img - The image element to check and potentially blur
     * @returns {void}
     */
    applyBlurIfNeeded(img) {
        if (!img || !img.src) return;
        
        const normalizedUrl = this.normalizeImageUrl(img.src);
        if (this.shouldBlur(normalizedUrl)) {
            // Apply blur to actively tracked images
            logWithEmoji('image', 'blurTracker.applyBlurIfNeeded', 'Applying blur to dynamically added image', 
                { url: normalizedUrl.substring(0, 50) + '...' });
            img.style.filter = 'blur(10px)';
            img.classList.add('blurred-image');
            img.setAttribute('data-faceone-processed', 'blurred');
        } else if (this.wasBlurredBefore(normalizedUrl)) {
            // If it was blurred before but expired, automatically reactivate it
            logWithEmoji('image', 'blurTracker.applyBlurIfNeeded', 'Reactivating previously blurred image', 
                { url: normalizedUrl.substring(0, 50) + '...' });
            this.markForBlur(normalizedUrl);
            img.style.filter = 'blur(10px)';
            img.classList.add('blurred-image');
            img.setAttribute('data-faceone-processed', 'blurred');
            img.setAttribute('data-auto-renewed', 'true');
        }
    },
    
    /**
     * Normalizes an image URL to handle platform-specific quirks
     * 
     * Particularly focused on Facebook image URLs which often contain changing
     * parameters while the core image remains the same. Strips out changing
     * parameters but keeps essential ones.
     * 
     * @param {string} url - The original image URL to normalize
     * @returns {string} - The normalized URL, or original URL if normalization fails
     */
    normalizeImageUrl(url) {
        try {
            // For Facebook images, strip out changing parameters but keep essential ones
            if (url.includes('fbcdn.net') || url.includes('facebook.com')) {
                const urlObj = new URL(url);
                
                // Keep only essential parameters for Facebook image URLs
                const essentialParams = ['stp', 'dst-jpg', 'set'];
                const searchParams = new URLSearchParams();
                
                for (const param of essentialParams) {
                    if (urlObj.searchParams.has(param)) {
                        searchParams.set(param, urlObj.searchParams.get(param));
                    }
                }
                
                // Build normalized URL with just the path and essential params
                let normalizedUrl = urlObj.origin + urlObj.pathname;
                if (searchParams.toString()) {
                    normalizedUrl += '?' + searchParams.toString();
                }
                
                return normalizedUrl;
            }
            
            // Return original URL for non-Facebook images
            return url;
        } catch (e) {
            logWithEmoji('error', 'blurTracker.normalizeImageUrl', 'Error normalizing URL', e);
            return url; // Return original if parsing fails
        }
    },
    
    //--------------------------------------------------------------------------
    // BLUR STATE MANAGEMENT METHODS
    //--------------------------------------------------------------------------
    
    /**
     * Marks an image URL for blurring
     * 
     * Adds the image URL to the active blur list with the current timestamp.
     * If the URL was previously in the historical list, it's removed from there.
     * 
     * @param {string} imageUrl - The URL of the image to mark for blurring
     * @returns {void}
     */
    markForBlur(imageUrl) {
        if (!imageUrl || typeof imageUrl !== 'string') return;
        
        const normalizedUrl = this.normalizeImageUrl(imageUrl);
        
        // Remove from historical blurs if it exists there
        if (this.historicalBlurs.has(normalizedUrl)) {
            this.historicalBlurs.delete(normalizedUrl);
        }
        
        // Add to active blur list with current timestamp
        this.blurredImages.set(normalizedUrl, Date.now());
        this.needsSave = true;
        logWithEmoji('image', 'blurTracker.markForBlur', 'Marked image for blur', 
            { url: normalizedUrl.substring(0, 50) + '...' });
    },
    
    /**
     * Checks if an image URL should be blurred
     * 
     * Determines if an image URL is in the active blur list. If it is,
     * updates its timestamp to extend its life in the active list.
     * 
     * @param {string} imageUrl - The URL of the image to check
     * @returns {boolean} - True if the image should be blurred, false otherwise
     */
    shouldBlur(imageUrl) {
        if (!imageUrl || typeof imageUrl !== 'string') return false;
        
        const normalizedUrl = this.normalizeImageUrl(imageUrl);
        
        // Check if URL is in our map
        const isBlurred = this.blurredImages.has(normalizedUrl);
        
        if (isBlurred) {
            logWithEmoji('info', 'blurTracker.shouldBlur', 'Found blurred image', 
                { url: normalizedUrl.substring(0, 50) + '...' });
            
            // Update timestamp to keep it fresh (automatic extension of expiration)
            this.blurredImages.set(normalizedUrl, Date.now());
            this.needsSave = true;
        }
        
        return isBlurred;
    },
    
    /**
     * Checks if an image was previously blurred but expired
     * 
     * Determines if an image URL is in the historical blur list, which
     * means it was once blurred but has since expired from the active list.
     * 
     * @param {string} imageUrl - The URL of the image to check
     * @returns {boolean} - True if the image was previously blurred, false otherwise
     */
    wasBlurredBefore(imageUrl) {
        if (!imageUrl || typeof imageUrl !== 'string') return false;
        
        const normalizedUrl = this.normalizeImageUrl(imageUrl);
        
        // Check historical records
        return this.historicalBlurs.has(normalizedUrl);
    },
    
    /**
     * Removes an image URL from the active blur list
     * 
     * Moves the image URL from the active blur list to the historical list,
     * recording when it was last blurred and when it expired.
     * 
     * @param {string} imageUrl - The URL of the image to unmark for blurring
     * @returns {void}
     */
    unmarkForBlur(imageUrl) {
        if (!imageUrl) return;
        
        const normalizedUrl = this.normalizeImageUrl(imageUrl);
        if (this.blurredImages.has(normalizedUrl)) {
            // Get the current timestamp for historical record
            const timestamp = this.blurredImages.get(normalizedUrl);
            
            // Remove from active blurs
            this.blurredImages.delete(normalizedUrl);
            
            // Add to historical blurs with expiration info
            this.historicalBlurs.set(normalizedUrl, {
                lastBlurredAt: timestamp,
                expiresAt: Date.now()
            });
            
            this.needsSave = true;
            logWithEmoji('image', 'blurTracker.unmarkForBlur', 'Unmarked image from blur', 
                { url: normalizedUrl.substring(0, 50) + '...' });
        }
    },
    
    //--------------------------------------------------------------------------
    // STORAGE AND MAINTENANCE METHODS
    //--------------------------------------------------------------------------
    
    /**
     * Saves the current blur data to Chrome storage
     * 
     * Saves both active and historical blur data to chrome.storage.local.
     * Before saving, cleans up old entries and ensures storage limits are respected.
     * 
     * @param {boolean} force - If true, saves even if needsSave flag is false
     * @returns {void}
     */
    save(force = false) {
        // Only save if needed or forced
        if (!this.needsSave && !force) return;
        
        // First clean up old entries
        this.cleanup();
        
        // Convert Maps to arrays for storage
        const currentBlurs = Array.from(this.blurredImages.entries());
        const historicalBlurs = Array.from(this.historicalBlurs.entries());
        
        // Ensure we don't exceed storage limits by keeping most recent items
        const MAX_ENTRIES = 10000;
        if (currentBlurs.length > MAX_ENTRIES) {
            // Sort by timestamp (newest first) and take top 10000
            currentBlurs.sort((a, b) => b[1] - a[1]);
            currentBlurs.splice(MAX_ENTRIES);
        }
        
        // Store the data
        chrome.storage.local.set({
            'blurredImageData': currentBlurs,
            'historicalBlurData': historicalBlurs
        }, () => {
            if (chrome.runtime.lastError) {
                logWithEmoji('error', 'blurTracker.save', 'Error saving blur data', chrome.runtime.lastError);
            } else {
                logWithEmoji('success', 'blurTracker.save', 'Saved blurred image data', 
                    { current: currentBlurs.length, historical: historicalBlurs.length });
                this.needsSave = false;
            }
        });
    },
    
    /**
     * Cleans up old entries from the active and historical blur lists
     * 
     * Moves entries older than MIN_AGE_MS from active to historical list.
     * Removes entries from historical list that are older than HISTORY_MAX_AGE.
     * 
     * @returns {void}
     */
    cleanup() {
        const now = Date.now();
        let removedCount = 0;
        let historicalCleanupCount = 0;
        
        // Move entries older than MIN_AGE_MS from active to historical
        for (const [url, timestamp] of this.blurredImages.entries()) {
            if (now - timestamp > this.MIN_AGE_MS) {
                // Remove from active list
                this.blurredImages.delete(url);
                
                // Add to historical record with expiration timestamp
                this.historicalBlurs.set(url, {
                    lastBlurredAt: timestamp,
                    expiresAt: now
                });
                
                removedCount++;
            }
        }
        
        // Clean very old historical entries
        for (const [url, data] of this.historicalBlurs.entries()) {
            if (now - data.expiresAt > this.HISTORY_MAX_AGE) {
                this.historicalBlurs.delete(url);
                historicalCleanupCount++;
            }
        }
        
        if (removedCount > 0 || historicalCleanupCount > 0) {
            logWithEmoji('info', 'blurTracker.cleanup', 'Cleaned up blur entries', 
                { movedToHistorical: removedCount, removedHistorical: historicalCleanupCount });
            this.needsSave = true;
        }
    },
    
    /**
     * Clears all blurred images from both active and historical lists
     * 
     * @returns {number} - The number of active blur entries that were cleared
     */
    clear() {
        const count = this.blurredImages.size;
        
        // Clear both current and historical
        this.blurredImages.clear();
        this.historicalBlurs.clear();
        
        this.needsSave = true;
        this.save(true); // Force immediate save
        logWithEmoji('success', 'blurTracker.clear', `Cleared ${count} blurred images and all historical records`, 
            { count, action: 'clear all' });
        return count;
    },
    
    /**
     * Returns the count of active blurred images
     * 
     * @returns {number} - The number of active blur entries
     */
    count() {
        return this.blurredImages.size;
    },
    
    /**
     * Returns the count of historical blurred images
     * 
     * @returns {number} - The number of historical blur entries
     */
    historicalCount() {
        return this.historicalBlurs.size;
    }
};

// Initialize blurTracker when the script loads
blurTracker.init(); 