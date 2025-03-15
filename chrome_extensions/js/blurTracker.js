/**
 * blurTracker.js - Lightweight URL-based image blurring system
 * Only stores URLs of images that need to be blurred, without keeping embeddings
 */

// Lightweight blurred image tracker
const blurTracker = {
    // Store URLs with timestamps
    blurredImages: new Map(),
    
    // Store historical records of previously blurred images
    historicalBlurs: new Map(),
    
    // Minimum age to keep entries (30 minutes in milliseconds)
    MIN_AGE_MS: 30 * 60 * 1000,
    
    // History retention (7 days in milliseconds)
    HISTORY_MAX_AGE: 7 * 24 * 60 * 60 * 1000,
    
    // Flag to track if we need to save
    needsSave: false,
    
    // Initialize from storage
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
                    this.historicalBlurs = new Map();
                }
            }
        });
        
        // Set up periodic save timer
        this.startSaveTimer();
        
        // Set up MutationObserver to catch newly added images
        this.setupMutationObserver();
    },
    
    // Start a timer to periodically save data
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
    
    // Set up mutation observer to catch newly added images (especially after scrolling)
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
                childList: true,
                subtree: true,
                attributes: false
            });
            
            logWithEmoji('setup', 'blurTracker.setupMutationObserver', 'MutationObserver setup complete');
        }
    },
    
    // Process nodes added to the DOM, looking for images that need blurring
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
    
    // Apply blur to image if it's in our list
    applyBlurIfNeeded(img) {
        if (!img || !img.src) return;
        
        const normalizedUrl = this.normalizeImageUrl(img.src);
        if (this.shouldBlur(normalizedUrl)) {
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
    
    // Normalize URL to handle Facebook's changing parameters
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
    
    // Mark an image for blurring
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
    
    // Check if an image should be blurred
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
    
    // Check if an image was previously blurred but expired
    wasBlurredBefore(imageUrl) {
        if (!imageUrl || typeof imageUrl !== 'string') return false;
        
        const normalizedUrl = this.normalizeImageUrl(imageUrl);
        
        // Check historical records
        return this.historicalBlurs.has(normalizedUrl);
    },
    
    // Remove from blur list if needed
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
    
    // Save the current set of URLs to storage
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
    
    // Clean up old entries (older than MIN_AGE_MS)
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
    
    // Clear all blurred images
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
    
    // Get the number of blurred images
    count() {
        return this.blurredImages.size;
    },
    
    // Get the number of historical records
    historicalCount() {
        return this.historicalBlurs.size;
    }
};

// Initialize on load
blurTracker.init(); 