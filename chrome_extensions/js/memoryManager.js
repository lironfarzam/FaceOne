console.log('#### STARTING MEMORY_MANAGER.JS ####');


/**
 * @fileoverview Advanced memory management for TensorFlow.js operations
 * Provides tensor tracking, memory monitoring, and cleanup utilities
 * @author Liron Farzam
 * @version 1.0.0
 */

//=============================================================================
// TensorFlow.js Memory Management
//=============================================================================

/**
 * Enhanced tensor tracker that monitors all tensor allocations and helps identify leaks
 */
class TensorTracker {
  constructor() {
    this.activeReferences = new Map();
    this.stats = {
      created: 0,
      disposed: 0,
      leaked: 0,
      maxActive: 0
    };
    this.debugMode = false;
  }

  /**
   * Track a tensor and its creation context
   * @param {tf.Tensor} tensor - The tensor to track
   * @param {string} context - Description of where/why this tensor was created
   * @returns {tf.Tensor} The original tensor (for chaining)
   */
  track(tensor, context = 'unknown') {
    if (!tensor || !tensor.id) return tensor;
    
    this.stats.created++;
    const currentActive = this.activeReferences.size;
    this.stats.maxActive = Math.max(this.stats.maxActive, currentActive + 1);
    
    this.activeReferences.set(tensor.id, {
      tensor,
      context,
      shape: tensor.shape,
      dtype: tensor.dtype,
      createdAt: Date.now(),
      stack: this.debugMode ? new Error().stack : null
    });
    
    return tensor;
  }

  /**
   * Explicitly dispose a tracked tensor
   * @param {tf.Tensor} tensor - The tensor to dispose
   */
  dispose(tensor) {
    if (!tensor || !tensor.id) return;
    
    if (tensor.dispose && !tensor.isDisposed) {
      tensor.dispose();
    }
    
    if (this.activeReferences.has(tensor.id)) {
      this.activeReferences.delete(tensor.id);
      this.stats.disposed++;
    }
  }

  /**
   * Dispose all tracked tensors
   */
  disposeAll() {
    tf.tidy(() => {
      const tensors = Array.from(this.activeReferences.values());
      tensors.forEach(ref => {
        try {
          if (ref.tensor && !ref.tensor.isDisposed && ref.tensor.dispose) {
            ref.tensor.dispose();
            this.stats.disposed++;
          }
        } catch (e) {
          console.warn(`Failed to dispose tensor from ${ref.context}:`, e);
        }
      });
      this.activeReferences.clear();
    });
  }

  /**
   * Get information about potential tensor leaks
   * @param {number} olderThanMs - Only report tensors older than this many milliseconds
   * @returns {Array} Array of leaked tensor information
   */
  getLeaks(olderThanMs = 60000) {
    const now = Date.now();
    const leaks = [];
    
    this.activeReferences.forEach((ref, id) => {
      if (now - ref.createdAt > olderThanMs) {
        leaks.push({
          id,
          context: ref.context,
          shape: ref.shape,
          dtype: ref.dtype,
          age: Math.round((now - ref.createdAt) / 1000) + 's',
          stack: ref.stack
        });
      }
    });
    
    this.stats.leaked = leaks.length;
    return leaks;
  }

  /**
   * Enable or disable detailed stack traces (impacts performance)
   * @param {boolean} enabled - Whether to enable debug mode
   */
  setDebugMode(enabled) {
    this.debugMode = enabled;
  }

  /**
   * Get current memory statistics
   * @returns {Object} Memory statistics
   */
  getStats() {
    return {
      ...this.stats,
      currentActive: this.activeReferences.size,
      tfMemory: tf.memory()
    };
  }

  /**
   * Reset statistics
   */
  resetStats() {
    this.stats = {
      created: 0,
      disposed: 0,
      leaked: 0,
      maxActive: 0
    };
  }
}

/**
 * Memory-aware task scheduler that pauses processing when memory pressure is high
 */
class MemoryAwareScheduler {
  constructor(options = {}) {
    // Detect if we're in a content script (tf will be undefined)
    const isContentScript = typeof tf === 'undefined';
    
    this.options = {
      maxTasks: 10,
      maxTensors: 1000,
      maxBytes: 200 * 1024 * 1024, // 200MB
      checkIntervalMs: 5000,
      pauseThreshold: 0.9, // 90% of max
      resumeThreshold: 0.7, // 70% of max
      useAlternativeMemoryCheck: isContentScript,
      ...options
    };
    
    this.taskQueue = [];
    this.activePromises = new Set();
    this.isPaused = false;
    this.isDestroyed = false;
    this.checkInterval = null;
    this.stats = {
      enqueued: 0,
      processed: 0,
      dropped: 0,
      pauses: 0
    };
    
    this.startMonitoring();
  }

  /**
   * Start periodic memory monitoring
   */
  startMonitoring() {
    this.checkInterval = setInterval(() => {
      this.checkMemoryUsage();
    }, this.options.checkIntervalMs);
  }

  /**
   * Check current memory usage and pause/resume processing as needed
   */
  checkMemoryUsage() {
    try {
      // Check if tf is defined in this context
      if (typeof tf === 'undefined') {
        // If in content script context, use a different approach
        if (this.options.useAlternativeMemoryCheck) {
          // Use performance memory API if available
          if (window.performance && window.performance.memory) {
            const memInfo = window.performance.memory;
            const memoryUsage = memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit;
            
            if (!this.isPaused && memoryUsage > this.options.pauseThreshold) {
              this.pause();
              this.stats.pauses++;
              console.warn('Memory pressure detected, pausing processing', {
                memoryUsage: Math.round(memoryUsage * 100) + '%',
                usedHeap: (memInfo.usedJSHeapSize / (1024 * 1024)).toFixed(2) + 'MB',
                totalHeap: (memInfo.jsHeapSizeLimit / (1024 * 1024)).toFixed(2) + 'MB'
              });
            } else if (this.isPaused && memoryUsage < this.options.resumeThreshold) {
              this.resume();
              console.log('Memory pressure relieved, resuming processing');
            }
          }
        }
        return;
      }
      
      const memInfo = tf.memory();
      const tensorUsage = memInfo.numTensors / this.options.maxTensors;
      const memoryUsage = memInfo.numBytes / this.options.maxBytes;
      const usage = Math.max(tensorUsage, memoryUsage);
      
      if (!this.isPaused && usage > this.options.pauseThreshold) {
        this.pause();
        this.stats.pauses++;
        
        // Force garbage collection and tensor cleanup
        if (window.tensorTracker) {
          window.tensorTracker.disposeAll();
        }
        
        try {
          tf.engine().endScope();
          tf.engine().startScope();
        } catch (e) {
          console.warn('Error during forced cleanup:', e);
        }
        
        console.warn('Memory pressure detected, pausing processing', {
          tensorUsage: Math.round(tensorUsage * 100) + '%',
          memoryUsage: Math.round(memoryUsage * 100) + '%',
          numTensors: memInfo.numTensors,
          numBytes: (memInfo.numBytes / (1024 * 1024)).toFixed(2) + 'MB'
        });
      } else if (this.isPaused && usage < this.options.resumeThreshold) {
        this.resume();
        console.log('Memory pressure relieved, resuming processing');
      }
    } catch (e) {
      console.warn('Error checking memory usage:', e);
    }
  }

  /**
   * Pause task processing
   */
  pause() {
    this.isPaused = true;
  }

  /**
   * Resume task processing
   */
  resume() {
    this.isPaused = false;
  }

  /**
   * Add a task to the processing queue
   * @param {Function} task - The task function to execute
   * @param {number} priority - Priority level (higher numbers = higher priority)
   * @returns {Promise} Promise that resolves with the task result
   */
  enqueue(task, priority = 0) {
    this.stats.enqueued++;
    
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ 
        task, 
        priority, 
        resolve, 
        reject,
        enqueuedAt: Date.now()
      });
      
      // Sort by priority (higher first), then by enqueue time (older first)
      this.taskQueue.sort((a, b) => {
        if (a.priority !== b.priority) {
          return b.priority - a.priority;
        }
        return a.enqueuedAt - b.enqueuedAt;
      });
      
      if (!this.isPaused) {
        this.processNext();
      }
    });
  }

  /**
   * Process the next task in the queue
   */
  async processNext() {
    if (this.isPaused || this.taskQueue.length === 0) {
      return;
    }
    
    const { task, resolve, reject } = this.taskQueue.shift();
    
    try {
      // Execute task in a tidy environment
      const result = await tf.tidy(() => task());
      this.stats.processed++;
      resolve(result);
    } catch (error) {
      this.stats.dropped++;
      reject(error);
    } finally {
      // Process next task
      this.processNext();
    }
  }

  /**
   * Get current scheduler statistics
   * @returns {Object} Scheduler statistics
   */
  getStats() {
    return {
      ...this.stats,
      queueLength: this.taskQueue.length,
      isPaused: this.isPaused
    };
  }

  /**
   * Clean up resources
   */
  destroy() {
    clearInterval(this.checkInterval);
    this.taskQueue = [];
    this.isDestroyed = true;
  }
}

/**
 * Enhanced image cache with size-aware management
 */
class SmartImageCache {
  constructor(options = {}) {
    this.options = {
      maxEntries: 1000,
      maxMemoryUsage: 100 * 1024 * 1024, // 100MB
      entryTTL: 5 * 60 * 1000, // 5 minutes
      cleanupInterval: 60000, // 1 minute
      ...options
    };
    
    this.cache = new Map();
    this.memoryUsage = 0;
    this.stats = {
      hits: 0,
      misses: 0,
      evictions: 0,
      expirations: 0
    };
    
    // Start cleanup interval
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, this.options.cleanupInterval);
  }

  /**
   * Estimate the memory size of image data
   * @param {ImageData|Object} imageData - The image data to estimate
   * @returns {number} Estimated size in bytes
   */
  estimateSize(imageData) {
    if (!imageData) return 0;
    
    // If it's ImageData with a data property (Uint8ClampedArray)
    if (imageData.data && imageData.data.length) {
      return imageData.data.length;
    }
    
    // If it has width and height
    if (imageData.width && imageData.height) {
      // Estimate 4 bytes per pixel (RGBA)
      return imageData.width * imageData.height * 4;
    }
    
    // If it's an array or typed array
    if (imageData.length) {
      return imageData.length;
    }
    
    return 0;
  }

  /**
   * Add or update an entry in the cache
   * @param {string} key - Cache key
   * @param {*} data - Data to cache
   * @param {Object} metadata - Additional metadata
   */
  set(key, data, metadata = {}) {
    // Remove if already exists
    if (this.cache.has(key)) {
      this.remove(key);
    }
    
    // Estimate size
    const size = this.estimateSize(data);
    
    // Check if we have space
    if (this.memoryUsage + size > this.options.maxMemoryUsage) {
      this.makeRoom(size);
    }
    
    // Add to cache
    this.cache.set(key, {
      data,
      metadata,
      size,
      timestamp: Date.now(),
      lastAccessed: Date.now(),
      accessCount: 0
    });
    
    this.memoryUsage += size;
  }

  /**
   * Retrieve an entry from the cache
   * @param {string} key - Cache key
   * @returns {*} Cached data or null if not found
   */
  get(key) {
    const entry = this.cache.get(key);
    if (!entry) {
      this.stats.misses++;
      return null;
    }
    
    // Update access stats
    entry.lastAccessed = Date.now();
    entry.accessCount++;
    this.stats.hits++;
    
    return entry.data;
  }

  /**
   * Remove an entry from the cache
   * @param {string} key - Cache key
   */
  remove(key) {
    const entry = this.cache.get(key);
    if (!entry) return;
    
    this.memoryUsage -= entry.size;
    this.cache.delete(key);
  }

  /**
   * Make room for a new entry by removing less important entries
   * @param {number} requiredSize - Size needed in bytes
   */
  makeRoom(requiredSize) {
    // If cache is empty, nothing to do
    if (this.cache.size === 0) return;
    
    // If required size is larger than max, we can't cache it
    if (requiredSize > this.options.maxMemoryUsage) {
      console.warn('Requested cache entry exceeds maximum cache size');
      return;
    }
    
    // Sort entries by priority (last accessed, then access count)
    const entries = Array.from(this.cache.entries())
      .map(([key, entry]) => ({ key, entry }))
      .sort((a, b) => {
        // First sort by last accessed (oldest first)
        const timeDiff = a.entry.lastAccessed - b.entry.lastAccessed;
        if (Math.abs(timeDiff) > 60000) { // If more than 1 minute difference
          return timeDiff;
        }
        // Then by access count (least accessed first)
        return a.entry.accessCount - b.entry.accessCount;
      });
    
    // Remove entries until we have enough space
    let removedSize = 0;
    for (const { key, entry } of entries) {
      this.remove(key);
      removedSize += entry.size;
      this.stats.evictions++;
      
      if (this.memoryUsage + requiredSize <= this.options.maxMemoryUsage) {
        break;
      }
    }
  }

  /**
   * Perform periodic cleanup of expired entries
   */
  cleanup() {
    const now = Date.now();
    const expiredKeys = [];
    
    // Find expired entries
    this.cache.forEach((entry, key) => {
      if (now - entry.timestamp > this.options.entryTTL) {
        expiredKeys.push(key);
      }
    });
    
    // Remove expired entries
    expiredKeys.forEach(key => {
      this.remove(key);
      this.stats.expirations++;
    });
    
    // If still too many entries, remove oldest
    if (this.cache.size > this.options.maxEntries) {
      const entries = Array.from(this.cache.entries())
        .map(([key, entry]) => ({ key, timestamp: entry.timestamp }))
        .sort((a, b) => a.timestamp - b.timestamp);
      
      const toRemove = entries.slice(0, this.cache.size - this.options.maxEntries);
      toRemove.forEach(({ key }) => {
        this.remove(key);
        this.stats.evictions++;
      });
    }
  }

  /**
   * Get cache statistics
   * @returns {Object} Cache statistics
   */
  getStats() {
    return {
      ...this.stats,
      size: this.cache.size,
      memoryUsage: this.memoryUsage,
      memoryUsageMB: (this.memoryUsage / (1024 * 1024)).toFixed(2) + 'MB'
    };
  }

  /**
   * Clean up resources
   */
  destroy() {
    clearInterval(this.cleanupInterval);
    this.cache.clear();
    this.memoryUsage = 0;
  }
}

// Export the classes for use in other modules
if (typeof window !== 'undefined') {
  // Browser environment
  window.TensorTracker = TensorTracker;
  window.MemoryAwareScheduler = MemoryAwareScheduler;
  window.SmartImageCache = SmartImageCache;
  
  // Create global instances for convenience
  window.tensorTracker = new TensorTracker();
  window.memoryScheduler = new MemoryAwareScheduler();
  window.imageCache = new SmartImageCache();
} else if (typeof self !== 'undefined') {
  // Web Worker environment
  self.TensorTracker = TensorTracker;
  self.MemoryAwareScheduler = MemoryAwareScheduler;
  self.SmartImageCache = SmartImageCache;
}

// CommonJS export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    TensorTracker,
    MemoryAwareScheduler,
    SmartImageCache
  };
}

// Initialize global instances
try {
  // Initialize in the current context if possible
  if (typeof window !== 'undefined') {
    // Only initialize if not already initialized
    if (!window.tensorTracker) {
      window.tensorTracker = new TensorTracker();
      console.log('Global TensorTracker initialized');
    }
    
    if (!window.memoryScheduler) {
      window.memoryScheduler = new MemoryAwareScheduler();
      console.log('Global MemoryAwareScheduler initialized');
    }
    
    if (!window.imageCache) {
      window.imageCache = new SmartImageCache();
      console.log('Global SmartImageCache initialized');
    }
  }
} catch (e) {
  console.warn('Error initializing memory management components:', e);
} 