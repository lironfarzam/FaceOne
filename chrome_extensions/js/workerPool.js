/**
 * @fileoverview Manages a pool of web workers for parallel image processing.
 * Provides task queuing and worker lifecycle management.
 * @author Liron Farzam
 * @version 1.0.0
 * 
 * This module implements a sophisticated worker pool for distributing 
 * image processing tasks across multiple Web Worker threads. It provides
 * task prioritization, automatic worker lifecycle management, error handling,
 * and performance optimization to maximize throughput while maintaining
 * stability and resource efficiency.
 */

//==============================================================================
// PRIORITY QUEUE FOR TASK MANAGEMENT
//==============================================================================

/**
 * A priority queue implementation for managing tasks with timeouts and statistics
 * 
 * Handles prioritized enqueuing, efficient processing order, automatic cleanup
 * of stale tasks, and comprehensive metrics for monitoring queue performance.
 * 
 * @class
 */
class PriorityQueue {
    /**
     * Creates a new priority queue instance
     * 
     * @param {Object} options - Configuration options
     * @param {number} [options.taskTimeout=30000] - Maximum time in milliseconds before a task is considered stale
     */
    constructor(options = {}) {
        /**
         * Array of queued task items awaiting processing
         * @type {Array<Object>}
         * @private
         */
        this.items = [];
        
        /**
         * Set of items currently being processed
         * @type {Set<Object>}
         * @private
         */
        this.processing = new Set();
        
        /**
         * Configuration options with defaults
         * @type {Object}
         * @private
         */
        this.options = {
            taskTimeout: 30000, // 30 seconds
            ...options
        };
        
        /**
         * Statistics tracking for performance monitoring
         * @type {Object}
         * @private
         */
        this.stats = {
            enqueued: 0,    // Total tasks added to queue
            processed: 0,   // Successfully processed tasks
            timedOut: 0,    // Tasks that exceeded timeout
            errors: 0       // Tasks that failed with errors
        };
    }

    /**
     * Adds a task to the queue with specified priority
     * 
     * Higher priority tasks are processed before lower priority ones.
     * Each task is assigned a unique ID for tracking and potential cancellation.
     * 
     * @param {Object} task - The task to be processed
     * @param {number} [priority=0] - Task priority (higher values = higher priority)
     * @returns {string} Unique task ID for tracking or cancellation
     */
    add(task, priority = 0) {
        const item = { 
            task, 
            priority, 
            timestamp: Date.now(),
            id: `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        };
        
        this.items.push(item);
        this.stats.enqueued++;
        this.items.sort((a, b) => b.priority - a.priority);
        
        return item.id; // Return task ID for potential cancellation
    }

    /**
     * Retrieves the next highest-priority task from the queue
     * 
     * Removes the task from the queue and adds it to the processing set.
     * Returns null if the queue is empty.
     * 
     * @returns {Object|null} The next task item or null if queue is empty
     */
    next() {
        if (this.items.length === 0) return null;
        
        const item = this.items.shift();
        this.processing.add(item);
        return item;
    }
    
    /**
     * Marks a task as completed and removes it from the processing set
     * 
     * Updates task statistics for completed tasks.
     * 
     * @param {string} taskId - The ID of the task to mark as complete
     * @param {boolean} [success=true] - Whether the task completed successfully
     * @returns {boolean} True if the task was found and marked, false otherwise
     */
    markComplete(taskId, success = true) {
        // Find in processing set
        for (const item of this.processing) {
            if (item.id === taskId) {
                this.processing.delete(item);
                this.stats.processed++;
                return true;
            }
        }
        return false;
    }
    
    /**
     * Marks a task as failed with an error and removes it from the processing set
     * 
     * Updates error statistics for tracking failed tasks.
     * 
     * @param {string} taskId - The ID of the task that failed
     * @returns {boolean} True if the task was found and marked, false otherwise
     */
    markError(taskId) {
        // Find in processing set
        for (const item of this.processing) {
            if (item.id === taskId) {
                this.processing.delete(item);
                this.stats.errors++;
                return true;
            }
        }
        return false;
    }

    /**
     * Cleans up stale tasks that have exceeded their timeout period
     * 
     * Removes timed-out tasks from both the queue and processing sets.
     * Updates timeout statistics for monitoring.
     * 
     * @returns {number} Total number of stale tasks removed
     */
    cleanup() {
        const now = Date.now();
        
        // Clean queue items
        const initialQueueLength = this.items.length;
        this.items = this.items.filter(item => 
            now - item.timestamp < this.options.taskTimeout
        );
        
        const queueRemoved = initialQueueLength - this.items.length;
        
        // Clean processing items
        const processingArray = Array.from(this.processing);
        const initialProcessingSize = processingArray.length;
        
        for (const item of processingArray) {
            if (now - item.timestamp >= this.options.taskTimeout) {
                this.processing.delete(item);
                this.stats.timedOut++;
            }
        }
        
        const processingRemoved = initialProcessingSize - this.processing.size;
        const totalRemoved = queueRemoved + processingRemoved;
        
        if (totalRemoved > 0) {
            logWithEmoji('warning', 'PriorityQueue.cleanup', `Removed ${totalRemoved} stale tasks`, 
                { queuedRemoved: queueRemoved, processingRemoved: processingRemoved });
        }
        
        return totalRemoved;
    }
    
    /**
     * Returns the current number of tasks waiting in the queue
     * 
     * @returns {number} Number of tasks in the queue
     */
    size() {
        return this.items.length;
    }
    
    /**
     * Returns the current number of tasks being processed
     * 
     * @returns {number} Number of tasks currently being processed
     */
    processingCount() {
        return this.processing.size;
    }
    
    /**
     * Returns statistics about the queue's performance
     * 
     * @returns {Object} Statistics about enqueued, processed, timed out, and errored tasks
     */
    getStats() {
        return {...this.stats};
    }
    
    /**
     * Resets the queue, clearing all queued and processing tasks
     * 
     * @returns {void}
     */
    reset() {
        this.items = [];
        this.processing.clear();
    }
}

//==============================================================================
// WORKER POOL IMPLEMENTATION
//==============================================================================

/**
 * Enhanced worker pool with advanced resource management and error handling
 * 
 * Manages a pool of Web Workers for parallelized image processing with features like:
 * - Task priority queuing
 * - Worker health monitoring
 * - Automatic worker recycling
 * - Memory usage optimization
 * - Performance statistics
 * - Error handling and recovery
 * 
 * @class
 */
class EnhancedWorkerPool {
    /**
     * Creates a new worker pool
     * 
     * @param {Object} options - Configuration options
     * @param {number} [options.maxWorkers] - Maximum number of workers (defaults to hardware concurrency or 4)
     * @param {number} [options.taskTimeout=30000] - Task timeout in milliseconds
     * @param {number} [options.retryAttempts=2] - Number of retry attempts for failed tasks
     * @param {number} [options.batchSize=4] - Number of tasks to process in parallel
     * @param {number} [options.workerRestartThreshold=1000] - Number of tasks before recycling a worker
     * @param {number} [options.memoryCheckInterval=60000] - Interval for memory checks in milliseconds
     */
    constructor(options = {}) {
        this.options = {
            maxWorkers: navigator.hardwareConcurrency || 4,
            taskTimeout: 30000,
            retryAttempts: 2,
            batchSize: 4,
            workerRestartThreshold: 1000, // Restart worker after this many tasks
            memoryCheckInterval: 60000,   // Check memory usage every minute
            ...options
        };

        /**
         * Map of worker IDs to worker instances
         * @type {Map<number, Worker>}
         * @private
         */
        this.workers = new Map();
        
        /**
         * Queue for managing pending tasks
         * @type {PriorityQueue}
         * @private
         */
        this.taskQueue = new PriorityQueue({
            taskTimeout: this.options.taskTimeout
        });
        
        /**
         * Set of currently idle workers
         * @type {Set<Worker>}
         * @private
         */
        this.idleWorkers = new Set();
        
        /**
         * Performance and operational statistics
         * @type {Object}
         * @private
         */
        this.stats = {
            processed: 0,
            errors: 0,
            avgProcessingTime: 0,
            peakMemoryUsage: 0,
            workersCreated: 0,
            workersRestarted: 0
        };
        
        /**
         * Map of workers to their state information
         * @type {Map<Worker, Object>}
         * @private
         */
        this.workerState = new Map();
        
        /**
         * Interval ID for memory checking
         * @type {number|null}
         * @private
         */
        this.memoryCheckInterval = null;
    }

    /**
     * Initialize worker pool with warm-up
     * 
     * Creates all workers according to configuration, initializes them,
     * performs warm-up tasks to optimize initial performance, and
     * starts the task processor and memory monitoring systems.
     * 
     * @async
     * @returns {Promise<boolean>} True if initialization successful, false otherwise
     */
    async initialize() {
        try {
            logWithEmoji('setup', 'workerPool.initialize', 'Initializing worker pool', 
                { maxWorkers: this.options.maxWorkers });
                
            // Create workers
            for (let i = 0; i < this.options.maxWorkers; i++) {
                const worker = await this.createWorker(i);
                this.workers.set(i, worker);
                this.idleWorkers.add(worker);
                
                // Initialize worker state
                this.workerState.set(worker, {
                    id: i,
                    tasksProcessed: 0,
                    errors: 0,
                    lastTaskTime: 0,
                    createdAt: Date.now(),
                    memoryUsage: 0
                });
            }

            // Warm up workers
            await this.warmup();
            
            // Start task processor
            this.startTaskProcessor();
            
            // Start memory monitoring
            this.startMemoryMonitoring();
            
            logWithEmoji('success', 'workerPool.initialize', 'Worker pool initialized successfully', 
                { workers: this.workers.size });
                
            return true;
        } catch (error) {
            logWithEmoji('error', 'workerPool.initialize', 'Worker pool initialization failed', error);
            
            // Clean up any created workers on failure
            this.workers.forEach(worker => {
                try {
                    worker.terminate();
                } catch (e) {
                    // Ignore termination errors
                }
            });
            
            this.workers.clear();
            this.idleWorkers.clear();
            this.workerState.clear();
            
            return false;
        }
    }

    /**
     * Creates and initializes a new worker
     * 
     * Loads the worker script, sets up error handling, and initializes
     * the worker with appropriate configuration.
     * 
     * @async
     * @param {number} id - Worker identifier
     * @returns {Promise<Worker>} The initialized worker
     * @throws {Error} If worker creation or initialization fails
     */
    async createWorker(id) {
        try {
            logWithEmoji('setup', 'workerPool.createWorker', `Creating worker ${id}`);
            // Create blob URL for the worker to avoid CSP issues
            const workerScript = chrome.runtime.getURL('js/imageWorker.js');
            const worker = new Worker(workerScript);
            
            // Set up error handling
            worker.onerror = this.handleWorkerError.bind(this, worker);
            
            // Initialize worker
            await this.initializeWorker(worker, id);
            
            this.stats.workersCreated++;
            
            logWithEmoji('success', 'workerPool.createWorker', `Worker ${id} created successfully`);
            return worker;
        } catch (error) {
            logWithEmoji('error', 'workerPool.createWorker', `Failed to create worker ${id}`, error);
            throw error;
        }
    }
    
    /**
     * Initialize a worker with initial configuration
     */
    async initializeWorker(worker, id) {
        return new Promise((resolve, reject) => {
            const initTimeout = setTimeout(() => {
                reject(new Error(`Worker ${id} initialization timed out`));
            }, this.options.taskTimeout);
            
            // Setup one-time message handler for initialization
            const handleInitMessage = (e) => {
                if (e.data.type === 'WORKER_READY' && e.data.success) {
                    clearTimeout(initTimeout);
                    worker.removeEventListener('message', handleInitMessage);
                    resolve(worker);
                } else if (e.data.type === 'INIT_FAILED') {
                    clearTimeout(initTimeout);
                    worker.removeEventListener('message', handleInitMessage);
                    reject(new Error(e.data.error || 'Worker initialization failed'));
                }
            };
            
            worker.addEventListener('message', handleInitMessage);
            
            // Send initialization message
            worker.postMessage({
                type: 'INIT',
                workerId: id,
                config: {
                    maxImageSize: this.options.maxImageSize || 1024,
                    processingTimeout: this.options.taskTimeout,
                    debug: window.DEBUG // Pass debug flag to worker
                }
            });
        });
    }

    /**
     * Processes an image using the worker pool
     * 
     * Adds the image processing task to the queue with the specified priority.
     * The task will be picked up by the next available worker.
     * 
     * @async
     * @param {ImageData} imageData - The image data to process
     * @param {number} [priority=0] - Task priority (higher values = higher priority)
     * @returns {Promise<ImageData>} Processed image data
     * @throws {Error} If processing fails after retry attempts
     */
    async processImage(imageData, priority = 0) {
        const task = {
            imageData,
            attempts: 0,
            startTime: Date.now()
        };

        return new Promise((resolve, reject) => {
            const taskId = this.taskQueue.add({
                task,
                resolve,
                reject,
                priority
            }, priority);
            
            // Store task ID for potential cleanup or cancellation
            task.id = taskId;

            this.processNextTask();
        });
    }

    /**
     * Process next task in queue
     */
    async processNextTask() {
        if (this.idleWorkers.size === 0 || this.taskQueue.size() === 0) {
            return;
        }

        const worker = this.idleWorkers.values().next().value;
        const taskItem = this.taskQueue.next();

        if (!taskItem || !worker) return;

        this.idleWorkers.delete(worker);
        const { task, resolve, reject } = taskItem.task;
        
        try {
            const result = await this.executeTask(worker, task);
            this.taskQueue.markComplete(taskItem.id);
            resolve(result);
            
            // Update stats
            this.updateStats(task, worker);
            
            // Check if worker needs recycling
            await this.checkWorkerRecycling(worker);
            
        } catch (error) {
            this.taskQueue.markError(taskItem.id);
            
            if (task.attempts < this.options.retryAttempts) {
                task.attempts++;
                logWithEmoji('warning', 'workerPool.processNextTask', 
                    `Task failed, retrying (attempt ${task.attempts}/${this.options.retryAttempts})`, { error: error.message });
                this.taskQueue.add({
                    task,
                    resolve,
                    reject,
                    priority: taskItem.priority + 1 // Increase priority
                }, taskItem.priority + 1);
            } else {
                logWithEmoji('error', 'workerPool.processNextTask', 
                    `Task failed after ${task.attempts} attempts`, error);
                reject(error);
                this.stats.errors++;
                
                // Update worker error stats
                const state = this.workerState.get(worker);
                if (state) {
                    state.errors++;
                }
            }
        } finally {
            // Return worker to idle pool unless it was recycled
            if (this.workers.has(this.getWorkerId(worker))) {
                this.idleWorkers.add(worker);
            }
            
            // Process more tasks if available
            this.processNextTask();
        }
    }

    /**
     * Execute task with timeout and error handling
     */
    async executeTask(worker, task) {
        return Promise.race([
            new Promise((resolve, reject) => {
                const timeoutId = setTimeout(() => {
                    reject(new Error('Task timeout'));
                }, this.options.taskTimeout);

                const messageHandler = (e) => {
                    if (e.data.type === 'IMAGE_PROCESSED') {
                        clearTimeout(timeoutId);
                        worker.removeEventListener('message', messageHandler);
                        
                        if (e.data.success) {
                            resolve(e.data.data);
                        } else {
                            reject(new Error(e.data.error || 'Processing failed'));
                        }
                    }
                };

                worker.addEventListener('message', messageHandler);

                worker.postMessage({
                    type: 'PROCESS_IMAGE',
                    data: task.imageData,
                    width: task.imageData.width,
                    height: task.imageData.height
                });
            }),
            new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Task timeout')), 
                this.options.taskTimeout)
            )
        ]);
    }

    /**
     * Warm up workers with dummy tasks
     */
    async warmup() {
        try {
            logWithEmoji('setup', 'workerPool.warmup', 'Warming up worker pool');
            const dummyData = new ImageData(1, 1);
            const warmupPromises = [];
            
            // Warm up each worker with a dummy task
            for (const worker of this.workers.values()) {
                const warmupPromise = new Promise((resolve, reject) => {
                    const timeoutId = setTimeout(() => {
                        reject(new Error('Warmup timeout'));
                    }, this.options.taskTimeout);
                    
                    const messageHandler = (e) => {
                        if (e.data.type === 'IMAGE_PROCESSED') {
                            clearTimeout(timeoutId);
                            worker.removeEventListener('message', messageHandler);
                            resolve();
                        }
                    };
                    
                    worker.addEventListener('message', messageHandler);
                    
                    worker.postMessage({
                        type: 'PROCESS_IMAGE',
                        data: dummyData,
                        width: 1,
                        height: 1
                    });
                });
                
                warmupPromises.push(warmupPromise);
            }
            
            await Promise.all(warmupPromises);
            logWithEmoji('success', 'workerPool.warmup', 'Worker pool warmup completed successfully');
            return true;
        } catch (error) {
            logWithEmoji('error', 'workerPool.warmup', 'Worker pool warmup failed', error);
            return false;
        }
    }

    /**
     * Start the task processor at regular intervals
     */
    startTaskProcessor() {
        // Process tasks in the queue at regular intervals
        setInterval(() => {
            this.processNextTask();
        }, 50); // Check every 50ms
        
        // Clean up stale tasks
        setInterval(() => {
            this.taskQueue.cleanup();
        }, 10000); // Clean every 10 seconds
    }
    
    /**
     * Start memory monitoring for workers
     */
    startMemoryMonitoring() {
        if (this.memoryCheckInterval) {
            clearInterval(this.memoryCheckInterval);
        }
        
        this.memoryCheckInterval = setInterval(() => {
            this.checkWorkerMemoryUsage();
        }, this.options.memoryCheckInterval);
    }
    
    /**
     * Check memory usage of all workers
     */
    async checkWorkerMemoryUsage() {
        try {
            // Request memory stats from all workers
            for (const worker of this.workers.values()) {
                worker.postMessage({
                    type: 'GET_MEMORY_STATS'
                });
            }
        } catch (error) {
            logWithEmoji('error', 'workerPool.checkWorkerMemoryUsage', 'Error checking worker memory', error);
        }
    }
    
    /**
     * Check if worker needs to be recycled
     * @param {Worker} worker - The worker to check
     */
    async checkWorkerRecycling(worker) {
        const state = this.workerState.get(worker);
        if (!state) return;
        
        // Check if worker has processed too many tasks
        if (state.tasksProcessed >= this.options.workerRestartThreshold) {
            await this.recycleWorker(worker);
        }
        
        // Check if worker has too many errors
        if (state.errors >= 5) {
            await this.recycleWorker(worker);
        }
    }
    
    /**
     * Recycle a worker by replacing it with a new one
     * @param {Worker} worker - The worker to recycle
     */
    async recycleWorker(worker) {
        try {
            // Get worker ID
            const workerId = this.getWorkerId(worker);
            if (workerId === -1) return;
            
            // Get worker state for logging
            const state = this.workerState.get(worker);
            
            // Remove from idle set if present
            this.idleWorkers.delete(worker);
            
            // Terminate the old worker
            this.workers.delete(workerId);
            this.workerState.delete(worker);
            worker.terminate();
            
            // Create a new worker
            const newWorker = await this.createWorker(workerId);
            this.workers.set(workerId, newWorker);
            this.idleWorkers.add(newWorker);
            
            // Initialize new worker state
            this.workerState.set(newWorker, {
                id: workerId,
                tasksProcessed: 0,
                errors: 0,
                lastTaskTime: 0,
                createdAt: Date.now(),
                memoryUsage: 0
            });
            
            this.stats.workersRestarted++;
            
            logWithEmoji('info', 'workerPool.recycleWorker', `Recycled worker ${workerId}`, 
                { tasksProcessed: state?.tasksProcessed || 0, errors: state?.errors || 0 });
            
            return newWorker;
        } catch (error) {
            logWithEmoji('error', 'workerPool.recycleWorker', 'Error recycling worker', error);
            return null;
        }
    }
    
    /**
     * Get worker ID from Map
     * @param {Worker} worker - The worker to find
     * @returns {number} Worker ID or -1 if not found
     */
    getWorkerId(worker) {
        for (const [id, w] of this.workers.entries()) {
            if (w === worker) return id;
        }
        return -1;
    }

    /**
     * Update processing statistics
     */
    updateStats(task, worker) {
        this.stats.processed++;
        const processingTime = Date.now() - task.startTime;
        this.stats.avgProcessingTime = 
            (this.stats.avgProcessingTime * (this.stats.processed - 1) + processingTime) 
            / this.stats.processed;
            
        // Update worker-specific stats
        const state = this.workerState.get(worker);
        if (state) {
            state.tasksProcessed++;
            state.lastTaskTime = Date.now();
        }
    }

    /**
     * Handle worker errors
     */
    handleWorkerError(worker, error) {
        logWithEmoji('error', 'workerPool.handleWorkerError', 'Worker error', error);
        this.stats.errors++;
        
        // Update worker-specific stats
        const state = this.workerState.get(worker);
        if (state) {
            state.errors++;
        }
        
        // If a worker has too many errors, recycle it
        if (state && state.errors >= 5) {
            this.recycleWorker(worker);
        }
    }
    
    /**
     * Handle memory stat messages from workers
     */
    handleMemoryStats(worker, memoryStats) {
        const state = this.workerState.get(worker);
        if (state) {
            state.memoryUsage = memoryStats.memoryUsage;
            
            // Update peak memory usage
            if (memoryStats.memoryUsage > this.stats.peakMemoryUsage) {
                this.stats.peakMemoryUsage = memoryStats.memoryUsage;
            }
            
            // If memory usage is too high, consider recycling the worker
            if (memoryStats.memoryUsage > 100 * 1024 * 1024) { // 100MB
                logWithEmoji('warning', 'workerPool.handleMemoryStats', 
                    `Worker ${state.id} using excessive memory: ${Math.round(memoryStats.memoryUsage / (1024 * 1024))}MB`,
                    { workerId: state.id, memoryUsage: memoryStats.memoryUsage });
                this.recycleWorker(worker);
            }
        }
    }

    /**
     * Get detailed worker pool statistics
     */
    getDetailedStats() {
        const workerStats = Array.from(this.workerState.entries()).map(([worker, state]) => {
            return {
                id: state.id,
                tasksProcessed: state.tasksProcessed,
                errors: state.errors,
                uptime: Date.now() - state.createdAt,
                memoryUsage: state.memoryUsage,
                isIdle: this.idleWorkers.has(worker)
            };
        });
        
        return {
            ...this.stats,
            currentQueueSize: this.taskQueue.size(),
            currentProcessing: this.taskQueue.processingCount(),
            workerCount: this.workers.size,
            idleWorkerCount: this.idleWorkers.size,
            queueStats: this.taskQueue.getStats(),
            workers: workerStats
        };
    }

    /**
     * Clean up resources and terminate all workers
     * 
     * Properly shuts down the worker pool, terminating all workers,
     * clearing intervals, and resetting all internal state.
     * 
     * @returns {void}
     */
    terminate() {
        logWithEmoji('setup', 'workerPool.terminate', 'Terminating worker pool');
        
        // Clear intervals
        if (this.memoryCheckInterval) {
            clearInterval(this.memoryCheckInterval);
            this.memoryCheckInterval = null;
        }
        
        // Terminate all workers
        this.workers.forEach(worker => {
            try {
                worker.terminate();
            } catch (e) {
                // Ignore termination errors
            }
        });
        
        // Clear all collections
        this.workers.clear();
        this.idleWorkers.clear();
        this.workerState.clear();
        this.taskQueue.reset();
        
        logWithEmoji('success', 'workerPool.terminate', 'Worker pool terminated');
    }
}

//==============================================================================
// EXPORTED CLASSES
//==============================================================================

/**
 * Main WorkerPool class for use by the extension
 * 
 * This is an alias of EnhancedWorkerPool, provided for
 * backward compatibility and cleaner API.
 * 
 * @class
 * @extends EnhancedWorkerPool
 */
class WorkerPool extends EnhancedWorkerPool {
    /**
     * Creates a new worker pool instance
     * 
     * @param {Object} options - Configuration options (see EnhancedWorkerPool constructor)
     */
    constructor(options = {}) {
        super(options);
    }
}

// Export classes to global scope for use in extension
/**
 * Make WorkerPool class available globally
 * @type {typeof WorkerPool}
 */
window.WorkerPool = WorkerPool;

/**
 * Make EnhancedWorkerPool class available globally
 * @type {typeof EnhancedWorkerPool}
 */
window.EnhancedWorkerPool = EnhancedWorkerPool; 