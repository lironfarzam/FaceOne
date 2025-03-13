/**
 * @fileoverview Manages a pool of web workers for parallel image processing.
 * Provides task queuing and worker lifecycle management.
 * @author Liron Farzam
 * @version 1.0.0
 */

/**
 * Priority Queue for task management with improved cleanup
 */
class PriorityQueue {
    constructor(options = {}) {
        this.items = [];
        this.processing = new Set();
        this.options = {
            taskTimeout: 30000, // 30 seconds
            ...options
        };
        
        // Stats tracking
        this.stats = {
            enqueued: 0,
            processed: 0,
            timedOut: 0,
            errors: 0
        };
    }

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

    next() {
        if (this.items.length === 0) return null;
        
        const item = this.items.shift();
        this.processing.add(item);
        return item;
    }
    
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
            console.warn(`PriorityQueue: Removed ${totalRemoved} stale tasks (${queueRemoved} queued, ${processingRemoved} processing)`);
        }
        
        return totalRemoved;
    }
    
    size() {
        return this.items.length;
    }
    
    processingCount() {
        return this.processing.size;
    }
    
    getStats() {
        return {...this.stats};
    }
    
    reset() {
        this.items = [];
        this.processing.clear();
    }
}

/**
 * Enhanced Worker Pool with optimized resource management
 */
class EnhancedWorkerPool {
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

        this.workers = new Map(); // Worker instances
        this.taskQueue = new PriorityQueue({
            taskTimeout: this.options.taskTimeout
        });
        this.idleWorkers = new Set();
        this.stats = {
            processed: 0,
            errors: 0,
            avgProcessingTime: 0,
            peakMemoryUsage: 0,
            workersCreated: 0,
            workersRestarted: 0
        };
        
        // Worker state tracking
        this.workerState = new Map(); // Tracks per-worker stats
        
        // Memory check interval
        this.memoryCheckInterval = null;
    }

    /**
     * Initialize worker pool with warm-up
     */
    async initialize() {
        try {
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
            
            return true;
        } catch (error) {
            console.error('Worker pool initialization failed:', error);
            
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
     * Create and initialize a worker
     */
    async createWorker(id) {
        try {
            // Create blob URL for the worker to avoid CSP issues
            const workerScript = chrome.runtime.getURL('js/imageWorker.js');
            const worker = new Worker(workerScript);
            
            // Set up error handling
            worker.onerror = this.handleWorkerError.bind(this, worker);
            
            // Initialize worker
            await this.initializeWorker(worker, id);
            
            this.stats.workersCreated++;
            
            return worker;
        } catch (error) {
            console.error(`Failed to create worker ${id}:`, error);
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
                    processingTimeout: this.options.taskTimeout
                }
            });
        });
    }

    /**
     * Process image with automatic retry and fallback
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
                this.taskQueue.add({
                    task,
                    resolve,
                    reject,
                    priority: taskItem.priority + 1 // Increase priority
                }, taskItem.priority + 1);
            } else {
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
            console.log('Worker pool warmup completed successfully');
            return true;
        } catch (error) {
            console.error('Worker pool warmup failed:', error);
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
            console.error('Error checking worker memory:', error);
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
            
            console.log(`Recycled worker ${workerId} after ${state.tasksProcessed} tasks`);
            
            return newWorker;
        } catch (error) {
            console.error('Error recycling worker:', error);
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
        console.error('Worker error:', error);
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
                console.warn(`Worker ${state.id} using excessive memory: ${Math.round(memoryStats.memoryUsage / (1024 * 1024))}MB`);
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
     * Clean up resources
     */
    terminate() {
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
        
        console.log('Worker pool terminated');
    }
}

// Change the class name in the file
class WorkerPool extends EnhancedWorkerPool {
    constructor(options = {}) {
        super(options);
    }
}

// Export both classes
window.WorkerPool = WorkerPool;
window.EnhancedWorkerPool = EnhancedWorkerPool; 