/**
 * @fileoverview Manages a pool of web workers for parallel image processing.
 * Provides task queuing and worker lifecycle management.
 * @author Liron Farzam
 * @version 1.0.0
 */

/**
 * Priority Queue for task management
 */
class PriorityQueue {
    constructor() {
        this.items = [];
        this.processing = new Set();
    }

    add(task, priority = 0) {
        this.items.push({ task, priority, timestamp: Date.now() });
        this.items.sort((a, b) => b.priority - a.priority);
    }

    next() {
        return this.items.shift();
    }

    cleanup() {
        const now = Date.now();
        this.items = this.items.filter(item => 
            now - item.timestamp < this.options.taskTimeout
        );
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
            ...options
        };

        this.workers = new Map(); // Worker instances
        this.taskQueue = new PriorityQueue();
        this.idleWorkers = new Set();
        this.stats = {
            processed: 0,
            errors: 0,
            avgProcessingTime: 0
        };
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
            }

            // Warm up workers
            await this.warmup();
            
            // Start task processor
            this.startTaskProcessor();
            
            return true;
        } catch (error) {
            console.error('Worker pool initialization failed:', error);
            return false;
        }
    }

    /**
     * Create and initialize a worker
     */
    async createWorker(id) {
        const worker = new Worker(chrome.runtime.getURL('js/imageWorker.js'));
        
        // Set up error handling
        worker.onerror = this.handleWorkerError.bind(this);
        
        // Initialize worker
        await this.initializeWorker(worker, id);
        
        return worker;
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
            this.taskQueue.add({
                task,
                resolve,
                reject,
                priority
            });

            this.processNextTask();
        });
    }

    /**
     * Process next task in queue
     */
    async processNextTask() {
        if (this.idleWorkers.size === 0 || this.taskQueue.items.length === 0) {
            return;
        }

        const worker = this.idleWorkers.values().next().value;
        const task = this.taskQueue.next();

        if (!task) return;

        this.idleWorkers.delete(worker);
        
        try {
            const result = await this.executeTask(worker, task);
            task.resolve(result);
            
            // Update stats
            this.updateStats(task);
            
        } catch (error) {
            if (task.attempts < this.options.retryAttempts) {
                task.attempts++;
                this.taskQueue.add(task, task.priority + 1);
            } else {
                task.reject(error);
                this.stats.errors++;
            }
        } finally {
            this.idleWorkers.add(worker);
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

                worker.onmessage = (e) => {
                    clearTimeout(timeoutId);
                    if (e.data.success) {
                        resolve(e.data.result);
                    } else {
                        reject(new Error(e.data.error));
                    }
                };

                worker.postMessage({
                    type: 'PROCESS_IMAGE',
                    data: task.imageData
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
        const dummyData = new ImageData(1, 1);
        const warmupTasks = Array(this.options.maxWorkers).fill(dummyData)
            .map(data => this.processImage(data, -1));
        
        await Promise.all(warmupTasks);
    }

    /**
     * Update processing statistics
     */
    updateStats(task) {
        this.stats.processed++;
        const processingTime = Date.now() - task.startTime;
        this.stats.avgProcessingTime = 
            (this.stats.avgProcessingTime * (this.stats.processed - 1) + processingTime) 
            / this.stats.processed;
    }

    /**
     * Handle worker errors
     */
    handleWorkerError(error) {
        console.error('Worker error:', error);
        this.stats.errors++;
    }

    /**
     * Clean up resources
     */
    terminate() {
        this.workers.forEach(worker => worker.terminate());
        this.workers.clear();
        this.idleWorkers.clear();
        this.taskQueue = new PriorityQueue();
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