/**
 * @fileoverview Manages a pool of web workers for parallel image processing.
 * Provides task queuing and worker lifecycle management.
 * @author Liron Farzam
 * @version 1.0.0
 */

//=============================================================================
// Worker Pool Class Definition
//=============================================================================
/**
 * Manages a pool of Web Workers for parallel image processing
 * @class
 */
class WorkerPool {
    /**
     * Creates a new WorkerPool instance
     * @param {number} size - Number of workers to create (defaults to CPU core count)
     */
    constructor(size = navigator.hardwareConcurrency || 4) {
        /** @type {number} Number of workers in the pool */
        this.size = size;
        /** @type {Worker[]} Array of worker instances */
        this.workers = [];
        /** @type {Array<Object>} Queue of pending tasks */
        this.taskQueue = [];
        /** @type {Map<Worker, Object>} Map of active workers to their current tasks */
        this.activeWorkers = new Map();
    }

    //=========================================================================
    // Initialization Methods
    //=========================================================================
    /**
     * Initializes the worker pool
     * Creates and initializes the specified number of workers
     * @returns {Promise<void>}
     */
    async initialize() {
        for (let i = 0; i < this.size; i++) {
            try {
                const worker = new Worker(chrome.runtime.getURL('js/imageWorker.js'));
                await this.initializeWorker(worker, i);
                this.workers.push(worker);
            } catch (error) {
                console.warn(`Failed to initialize worker ${i}:`, error);
            }
        }
        console.log(`Initialized ${this.workers.length} workers`);
    }

    /**
     * Initializes a single worker
     * @param {Worker} worker - The worker to initialize
     * @param {number} id - The worker's ID
     * @returns {Promise<void>}
     */
    async initializeWorker(worker, id) {
        return new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                reject(new Error(`Worker ${id} initialization timeout`));
            }, 5000);

            const handleInit = (e) => {
                if (e.data?.type === 'WORKER_READY') {
                    clearTimeout(timeoutId);
                    worker.removeEventListener('message', handleInit);
                    if (e.data.success) {
                        worker.id = id;
                        resolve();
                    } else {
                        reject(new Error(e.data.error));
                    }
                }
            };

            worker.addEventListener('message', handleInit);
            worker.postMessage({ type: 'INIT' });
        });
    }

    //=========================================================================
    // Task Processing Methods
    //=========================================================================
    /**
     * Processes an image using an available worker
     * @param {ImageData} imageData - The image data to process
     * @returns {Promise<ImageData>} The processed image data
     */
    async processImage(imageData) {
        return new Promise((resolve, reject) => {
            const task = {
                imageData,
                resolve,
                reject,
                timestamp: Date.now()
            };

            this.taskQueue.push(task);
            this.processNextTask();
        });
    }

    /**
     * Processes the next task in the queue
     * @private
     */
    async processNextTask() {
        if (this.taskQueue.length === 0) return;

        const availableWorker = this.workers.find(w => !this.activeWorkers.has(w));
        if (!availableWorker) return;

        const task = this.taskQueue.shift();
        if (!task) return;

        try {
            this.activeWorkers.set(availableWorker, task);
            await this.executeTask(availableWorker, task);
        } catch (error) {
            this.activeWorkers.delete(availableWorker);
            task.reject(error);
            this.processNextTask();
        }
    }

    //=========================================================================
    // Task Execution Methods
    //=========================================================================
    /**
     * Executes a task on a specific worker
     * @param {Worker} worker - The worker to use
     * @param {Object} task - The task to execute
     * @private
     */
    async executeTask(worker, task) {
        const handleMessage = async (e) => {
            if (e.data?.type === 'IMAGE_PREPARED') {
                worker.removeEventListener('message', handleMessage);
                this.activeWorkers.delete(worker);

                if (e.data.success) {
                    task.resolve(e.data.data);
                } else {
                    task.reject(new Error(e.data.error));
                }

                this.processNextTask();
            }
        };

        worker.addEventListener('message', handleMessage);
        worker.postMessage({
            type: 'PREPARE_IMAGE',
            imageData: task.imageData,
            width: task.imageData.width,
            height: task.imageData.height
        });
    }

    //=========================================================================
    // Cleanup Methods
    //=========================================================================
    /**
     * Terminates all workers and cleans up resources
     */
    terminate() {
        this.workers.forEach(worker => worker.terminate());
        this.workers = [];
        this.activeWorkers.clear();
        this.taskQueue = [];
    }
} 