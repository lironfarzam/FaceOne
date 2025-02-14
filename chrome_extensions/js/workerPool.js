class WorkerPool {
    constructor(size = navigator.hardwareConcurrency || 4) {
        this.size = size;
        this.workers = [];
        this.taskQueue = [];
        this.activeWorkers = new Map(); // worker -> task mapping
    }

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

    async processNextTask() {
        if (this.taskQueue.length === 0) return;

        const availableWorker = this.workers.find(w => !this.activeWorkers.has(w));
        if (!availableWorker) return;

        const task = this.taskQueue.shift();
        if (!task) return;

        try {
            this.activeWorkers.set(availableWorker, task);

            const handleMessage = async (e) => {
                if (e.data?.type === 'IMAGE_PREPARED') {
                    availableWorker.removeEventListener('message', handleMessage);
                    this.activeWorkers.delete(availableWorker);

                    if (e.data.success) {
                        task.resolve(e.data.data);
                    } else {
                        task.reject(new Error(e.data.error));
                    }

                    this.processNextTask();
                }
            };

            availableWorker.addEventListener('message', handleMessage);
            availableWorker.postMessage({
                type: 'PREPARE_IMAGE',
                imageData: task.imageData,
                width: task.imageData.width,
                height: task.imageData.height
            });

        } catch (error) {
            this.activeWorkers.delete(availableWorker);
            task.reject(error);
            this.processNextTask();
        }
    }

    terminate() {
        this.workers.forEach(worker => worker.terminate());
        this.workers = [];
        this.activeWorkers.clear();
        this.taskQueue = [];
    }
} 