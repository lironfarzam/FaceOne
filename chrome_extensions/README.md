# FaceOne Chrome Extension

FaceOne is a powerful Chrome extension that provides real-time face detection and embedding generation for web images using FaceAPI.js and FaceNet. This extension can identify faces in images across the web, compare them against known faces, and optionally blur faces based on user preferences.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [File Structure](#file-structure)
5. [Workflow](#workflow)
6. [Technical Details](#technical-details)
7. [User Interface](#user-interface)
8. [Performance Optimizations](#performance-optimizations)
9. [Behind the Scenes: Code Execution Flow](#behind-the-scenes-code-execution-flow)
10. [Advanced Memory Management Recommendations](#advanced-memory-management-recommendations)

## Overview

FaceOne is designed to detect and analyze faces in web images in real-time. It uses machine learning models to:

- Detect faces in images across websites
- Generate face embeddings (numerical representations of faces)
- Compare faces against known embeddings to identify similarities
- Provide visual indicators for detected faces
- Optionally blur faces based on user preferences

The extension operates directly in the browser, processing images locally without sending data to external servers, ensuring privacy and security.

## Architecture

FaceOne uses a multi-layered architecture to ensure performance, security, and reliability:

1. **Content Script Layer**: Runs in the context of web pages to detect and process images
2. **Worker Layer**: Handles image processing in separate threads to maintain UI responsiveness
3. **Sandbox Layer**: Executes TensorFlow.js models in an isolated environment for security
4. **UI Layer**: Provides user controls through a popup interface

## Core Components

### 1. Manifest (manifest.json)

The manifest.json file defines the extension's metadata, permissions, and resource access. It specifies:

- Extension name, version, and description
- Required permissions (storage, activeTab)
- Content scripts to be injected into web pages
- Web-accessible resources (models, libraries)
- Security policies for the sandbox environment

### 2. Content Script (content.js)

The content script is the main engine of the extension, responsible for:

- Detecting images on web pages
- Managing the processing queue
- Coordinating with the sandbox for face detection and embedding generation
- Visualizing results (face frames, labels)
- Applying blur effects when needed
- Handling user interactions

It uses a sophisticated observer pattern to detect new images as they appear on the page and processes them according to user settings.

### 3. Worker Pool (workerPool.js)

The worker pool manages a collection of web workers for parallel image processing:

- Creates and maintains multiple worker instances
- Distributes tasks across workers for parallel processing
- Implements priority queuing for important images
- Handles worker lifecycle (creation, error handling, termination)
- Provides performance statistics and monitoring

### 4. Image Worker (imageWorker.js)

Each image worker runs in a separate thread and is responsible for:

- Pre-processing images for face detection
- Optimizing images for model input
- Handling image transformations (resizing, normalization)
- Communicating results back to the main thread

### 5. Sandbox (sandbox.html, sandboxInit.js)

The sandbox provides an isolated environment for running TensorFlow.js models:

- Loads and initializes TensorFlow.js
- Manages model loading and warmup
- Handles face detection and embedding generation
- Implements memory management for efficient operation
- Provides a secure execution context for ML operations

### 6. Popup UI (popup.html, popup.js)

The popup interface allows users to:

- Toggle between face detection and blur modes
- Configure detection settings (confidence threshold, auto-processing)
- Control visualization options (labels, face frames)
- Trigger manual reprocessing of images
- View status updates

## File Structure

```
chrome_extensions/
├── manifest.json       # Extension configuration
├── popup.html          # User interface HTML
├── popup.js            # UI interaction logic
├── sandbox.html        # Isolated TensorFlow environment
├── css/
│   └── styles.css      # UI styling
├── js/
│   ├── content.js      # Main content script
│   ├── workerPool.js   # Worker management
│   ├── imageWorker.js  # Image processing worker
│   └── sandboxInit.js  # TensorFlow initialization
├── lib/
│   ├── face-api.min.js # Face detection library
│   ├── tf.min.js       # TensorFlow.js core
│   └── tf-converter.min.js # TensorFlow model converter
├── models/
│   ├── FaceAPI/        # Face detection models
│   ├── FaceNet/        # Face embedding models
│   └── myModel/        # Custom similarity model
└── icons/              # Extension icons
```

## Workflow

The extension follows this workflow for processing images:

1. **Image Detection**: The content script observes the DOM for new images
2. **Queue Management**: Images are added to a processing queue with priority
3. **Worker Assignment**: The worker pool assigns images to available workers
4. **Pre-processing**: Workers prepare images for model input
5. **Face Detection**: The sandbox detects faces using FaceAPI.js
6. **Embedding Generation**: For detected faces, embeddings are generated using FaceNet
7. **Similarity Comparison**: Embeddings are compared against known faces
8. **Visualization**: Results are displayed on the page (frames, labels)
9. **Action Application**: Based on settings, actions like blurring may be applied

## Technical Details

### Face Detection

FaceOne uses a multi-stage approach for face detection:

1. **Initial Detection**: Uses FaceAPI.js to locate faces in images
2. **Landmark Detection**: Identifies 68 facial landmarks for precise face alignment
3. **Face Alignment**: Normalizes face orientation for consistent embedding generation
4. **Face Extraction**: Crops and prepares the face region for the embedding model

### Embedding Generation

Face embeddings are generated using a FaceNet model:

1. **Image Normalization**: Converts image to RGB and normalizes pixel values
2. **Model Inference**: Passes the normalized image through FaceNet
3. **Embedding Extraction**: Extracts the 512-dimensional face embedding
4. **Embedding Normalization**: Normalizes the embedding for consistent comparison

### Similarity Comparison

Face similarity is computed using:

1. **Vector Comparison**: Computes cosine similarity between embeddings
2. **Threshold Application**: Applies user-defined confidence threshold
3. **Result Classification**: Determines if faces match based on similarity score

## User Interface

The popup UI provides several controls:

1. **Mode Selection**: Choose between face detection and blur modes
2. **Auto-Processing**: Toggle automatic processing of images
3. **Show Labels**: Toggle display of detection results and similarity scores
4. **Show Face Frames**: Toggle colored frames around detected faces
5. **Confidence Threshold**: Adjust the minimum similarity percentage for matches
6. **Reprocess Button**: Manually trigger reprocessing of all images

## Performance Optimizations

FaceOne implements several optimizations for smooth operation:

1. **Worker Threading**: Offloads image processing to separate threads
2. **Batch Processing**: Groups similar operations for efficiency
3. **Priority Queuing**: Processes important images first
4. **Model Caching**: Reuses loaded models to reduce memory usage
5. **Embedding Caching**: Stores computed embeddings to avoid redundant processing
6. **Lazy Loading**: Loads resources only when needed
7. **Memory Management**: Implements automatic cleanup of unused resources
8. **Throttling**: Limits processing rate to maintain browser responsiveness
9. **Incremental Processing**: Processes images in stages as they become visible

## Behind the Scenes: Code Execution Flow

This section explains the detailed execution flow of the FaceOne Chrome extension, including the loading sequence, initialization process, and key functions in each file.

### Loading Sequence

When the Chrome extension is installed and activated, the following loading sequence occurs:

1. **manifest.json**: Chrome reads this file first to understand the extension's structure, permissions, and resources.
2. **content.js**: Injected into web pages based on the manifest's content_scripts configuration.
3. **workerPool.js**: Loaded by content.js to set up the worker infrastructure.
4. **popup.html/popup.js**: Loaded when the user clicks the extension icon in the toolbar.
5. **sandbox.html**: Loaded in an isolated context for secure TensorFlow.js execution.

### Initialization Process

The initialization process follows these steps:

1. **Extension Startup**:

   - Chrome loads the manifest and registers the extension
   - Content scripts are injected into matching web pages
   - Extension icon and popup are configured

2. **Content Script Initialization** (content.js):

   ```javascript
   // Core initialization function
   async function initialize() {
     // Set up mutation observers to detect new images
     setupImageObservers();
     // Load user settings from storage
     await loadSettings();
     // Initialize the worker pool for parallel processing
     initializeWorkerPool();
     // Set up the sandbox for TensorFlow operations
     await setupSandbox();
     // Load ML models
     await loadModels();
   }
   ```

3. **Worker Pool Setup** (workerPool.js):

   ```javascript
   // Worker pool initialization
   async function initialize() {
     // Create workers based on hardware capabilities
     for (let i = 0; i < this.options.maxWorkers; i++) {
       const worker = await this.createWorker(i);
       this.workers.set(i, worker);
       this.idleWorkers.add(worker);
     }
     // Warm up workers with dummy tasks
     await this.warmup();
     // Start task processor
     this.startTaskProcessor();
   }
   ```

4. **Sandbox Initialization** (sandbox.html, via sandboxInit.js):

   ```javascript
   // TensorFlow initialization
   async function initTensorFlow() {
     // Wait for TF to be ready
     await tf.ready();
     // Initialize backend
     await waitForBackend();
     // Set up WebGL backend if available
     if (tf.findBackend("webgl")) {
       await tf.setBackend("webgl");
       // Configure WebGL for optimal performance
       const backend = tf.backend();
       if (backend && backend.setWebGLFlag) {
         backend.setWebGLFlag("WEBGL_FORCE_F16_TEXTURES", true);
         backend.setWebGLFlag("WEBGL_VERSION", 2);
         backend.setWebGLFlag("WEBGL_PACK", true);
       }
     } else {
       // Fall back to CPU if WebGL is not available
       await tf.setBackend("cpu");
     }
   }
   ```

5. **Model Loading** (sandbox.html):

   ```javascript
   // Model loading function
   async function loadModel(modelPath) {
     // Load model with timeout protection
     const modelLoadPromise = tf.loadGraphModel(modelPath);
     const timeoutPromise = new Promise((_, reject) =>
       setTimeout(
         () => reject(new Error("Model load timeout")),
         MODEL_LOAD_TIMEOUT
       )
     );
     // Race to ensure loading doesn't hang
     faceNetModel = await Promise.race([modelLoadPromise, timeoutPromise]);

     // Warm up model with dummy inputs
     const dummyInputs = tf.tidy(() => tf.zeros([1, 160, 160, 3]));
     const warmupResult = await faceNetModel.predict(dummyInputs);
     await warmupResult.data();

     // Dispose warmup tensors
     dummyInputs.dispose();
     warmupResult.dispose();

     modelWarmedUp = true;
   }
   ```

6. **Popup UI Initialization** (popup.js):
   ```javascript
   // Popup initialization
   document.addEventListener("DOMContentLoaded", async () => {
     // Cache DOM elements
     UI.init();
     // Set up event handlers
     EventHandler.init();
     // Load and apply user settings
     await Settings.init();
   });
   ```

### Key Functions by File

#### 1. content.js - Main Content Script

```javascript
// Process an image to detect faces
async function processImage(img) {
  if (state.isProcessing || !state.modelsLoaded) return;

  state.isProcessing = true;
  try {
    // Create canvas and get image data
    const imageData = await getImageData(img);
    // Detect faces using FaceAPI
    const detections = await detectFaces(imageData);
    // For each detected face
    for (const detection of detections) {
      // Extract face region
      const faceData = extractFace(imageData, detection);
      // Generate face embedding
      const embedding = await generateEmbedding(faceData);
      // Compare with known faces
      const matches = await compareFaces(embedding);
      // Visualize results
      visualizeResults(img, detection, matches);
      // Apply actions (blur, etc.) based on settings
      applyActions(img, detection, matches);
    }
  } catch (error) {
    console.error("Image processing error:", error);
  } finally {
    state.isProcessing = false;
  }
}

// Detect faces in an image
async function detectFaces(imageData) {
  return new Promise((resolve, reject) => {
    // Send message to sandbox for face detection
    sandbox.postMessage({
      type: "DETECT_FACES",
      imageData: imageData,
    });

    // Set up one-time message handler for response
    const handleMessage = (event) => {
      if (event.data.type === "FACES_DETECTED") {
        window.removeEventListener("message", handleMessage);
        if (event.data.success) {
          resolve(event.data.detections);
        } else {
          reject(new Error(event.data.error));
        }
      }
    };

    window.addEventListener("message", handleMessage);
  });
}

// Generate embedding for a face
async function generateEmbedding(faceData) {
  // Use worker pool to distribute processing
  return workerPool.processImage(faceData);
}
```

#### 2. workerPool.js - Worker Management

```javascript
// Process an image using an available worker
async function processImage(imageData, priority = 0) {
  const task = {
    imageData,
    attempts: 0,
    startTime: Date.now(),
  };

  return new Promise((resolve, reject) => {
    // Add task to queue with priority
    this.taskQueue.add({
      task,
      resolve,
      reject,
      priority,
    });
    // Start processing
    this.processNextTask();
  });
}

// Process next task in queue
async function processNextTask() {
  if (this.idleWorkers.size === 0 || this.taskQueue.items.length === 0) {
    return;
  }

  // Get next available worker and task
  const worker = this.idleWorkers.values().next().value;
  const task = this.taskQueue.next();

  if (!task) return;

  // Mark worker as busy
  this.idleWorkers.delete(worker);

  try {
    // Execute task on worker
    const result = await this.executeTask(worker, task);
    task.resolve(result);

    // Update performance stats
    this.updateStats(task);
  } catch (error) {
    // Retry logic for failed tasks
    if (task.attempts < this.options.retryAttempts) {
      task.attempts++;
      this.taskQueue.add(task, task.priority + 1);
    } else {
      task.reject(error);
      this.stats.errors++;
    }
  } finally {
    // Return worker to idle pool
    this.idleWorkers.add(worker);
    // Process next task
    this.processNextTask();
  }
}
```

#### 3. imageWorker.js - Image Processing Worker

```javascript
// Main message handler for the worker
self.onmessage = async function (e) {
  const { type, imageData, width, height } = e.data;

  try {
    switch (type) {
      case "INIT":
        await handleInit();
        break;

      case "PROCESS_IMAGE":
        await handleImageProcessing(imageData, width, height);
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    self.postMessage({
      type: `${type}_FAILED`,
      success: false,
      error: error.message,
    });
  }
};

// Process image with performance optimizations
async function handleImageProcessing(imageData, width, height) {
  if (!state.initialized) {
    throw new Error("Worker not initialized");
  }

  const startTime = performance.now();
  state.processingCount++;

  try {
    // Resize canvas if needed
    if (sharedCanvas.width < width || sharedCanvas.height < height) {
      sharedCanvas.width = width;
      sharedCanvas.height = height;
    }

    // Process image
    const processedData = await processImageOptimized(imageData, width, height);

    // Track performance
    state.lastProcessingTime = performance.now() - startTime;

    self.postMessage({
      type: "IMAGE_PROCESSED",
      success: true,
      data: processedData,
      stats: {
        processingTime: state.lastProcessingTime,
        totalProcessed: state.processingCount,
      },
    });
  } catch (error) {
    throw error;
  }
}
```

#### 4. sandbox.html - TensorFlow Execution Environment

```javascript
// Generate embedding for a face image
async function generateEmbedding(imageData) {
  if (!isTfBackendReady()) {
    throw new Error("TensorFlow not initialized or ready");
  }

  if (!faceNetModel) {
    throw new Error("Model not loaded");
  }

  return tf.tidy(() => {
    try {
      // Convert image data to tensor
      const img = tf.tensor(imageData, [160, 160, 4]);

      // Extract RGB channels
      const rgb = img.slice([0, 0, 0], [-1, -1, 3]);

      // Preprocess for model input
      const processed = rgb.expandDims(0).toFloat().div(127.5).sub(1);

      // Generate embedding
      const embedding = faceNetModel.predict(processed);
      const embeddingData = embedding.squeeze();

      // Normalize embedding
      const normalizedEmbedding = tf.div(embeddingData, tf.norm(embeddingData));

      // Convert to regular array
      const finalEmbedding = normalizedEmbedding.dataSync();

      return {
        type: "EMBEDDING_GENERATED",
        success: true,
        embedding: Array.from(finalEmbedding),
        fromCache: false,
      };
    } catch (error) {
      return {
        type: "EMBEDDING_GENERATED",
        success: false,
        error: error.message,
      };
    }
  });
}

// Compare two face embeddings
async function compareFaceEmbeddings(embedding1, embedding2) {
  if (!this.isReady("myModel")) {
    throw new Error("Similarity model not ready");
  }

  return tf.tidy(() => {
    try {
      // Convert embeddings to tensors
      const tensor1 = tf.tensor2d([embedding1], [1, 512]);
      const tensor2 = tf.tensor2d([embedding2], [1, 512]);

      // Run inference
      const similarity = this.models.myModel.predict([tensor1, tensor2]);
      const result = similarity.dataSync()[0];

      return {
        type: "SIMILARITY_COMPUTED",
        success: true,
        similarity: result,
      };
    } catch (error) {
      return {
        type: "SIMILARITY_COMPUTED",
        success: false,
        error: error.message,
      };
    }
  });
}
```

#### 5. popup.js - User Interface Management

```javascript
// Settings management
const Settings = {
  async init() {
    const defaults = {
      processingMode: "face_detection",
      autoProcessImages: true,
      addLabel: true,
      frameFaceDetected: true,
      confidenceThreshold: 70,
    };

    try {
      // Load saved settings or use defaults
      const saved = await chrome.storage.sync.get(defaults);
      this.applySettings(saved);
      UI.cache.set("settings", saved);
    } catch (error) {
      console.error("Settings initialization failed:", error);
      this.applySettings(defaults);
    }
  },

  // Apply settings to UI elements
  applySettings(settings) {
    // Update mode selection
    const modeButtons = document.querySelectorAll(".mode-button");
    modeButtons.forEach((button) => {
      button.classList.toggle(
        "active",
        button.id === `${settings.processingMode}Mode`
      );
    });

    // Update checkboxes
    document.getElementById("autoProcessImages").checked =
      settings.autoProcessImages;
    document.getElementById("addLabel").checked = settings.addLabel;
    document.getElementById("frameFaceDetected").checked =
      settings.frameFaceDetected;

    // Update slider
    const slider = document.getElementById("confidenceThreshold");
    slider.value = settings.confidenceThreshold;
    document.getElementById(
      "confidenceValue"
    ).textContent = `${settings.confidenceThreshold}%`;

    // Update UI visibility based on mode
    updateUIForMode(settings.processingMode);
  },

  // Update a setting and save it
  async update(key, value) {
    const settings = UI.cache.get("settings") || {};
    settings[key] = value;

    try {
      // Save to Chrome storage
      await chrome.storage.sync.set({ [key]: value });
      // Notify content script of changes
      await this.notifyContentScript(settings);
      // Update cache
      UI.cache.set("settings", settings);
    } catch (error) {
      console.error("Settings update failed:", error);
      StatusManager.show("Settings update failed", "error");
    }
  },
};

// Update UI based on selected mode
function updateUIForMode(mode) {
  // Show/hide settings based on mode
  document.querySelectorAll("[data-mode]").forEach((el) => {
    const modes = el.dataset.mode.split(",");
    el.classList.toggle(
      "hidden",
      !modes.includes(mode) && !modes.includes("both")
    );
  });
}
```

### Data Flow Between Components

The extension components communicate through several mechanisms:

1. **Content Script ↔ Sandbox**:

   - Uses `window.postMessage()` for bidirectional communication
   - Sends image data for processing
   - Receives detection results and embeddings

2. **Content Script ↔ Worker Pool**:

   - Direct function calls to queue tasks
   - Promise-based interface for results

3. **Worker Pool ↔ Image Workers**:

   - Uses the Web Worker `postMessage()` API
   - Sends image data for preprocessing
   - Receives processed results

4. **Popup ↔ Content Script**:

   - Uses Chrome's messaging API (`chrome.tabs.sendMessage()`)
   - Sends user settings and commands
   - Receives status updates

5. **Settings Storage**:
   - Uses Chrome's storage API (`chrome.storage.sync`)
   - Persists user preferences across browser sessions

### Memory Management

The extension implements sophisticated memory management to prevent leaks:

1. **TensorFlow.js Memory**:

   ```javascript
   // Safely dispose tensors
   function safeDisposeTensors() {
     try {
       if (!isTfEngineAvailable()) return;

       const tensorsArray = Array.from(tensorsToDispose);
       tensorsToDispose.clear(); // Clear first to prevent circular issues

       tf.tidy(() => {
         tensorsArray.forEach((tensor) => {
           try {
             if (tensor && !tensor.isDisposed && tensor.dispose) {
               tensor.dispose();
             }
           } catch (e) {
             // Silently ignore disposal errors
           }
         });
       });
     } catch (e) {
       // Silently fail if cleanup isn't possible
     }
   }

   // Periodic memory cleanup
   const memoryCleanupInterval = setInterval(() => {
     try {
       if (!isTfEngineAvailable()) return;

       tf.tidy(() => {
         try {
           const memoryInfo = tf.memory();
           if (memoryInfo.numTensors > 100 || memoryInfo.numBytes > 50000000) {
             console.warn("Memory usage:", {
               tensors: memoryInfo.numTensors,
               bytes: Math.round(memoryInfo.numBytes / (1024 * 1024)) + " MB",
               gpu:
                 Math.round((memoryInfo.numBytesInGPU || 0) / (1024 * 1024)) +
                 " MB",
             });

             if (memoryInfo.numTensors > 1000) {
               safeDisposeTensors();
               if (isTfEngineAvailable()) {
                 try {
                   tf.engine().endScope();
                   tf.engine().startScope();
                 } catch (e) {
                   // Silently ignore scope errors
                 }
               }
             }
           }
         } catch (e) {
           // Silently ignore memory info errors
         }
       });
     } catch (error) {
       console.warn("Memory cleanup:", error);
     }
   }, 30000);
   ```

2. **Image Cache Management**:

   ```javascript
   // Add cache for processed images
   const imageCache = {
     images: new Map(), // Map to store image data
     maxSize: 1000,

     add(src, isProcessed = true, canDelete = false) {
       if (this.images.size >= this.maxSize) {
         // Remove oldest entry by converting to array and removing first element
         const sources = Array.from(this.images.keys());
         this.images.delete(sources[0]);
       }
       this.images.set(src, {
         isProcessed,
         canDelete,
         timestamp: Date.now(),
       });
     },

     // Additional cache management methods...
   };
   ```

3. **Worker Resource Management**:
   ```javascript
   // Clean up resources
   function terminate() {
     this.workers.forEach((worker) => worker.terminate());
     this.workers.clear();
     this.idleWorkers.clear();
     this.taskQueue = new PriorityQueue();
   }
   ```

## Advanced Memory Management Recommendations

To improve memory management and prevent performance issues in the FaceOne Chrome extension, consider implementing the following optimizations:

### 1. TensorFlow.js Memory Optimizations

#### Implement Aggressive Tensor Cleanup

```javascript
// Enhanced tensor disposal with reference tracking
const tensorTracker = {
  activeReferences: new Map(),

  track(tensor, context = "unknown") {
    if (!tensor || !tensor.id) return tensor;
    this.activeReferences.set(tensor.id, {
      tensor,
      context,
      createdAt: Date.now(),
      stack: new Error().stack,
    });
    return tensor;
  },

  dispose(tensor) {
    if (!tensor || !tensor.id) return;
    if (tensor.dispose && !tensor.isDisposed) {
      tensor.dispose();
    }
    this.activeReferences.delete(tensor.id);
  },

  disposeAll() {
    tf.tidy(() => {
      const tensors = Array.from(this.activeReferences.values());
      tensors.forEach((ref) => {
        try {
          if (ref.tensor && !ref.tensor.isDisposed && ref.tensor.dispose) {
            ref.tensor.dispose();
          }
        } catch (e) {
          console.warn(`Failed to dispose tensor from ${ref.context}:`, e);
        }
      });
      this.activeReferences.clear();
    });
  },

  getLeaks() {
    const now = Date.now();
    const leaks = [];
    this.activeReferences.forEach((ref, id) => {
      if (now - ref.createdAt > 60000) {
        // Older than 1 minute
        leaks.push({
          id,
          context: ref.context,
          age: Math.round((now - ref.createdAt) / 1000) + "s",
          stack: ref.stack,
        });
      }
    });
    return leaks;
  },
};

// Usage example
function processImage(imageData) {
  return tf.tidy(() => {
    const img = tensorTracker.track(tf.tensor(imageData), "processImage:input");
    const processed = tensorTracker.track(
      img.expandDims(0),
      "processImage:expand"
    );
    // ... processing ...
    return result; // tf.tidy will clean up tracked tensors
  });
}
```

#### Implement Memory-Aware Processing Queue

```javascript
// Memory-aware task scheduler
class MemoryAwareScheduler {
  constructor(options = {}) {
    this.options = {
      maxTensors: 1000,
      maxBytes: 200 * 1024 * 1024, // 200MB
      checkInterval: 1000, // 1 second
      pauseThreshold: 0.9, // 90% of max
      resumeThreshold: 0.7, // 70% of max
      ...options,
    };

    this.queue = [];
    this.isPaused = false;
    this.isProcessing = false;

    // Start memory monitoring
    this.startMonitoring();
  }

  startMonitoring() {
    this.monitorInterval = setInterval(() => {
      this.checkMemoryUsage();
    }, this.options.checkInterval);
  }

  checkMemoryUsage() {
    try {
      const memInfo = tf.memory();
      const tensorUsage = memInfo.numTensors / this.options.maxTensors;
      const memoryUsage = memInfo.numBytes / this.options.maxBytes;
      const usage = Math.max(tensorUsage, memoryUsage);

      if (!this.isPaused && usage > this.options.pauseThreshold) {
        this.pause();
        // Force garbage collection and tensor cleanup
        tensorTracker.disposeAll();
        tf.engine().endScope();
        tf.engine().startScope();
        console.warn("Memory pressure detected, pausing processing");
      } else if (this.isPaused && usage < this.options.resumeThreshold) {
        this.resume();
        console.log("Memory pressure relieved, resuming processing");
      }
    } catch (e) {
      console.warn("Error checking memory usage:", e);
    }
  }

  pause() {
    this.isPaused = true;
  }

  resume() {
    this.isPaused = false;
    if (!this.isProcessing) {
      this.processNext();
    }
  }

  enqueue(task, priority = 0) {
    return new Promise((resolve, reject) => {
      this.queue.push({ task, priority, resolve, reject });
      this.queue.sort((a, b) => b.priority - a.priority);

      if (!this.isProcessing && !this.isPaused) {
        this.processNext();
      }
    });
  }

  async processNext() {
    if (this.isPaused || this.queue.length === 0) {
      this.isProcessing = false;
      return;
    }

    this.isProcessing = true;
    const { task, resolve, reject } = this.queue.shift();

    try {
      // Execute task in a tidy environment
      const result = await tf.tidy(() => task());
      resolve(result);
    } catch (error) {
      reject(error);
    } finally {
      // Process next task
      this.processNext();
    }
  }

  destroy() {
    clearInterval(this.monitorInterval);
    this.queue = [];
    this.isProcessing = false;
  }
}

// Usage
const scheduler = new MemoryAwareScheduler();
scheduler.enqueue(() => processImage(imageData), 1);
```

### 2. WebWorker Memory Optimizations

#### Implement Worker Lifecycle Management

```javascript
// Enhanced worker pool with lifecycle management
class EnhancedWorkerPool {
  constructor(options = {}) {
    this.options = {
      maxWorkers: navigator.hardwareConcurrency || 4,
      maxIdleTime: 30000, // 30 seconds
      maxProcessingTime: 10000, // 10 seconds
      maxTasksPerWorker: 100,
      ...options,
    };

    this.workers = new Map();
    this.idleWorkers = new Set();
    this.taskQueue = [];
    this.workerStats = new Map();
  }

  async createWorker(id) {
    const worker = new Worker(chrome.runtime.getURL("js/imageWorker.js"));

    // Initialize worker
    await this.initWorker(worker, id);

    // Set up worker monitoring
    this.workerStats.set(id, {
      created: Date.now(),
      lastActive: Date.now(),
      taskCount: 0,
      errors: 0,
      avgProcessingTime: 0,
    });

    // Set up idle timeout
    this.setupIdleTimeout(id);

    return worker;
  }

  setupIdleTimeout(id) {
    const worker = this.workers.get(id);
    if (!worker) return;

    const stats = this.workerStats.get(id);
    if (!stats) return;

    // Clear existing timeout
    if (stats.idleTimeout) {
      clearTimeout(stats.idleTimeout);
    }

    // Set new timeout
    stats.idleTimeout = setTimeout(() => {
      const idleTime = Date.now() - stats.lastActive;
      if (idleTime > this.options.maxIdleTime && this.idleWorkers.has(worker)) {
        // Worker has been idle too long, terminate and recreate
        this.recycleWorker(id);
      } else {
        // Check again later
        this.setupIdleTimeout(id);
      }
    }, this.options.maxIdleTime / 2);
  }

  async recycleWorker(id) {
    const worker = this.workers.get(id);
    if (!worker) return;

    // Remove from idle set
    this.idleWorkers.delete(worker);

    // Terminate worker
    worker.terminate();

    // Create new worker
    const newWorker = await this.createWorker(id);

    // Update maps
    this.workers.set(id, newWorker);
    this.idleWorkers.add(newWorker);

    console.log(`Recycled worker #${id} after extended idle period`);
  }

  async executeTask(worker, task) {
    const id = this.getWorkerId(worker);
    const stats = this.workerStats.get(id);

    if (!stats) {
      throw new Error("Worker stats not found");
    }

    // Update stats
    stats.lastActive = Date.now();
    stats.taskCount++;

    // Set up task timeout
    const timeoutPromise = new Promise((_, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error("Task execution timeout"));
        // Force recycle the worker
        this.recycleWorker(id);
      }, this.options.maxProcessingTime);

      // Store timeout ID for cleanup
      task.timeoutId = timeoutId;
    });

    // Execute task with timeout
    try {
      const result = await Promise.race([
        this.doExecuteTask(worker, task),
        timeoutPromise,
      ]);

      // Clear timeout
      if (task.timeoutId) {
        clearTimeout(task.timeoutId);
      }

      // Check if worker needs recycling
      if (stats.taskCount >= this.options.maxTasksPerWorker) {
        // Schedule worker recycling
        setTimeout(() => {
          if (this.idleWorkers.has(worker)) {
            this.recycleWorker(id);
          }
        }, 0);
      }

      return result;
    } catch (error) {
      // Clear timeout
      if (task.timeoutId) {
        clearTimeout(task.timeoutId);
      }

      // Update error stats
      stats.errors++;

      // Recycle worker if too many errors
      if (stats.errors > 3) {
        this.recycleWorker(id);
      }

      throw error;
    }
  }

  getWorkerId(worker) {
    for (const [id, w] of this.workers.entries()) {
      if (w === worker) return id;
    }
    return null;
  }

  // Other methods...
}
```

### 3. Image Processing Optimizations

#### Implement Progressive Image Processing

```javascript
// Progressive image processing
async function processImageProgressively(img) {
  // Skip if image is too small
  if (img.width < 100 || img.height < 100) {
    return;
  }

  // Process at multiple scales for large images
  const maxDimension = Math.max(img.width, img.height);

  if (maxDimension > 1024) {
    // First pass: low resolution for quick results
    const scale = 512 / maxDimension;
    const lowResData = await getScaledImageData(img, scale);
    const lowResResults = await detectFaces(lowResData);

    // Show preliminary results
    if (lowResResults.length > 0) {
      visualizeResults(img, lowResResults, { preliminary: true });
    }

    // Second pass: full resolution for accuracy
    if (lowResResults.length > 0) {
      const fullResData = await getImageData(img);
      const fullResResults = await detectFaces(fullResData);

      // Update with final results
      visualizeResults(img, fullResResults, { preliminary: false });
    }
  } else {
    // Standard processing for smaller images
    const imageData = await getImageData(img);
    const results = await detectFaces(imageData);
    visualizeResults(img, results, { preliminary: false });
  }
}
```

#### Implement Smart Image Caching

```javascript
// Enhanced image cache with size-aware management
class SmartImageCache {
  constructor(options = {}) {
    this.options = {
      maxEntries: 1000,
      maxMemoryUsage: 100 * 1024 * 1024, // 100MB
      entryTTL: 5 * 60 * 1000, // 5 minutes
      ...options,
    };

    this.cache = new Map();
    this.memoryUsage = 0;

    // Start cleanup interval
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, 60000); // Cleanup every minute
  }

  estimateSize(imageData) {
    if (!imageData) return 0;
    if (imageData.data && imageData.data.length) {
      return imageData.data.length;
    }
    return 0;
  }

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
      accessCount: 0,
    });

    this.memoryUsage += size;
  }

  get(key) {
    const entry = this.cache.get(key);
    if (!entry) return null;

    // Update access stats
    entry.lastAccessed = Date.now();
    entry.accessCount++;

    return entry.data;
  }

  remove(key) {
    const entry = this.cache.get(key);
    if (!entry) return;

    this.memoryUsage -= entry.size;
    this.cache.delete(key);
  }

  makeRoom(requiredSize) {
    // If cache is empty, nothing to do
    if (this.cache.size === 0) return;

    // If required size is larger than max, we can't cache it
    if (requiredSize > this.options.maxMemoryUsage) {
      console.warn("Requested cache entry exceeds maximum cache size");
      return;
    }

    // Sort entries by priority (last accessed, then access count)
    const entries = Array.from(this.cache.entries())
      .map(([key, entry]) => ({ key, entry }))
      .sort((a, b) => {
        // First sort by last accessed (oldest first)
        const timeDiff = a.entry.lastAccessed - b.entry.lastAccessed;
        if (Math.abs(timeDiff) > 60000) {
          // If more than 1 minute difference
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

      if (this.memoryUsage + requiredSize <= this.options.maxMemoryUsage) {
        break;
      }
    }
  }

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
    expiredKeys.forEach((key) => this.remove(key));

    // If still too many entries, remove oldest
    if (this.cache.size > this.options.maxEntries) {
      const entries = Array.from(this.cache.entries())
        .map(([key, entry]) => ({ key, timestamp: entry.timestamp }))
        .sort((a, b) => a.timestamp - b.timestamp);

      const toRemove = entries.slice(
        0,
        this.cache.size - this.options.maxEntries
      );
      toRemove.forEach(({ key }) => this.remove(key));
    }
  }

  destroy() {
    clearInterval(this.cleanupInterval);
    this.cache.clear();
    this.memoryUsage = 0;
  }
}
```

### 4. Sandbox Memory Management

#### Implement Sandbox Lifecycle Management

```javascript
// Enhanced sandbox management
class SandboxManager {
  constructor() {
    this.iframe = null;
    this.isReady = false;
    this.pendingRequests = new Map();
    this.requestId = 0;
    this.lastActivity = Date.now();
    this.healthCheckInterval = null;
    this.restartCount = 0;
  }

  async initialize() {
    if (this.iframe) {
      this.destroy();
    }

    // Create sandbox iframe
    this.iframe = document.createElement("iframe");
    this.iframe.src = chrome.runtime.getURL("sandbox.html");
    this.iframe.style.display = "none";
    document.body.appendChild(this.iframe);

    // Set up message handler
    window.addEventListener("message", this.handleMessage.bind(this));

    // Wait for sandbox to be ready
    await this.waitForReady();

    // Start health check
    this.startHealthCheck();

    return this.iframe;
  }

  async waitForReady() {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("Sandbox initialization timeout"));
      }, 10000);

      const readyHandler = (event) => {
        if (event.data && event.data.type === "SANDBOX_READY") {
          window.removeEventListener("message", readyHandler);
          clearTimeout(timeout);
          this.isReady = true;
          resolve();
        }
      };

      window.addEventListener("message", readyHandler);
    });
  }

  startHealthCheck() {
    this.healthCheckInterval = setInterval(() => {
      this.checkHealth();
    }, 30000); // Check every 30 seconds
  }

  async checkHealth() {
    try {
      // Check if sandbox is responsive
      const status = await this.sendMessage({ type: "HEALTH_CHECK" }, 5000);

      // Check memory usage
      if (status.memoryInfo && status.memoryInfo.numTensors > 1000) {
        console.warn(
          "High tensor count in sandbox:",
          status.memoryInfo.numTensors
        );
        await this.sendMessage({ type: "CLEANUP" });
      }

      // Check for long inactivity
      const inactiveTime = Date.now() - this.lastActivity;
      if (inactiveTime > 5 * 60 * 1000) {
        // 5 minutes
        console.log("Sandbox inactive for 5 minutes, performing cleanup");
        await this.sendMessage({ type: "CLEANUP" });
      }
    } catch (error) {
      console.error("Sandbox health check failed:", error);

      // Restart sandbox if health check fails
      this.restartCount++;
      if (this.restartCount <= 3) {
        console.warn(`Restarting sandbox (attempt ${this.restartCount})`);
        await this.restart();
      } else {
        console.error("Too many sandbox restart attempts, giving up");
      }
    }
  }

  async restart() {
    try {
      this.destroy();
      await this.initialize();
      console.log("Sandbox restarted successfully");
    } catch (error) {
      console.error("Failed to restart sandbox:", error);
    }
  }

  sendMessage(message, timeout = 30000) {
    return new Promise((resolve, reject) => {
      const id = this.requestId++;
      message.id = id;

      // Set up timeout
      const timeoutId = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error("Sandbox request timeout"));
      }, timeout);

      // Store pending request
      this.pendingRequests.set(id, { resolve, reject, timeoutId });

      // Send message
      this.iframe.contentWindow.postMessage(message, "*");

      // Update activity timestamp
      this.lastActivity = Date.now();
    });
  }

  handleMessage(event) {
    const data = event.data;
    if (!data || !data.id) return;

    const request = this.pendingRequests.get(data.id);
    if (!request) return;

    // Clear timeout
    clearTimeout(request.timeoutId);

    // Remove from pending requests
    this.pendingRequests.delete(data.id);

    // Update activity timestamp
    this.lastActivity = Date.now();

    // Resolve or reject
    if (data.error) {
      request.reject(new Error(data.error));
    } else {
      request.resolve(data);
    }
  }

  destroy() {
    // Clear interval
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }

    // Clear all pending requests
    this.pendingRequests.forEach((request) => {
      clearTimeout(request.timeoutId);
      request.reject(new Error("Sandbox destroyed"));
    });
    this.pendingRequests.clear();

    // Remove iframe
    if (this.iframe) {
      document.body.removeChild(this.iframe);
      this.iframe = null;
    }

    this.isReady = false;
  }
}
```

### 5. Implementation Strategy

To implement these memory management improvements, follow this phased approach:

1. **Phase 1: Monitoring**

   - Implement the tensor tracking system to identify memory leaks
   - Add detailed logging of memory usage patterns
   - Create a dashboard for visualizing memory usage

2. **Phase 2: Critical Fixes**

   - Implement the enhanced tensor disposal system
   - Add worker lifecycle management
   - Implement sandbox health checks and recovery

3. **Phase 3: Advanced Optimizations**

   - Implement the memory-aware scheduler
   - Add progressive image processing
   - Implement the smart image cache

4. **Phase 4: Testing and Validation**
   - Perform stress testing with large numbers of images
   - Monitor memory usage over extended periods
   - Validate performance improvements

### 6. Best Practices for Ongoing Development

1. **Always use tf.tidy()**

   - Wrap all TensorFlow.js operations in tf.tidy() to automatically clean up tensors
   - Explicitly return tensors that should be preserved

2. **Implement memory budgets**

   - Set explicit limits on memory usage for different components
   - Pause processing when approaching limits

3. **Use WeakMap for caching**

   - When caching results tied to DOM elements, use WeakMap to allow garbage collection

4. **Implement graceful degradation**

   - When memory pressure is high, reduce quality or disable features rather than crashing

5. **Regular cleanup cycles**
   - Implement periodic deep cleanup during idle times
   - Clear caches when tab visibility changes

By implementing these recommendations, the FaceOne Chrome extension will achieve significantly improved memory management, leading to better performance, stability, and user experience.

---

This extension demonstrates advanced browser capabilities for machine learning and computer vision tasks while maintaining performance and security.
