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
10. [Advanced Techniques: Image Processing](#advanced-techniques-image-processing)
11. [Deep Dive: Face Detection and Recognition](#deep-dive-face-detection-and-recognition)
12. [Storage and Persistence](#storage-and-persistence)
13. [Security and Privacy Considerations](#security-and-privacy-considerations)
14. [Development and Contribution](#development-and-contribution)
15. [Scientific Foundations](#scientific-foundations)
16. [Recent Improvements: Enhanced Blur Persistence](#recent-improvements-enhanced-blur-persistence)

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
5. **Storage Layer**: Manages persistent data for blurred images and settings
6. **Mutation Observer Layer**: Monitors DOM changes to handle dynamically loaded content

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
- Clear blurred images list

### 7. Blur Tracker (blurTracker.js)

The blur tracker is a lightweight system for managing blurred images:

- Stores URLs of images that need to be blurred
- Normalizes URLs to handle dynamic parameters
- Provides persistent storage using Chrome's storage API
- Implements timestamp-based cleanup for old entries
- Monitors DOM changes to apply blur effects to newly added images

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
│   ├── sandboxInit.js  # TensorFlow initialization
│   └── blurTracker.js  # Blurred image management
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
10. **Result Storage**: Blurred image information is stored for persistence
11. **DOM Monitoring**: MutationObserver watches for new content and applies saved settings

## Technical Details

### Face Detection

FaceOne uses a multi-stage approach for face detection:

1. **Initial Detection**: Uses FaceAPI.js to locate faces in images

   - SSD MobileNet for large images (>300px)
   - TinyFaceDetector for smaller images
   - Detection parameters tuned for optimal performance/accuracy balance

2. **Landmark Detection**: Identifies 68 facial landmarks for precise face alignment

   - Eye positions for rotation normalization
   - Facial contours for accurate boundary detection
   - Landmark points used to calculate face orientation

3. **Face Alignment**: Normalizes face orientation for consistent embedding generation

   - Rotation correction based on eye positions
   - Scale normalization to standardized size (160x160 pixels)
   - Multiple orientation attempts for challenging angles (0°, ±45°, ±90°)

4. **Face Extraction**: Crops and prepares the face region for the embedding model
   - Canvas-based extraction for efficient memory usage
   - Padding to capture complete facial features
   - Resolution adaptations based on original image quality

### Embedding Generation

Face embeddings are generated using a FaceNet model:

1. **Image Normalization**: Converts image to RGB and normalizes pixel values

   - Pixel values scaled to [-1, 1] range
   - RGB channel extraction and optimization
   - Tensor conversion with appropriate dimensions [1, 160, 160, 3]

2. **Model Inference**: Passes the normalized image through FaceNet

   - WebGL acceleration when available
   - CPU fallback for compatibility
   - Batch processing optimization

3. **Embedding Extraction**: Extracts the 512-dimensional face embedding

   - Dense vector representation of facial features
   - Float32Array for efficient storage
   - Compact numerical representation of facial identity

4. **Embedding Normalization**: Normalizes the embedding for consistent comparison
   - L2 normalization to unit vector
   - Ensures consistent similarity comparison regardless of image conditions
   - Reduces effects of lighting and pose variations

### Similarity Comparison

Face similarity is computed using:

1. **Vector Comparison**: Computes cosine similarity between embeddings

   - Dot product of normalized embeddings
   - Range from -1 (opposite) to 1 (identical)
   - Adjusted to 0-100% scale for user interface

2. **Threshold Application**: Applies user-defined confidence threshold

   - Configurable via UI slider (0-100%)
   - Default threshold set to 70% for balanced results
   - Dynamically adjustable for different sensitivity needs

3. **Result Classification**: Determines if faces match based on similarity score
   - Binary decision (match/no match) based on threshold
   - Confidence score displayed to user
   - Multi-face handling for images with multiple detections

### Blurring Mechanism

The blurring process follows these steps:

1. **Blur Decision**: Determines if an image should be blurred

   - Based on similarity comparison results
   - User-defined confidence threshold
   - Previously stored blur decisions

2. **Blur Application**: Applies blur effect using multiple methods

   - CSS filter property (primary method)
   - HTML class-based targeting
   - Attribute-based selectors for persistence

3. **Blur Persistence**: Ensures blur effect remains consistent

   - URL-based tracking with normalization
   - Timestamp-based freshness management
   - Multi-level targeting for DOM structure changes

4. **DOM Monitoring**: Applies blur to dynamically added content
   - MutationObserver to detect new images
   - URL matching against stored blur list
   - Dynamic CSS rule injection for comprehensive targeting

## User Interface

FaceOne provides a clean, intuitive interface for controlling its behavior:

### Settings Panel

Access the extension's settings by clicking the FaceOne icon in your Chrome toolbar. The settings panel allows you to:

- Switch between **Face Detection** and **Blur** modes
- Enable/disable automatic image processing
- Show/hide labels and face frames
- Adjust the confidence threshold for face matching
- Enable debug mode for troubleshooting (see Debug Mode section below)
- Reprocess all images on the current page
- Clear all blurred images

### Visual Indicators

When processing images, FaceOne provides visual feedback:

- Green frames indicate detected faces
- Red frames indicate faces matching known embeddings
- Optional labels show detection confidence and processing time
- Blurred images indicate faces that have been automatically obscured

### Debug Mode

The extension includes a debug mode intended for developers and troubleshooting:

#### Purpose

Debug mode enables verbose console logging that shows detailed information about each step of the face detection and processing pipeline. This is useful for:

- Troubleshooting issues with face detection
- Understanding the extension's processing flow
- Diagnosing performance problems
- Development and testing

#### Enabling Debug Mode

1. Click on the FaceOne extension icon to open the settings panel
2. Toggle the "Debug Mode" switch to the ON position
3. Refresh the current page to see full debug output

#### What Gets Logged

When debug mode is enabled, the extension logs details about:

- Function entry and exit points
- Model loading status
- Image processing steps
- Face detection results
- Memory management operations
- Performance metrics

#### Performance Impact

Be aware that enabling debug mode may slightly impact performance due to the additional logging overhead. It is recommended to disable debug mode during normal usage.

#### Disabling Debug Mode

Simply toggle the Debug Mode switch to the OFF position in the settings panel and refresh the page.

## Performance Optimizations

FaceOne implements several optimizations for smooth operation:

1. **Worker Threading**: Offloads image processing to separate threads

   - Prevents UI thread blocking
   - Parallelizes computations based on CPU cores
   - Prioritizes visible content

2. **Batch Processing**: Groups similar operations for efficiency

   - Processes images in batches of optimal size
   - Reduces overhead of model initialization
   - Prioritizes visible content first

3. **Priority Queuing**: Processes important images first

   - Visibility-based prioritization
   - Size-based optimization (smaller images first)
   - User interaction responsiveness

4. **Model Caching**: Reuses loaded models to reduce memory usage

   - Single instance of TensorFlow models
   - Incremental model loading
   - Memory-aware model management

5. **Embedding Caching**: Stores computed embeddings to avoid redundant processing

   - URL-based caching for repeated images
   - Session-based persistence
   - LRU cache implementation for memory efficiency

6. **Lazy Loading**: Loads resources only when needed

   - On-demand model loading
   - Progressive feature availability
   - Reduced startup time

7. **Memory Management**: Implements automatic cleanup of unused resources

   - Tensor disposal after processing
   - Periodic garbage collection
   - Memory threshold monitoring

8. **Throttling**: Limits processing rate to maintain browser responsiveness

   - Request animation frame synchronization
   - Idle callback utilization
   - Processing queue rate limiting

9. **Incremental Processing**: Processes images in stages as they become visible

   - Intersection Observer API integration
   - Viewport prioritization
   - Background processing of off-screen content

10. **URL Normalization**: Efficiently handles changing URL parameters
    - Extraction of essential URL components
    - Parameter filtering for consistency
    - Pattern matching for similar resources

## Advanced Techniques: Image Processing

### Image Preprocessing Pipeline

The extension implements a sophisticated image preprocessing pipeline to optimize face detection:

1. **Image Acquisition**:

   - Direct image element access via DOM
   - Proxy image creation for cross-origin resources
   - Canvas-based image data extraction

2. **Size Analysis and Adaptation**:

   - Dynamic resizing based on image dimensions
   - Aspect ratio preservation
   - Resolution optimization for model performance

3. **Orientation Detection and Correction**:

   - EXIF metadata extraction when available
   - Multi-angle attempt strategy (0°, ±45°, ±90°)
   - Rotation correction using canvas transformations

4. **Color Space Optimization**:

   - RGB channel extraction and normalization
   - Alpha channel handling for transparent images
   - Color profile adaptation

5. **Canvas Management**:

   - Efficient canvas reuse
   - Memory-optimized drawing operations
   - Web worker offloading when possible

6. **Image Enhancement**:
   - Contrast normalization
   - Brightness adaptation
   - Edge enhancement for improved detection

### Dynamic Content Handling

The extension has been engineered to handle the challenges of modern web applications:

1. **Virtual DOM Detection**:

   - Recognition of framework-specific DOM patterns
   - Adaptation to React, Angular, and Vue rendering cycles
   - Special handling for SPAs (Single Page Applications)

2. **Mutation Observation Strategy**:

   - Efficient DOM mutation tracking
   - Subtree modifications monitoring
   - Attribute change detection for dynamic styling

3. **Lazy-loaded Image Handling**:

   - Intersection Observer for detecting newly visible images
   - Event listeners for dynamic content loading
   - Scroll position tracking for timely processing

4. **Responsive Design Adaptation**:

   - Media query monitoring
   - Viewport size change detection
   - Responsive image srcset handling

5. **Cross-origin Resource Management**:
   - CORS policy handling
   - Proxy methods for cross-domain images
   - Fallback strategies for restricted content

## Deep Dive: Face Detection and Recognition

### Model Architecture Details

FaceOne leverages multiple neural network models for its core functionality:

1. **Face Detection Models**:

   - SSD MobileNet v1: Optimized for larger images and accuracy
   - TinyFaceDetector: Optimized for speed and smaller images
   - Model selection based on image size and processing requirements

2. **FaceNet Architecture**:

   - 512-dimensional embedding output
   - Input size of 160x160 pixels
   - Inception ResNet v1 backbone
   - Triplet loss function training

3. **Custom Similarity Model**:
   - Fine-tuned for web image comparison
   - Optimized for browser execution
   - Enhanced robustness against lighting and pose variations

### Feature Extraction Process

The feature extraction process involves several technical steps:

1. **Face Region Normalization**:

   - Scale normalization to fixed dimensions
   - Alignment based on facial landmarks
   - Background removal and masking

2. **Deep Feature Extraction**:

   - Convolutional layer activation extraction
   - Progressive feature map generation
   - Dimensionality reduction

3. **Embedding Generation**:

   - Non-linear transformations through deep network
   - L2 normalization for vector standardization
   - Euclidean embedding space representation

4. **Quality Assessment**:
   - Confidence score calculation
   - Blur and lighting quality estimation
   - Pose angle estimation

### Multi-face Handling

The extension implements sophisticated algorithms for handling multiple faces in a single image:

1. **Detection Clustering**:

   - Non-maximum suppression for overlapping detections
   - Intersection over Union (IoU) calculation
   - Threshold-based merging of similar detections

2. **Face Prioritization**:

   - Size-based ranking (larger faces prioritized)
   - Central position prioritization
   - Confidence score weighting

3. **Batch Processing Strategy**:

   - Efficient processing of multiple faces
   - Parallel embedding generation
   - Resource-aware batch size adjustment

4. **Result Aggregation**:
   - Multiple detection visualization
   - Individual face similarity scoring
   - Composite blurring decision logic

## Storage and Persistence

### Chrome Storage Implementation

FaceOne uses Chrome's storage APIs for data persistence:

1. **Storage Types**:

   - `chrome.storage.local`: For large data storage (blurred image URLs)
   - `chrome.storage.sync`: For user settings synchronization

2. **Data Structures**:

   - URL maps with timestamps for blurred images
   - Settings objects for user preferences
   - Processing statistics for performance monitoring

3. **Storage Optimization**:

   - URL normalization to reduce duplicate entries
   - Timestamp-based data expiration
   - Chunked storage for large datasets

4. **Error Handling**:
   - Storage limit management
   - Error recovery mechanisms
   - Data validation before storage

### BlurTracker Implementation

The `blurTracker.js` module provides a sophisticated system for managing blurred images:

1. **Data Structure**:

   - Map-based storage with URLs as keys and timestamps as values
   - Efficient O(1) lookup performance
   - Memory-optimized representation

2. **URL Normalization**:

   - Parameter filtering for dynamic URLs
   - Path-based normalization
   - Domain-specific handling (e.g., Facebook image URLs)

3. **Timestamp Management**:

   - Automatic refreshing of accessed URLs
   - Age-based cleanup (default: 30 minutes)
   - Periodic pruning to manage size

4. **Persistence Strategy**:

   - Periodic automatic saving (every 5 minutes)
   - On-demand saving when changes occur
   - Page unload saving to prevent data loss

5. **MutationObserver Integration**:
   - DOM change monitoring
   - Dynamic image blur application
   - Efficient node filtering

### Dynamic CSS Implementation

For ensuring blur persistence, the extension implements dynamic CSS injection:

1. **Rule Generation**:

   - URL-based selector creation
   - Specificity optimization
   - Platform-specific targeting

2. **Style Application**:

   - `!important` rule usage for override guarantee
   - Multiple selector strategies for resilience
   - Class and attribute-based targeting

3. **Update Mechanism**:
   - Periodic style sheet refreshing
   - On-demand rule regeneration
   - DOM-change triggered updates

## Security and Privacy Considerations

### Local Processing

FaceOne prioritizes user privacy through local processing:

1. **Client-side Execution**:

   - All face detection and recognition occurs locally in the browser
   - No image data sent to external servers
   - No communication with third-party services

2. **Sandbox Isolation**:

   - TensorFlow.js execution in isolated context
   - Content Security Policy restrictions
   - Cross-origin resource protection

3. **Permission Minimization**:
   - Limited to essential Chrome APIs
   - No unnecessary permissions requested
   - Clear separation of privilege contexts

### Data Protection

User data is protected through several mechanisms:

1. **Scoped Storage**:

   - Data accessible only by the extension itself
   - No exposure to websites or other extensions
   - Chrome's built-in storage encryption

2. **Minimal Persistence**:

   - Only essential data stored (URLs and timestamps)
   - No personal information or image content stored
   - Configurable data cleanup policies

3. **Memory Management**:
   - Immediate disposal of sensitive data after processing
   - Tensor memory clearing after model inference
   - Garbage collection optimization

### Ethical Considerations

The extension is designed with ethical use in mind:

1. **User Control**:

   - Full transparency of all processing
   - User-configurable confidence thresholds
   - Easy disabling of features

2. **Processing Indicators**:

   - Visual feedback during detection
   - Clear status messaging
   - Result explanation

3. **Performance Impact**:
   - Resource usage monitoring
   - Adaptive processing based on device capabilities
   - Background processing throttling

## Development and Contribution

### Building From Source

To build the extension from source:

1. Clone the repository
2. Install dependencies: `npm install`
3. Build the extension: `npm run build`
4. Load the unpacked extension in Chrome

### Testing

The extension includes testing infrastructure:

1. **Unit Tests**:

   - Model functionality tests
   - URL normalization tests
   - Storage implementation tests

2. **Integration Tests**:

   - End-to-end processing workflow tests
   - DOM manipulation tests
   - Chrome API interaction tests

3. **Performance Testing**:
   - Memory usage benchmarks
   - Processing time measurements
   - Storage efficiency tests

### Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Structure Guidelines

When contributing, please follow these code organization principles:

1. **Modularity**: Keep components isolated and focused
2. **Error Handling**: Implement robust error recovery
3. **Performance**: Consider resource usage and efficiency
4. **Documentation**: Comment complex algorithms and approaches
5. **Compatibility**: Ensure broad browser support

## Scientific Foundations

FaceOne is built on solid scientific principles from computer vision, deep learning, and human-computer interaction. This section explains the foundational science behind the extension's capabilities.

### Neural Network Architectures

#### Convolutional Neural Networks (CNNs)

The face detection and recognition systems are powered by Convolutional Neural Networks:

1. **Hierarchical Feature Learning**:

   - Low-level features (edges, corners) in early layers
   - Mid-level features (textures, patterns) in middle layers
   - High-level features (facial components) in deep layers
   - Final layer provides complete facial identity encoding

2. **Architectural Innovations**:

   - Inception modules for multi-scale feature extraction
   - Residual connections for gradient propagation in deep networks
   - Depth-wise separable convolutions for efficiency
   - Feature pyramid networks for scale invariance

3. **Transfer Learning Application**:
   - Pre-trained models fine-tuned for face detection
   - Domain adaptation techniques for web image variations
   - Knowledge distillation for model compression

#### Embedding Spaces and Metric Learning

The facial recognition system operates on principles of metric learning:

1. **Embedding Space Properties**:

   - 512-dimensional Euclidean space representation
   - Faces of the same person cluster together
   - Different identities form separate clusters
   - Distance metrics correlate with visual similarity

2. **Loss Function Formulation**:

   - Triplet loss minimizes distance between same identity
   - Contrastive loss maximizes distance between different identities
   - Center loss enforces tighter clusters
   - Hyperparameter tuning for optimal separation margins

3. **Mathematical Foundations**:
   - L2 normalization: $\hat{v} = \frac{v}{||v||_2}$
   - Cosine similarity: $similarity(A,B) = \frac{A \cdot B}{||A||_2 \times ||B||_2}$
   - Threshold application: $match = similarity(A,B) \geq threshold$

### Computer Vision Techniques

#### Image Processing Pipeline

The image preprocessing pipeline applies several scientific techniques:

1. **Multi-scale Processing**:

   - Scale-space theory application
   - Pyramid representations for efficiency
   - Resolution-adaptive processing thresholds

2. **Illumination Normalization**:

   - Gamma correction: $I_{out} = I_{in}^{\gamma}$
   - Histogram equalization for contrast enhancement
   - Local contrast normalization for lighting invariance

3. **Geometric Transformations**:
   - Affine transformations for pose correction
   - Homography estimation for perspective handling
   - Procrustes analysis for landmark alignment

#### Face Detection Algorithms

The face detection system combines multiple algorithmic approaches:

1. **Cascaded Detection**:

   - Multi-stage classification process
   - Early rejection of non-face regions
   - Progressive refinement of candidate regions

2. **Anchor-based Detection**:

   - Pre-defined box generation at multiple scales
   - Regression for bounding box refinement
   - Non-maximum suppression for overlapping detections

3. **Landmark Localization**:
   - Shape regression techniques
   - Heatmap-based keypoint prediction
   - 68-point facial landmark model

### Human-Computer Interaction Principles

The extension applies HCI research for optimal user experience:

1. **Visual Perception Considerations**:

   - Blur radius optimized for human visual processing
   - Color coding aligned with perceptual understanding
   - Visual indicators positioned for optimal scanpath efficiency

2. **Cognitive Load Management**:

   - Progressive disclosure of complex information
   - Minimized configuration requirements
   - Predictable system behavior

3. **Interaction Design**:
   - Direct manipulation principles
   - Immediate visual feedback
   - Undo capability for all actions

## Recent Improvements: Enhanced Blur Persistence

The latest version implements significant enhancements to the blur persistence system, addressing challenges with dynamic DOM manipulation in modern web applications.

### Technical Challenges Addressed

1. **Virtual DOM Reconciliation**:
   Modern web frameworks like React often completely recreate DOM elements during updates, destroying applied styles and blur effects. Our solution tackles this through multiple persistence mechanisms:

   ```javascript
   // Multiple persistence mechanisms
   img.style.filter = "blur(10px)"; // Inline CSS (Level 1)
   img.classList.add("blurred-image"); // Class-based (Level 2)
   img.setAttribute("data-faceone-processed", "blurred"); // Attribute-based (Level 3)
   ```

2. **URL Normalization for Dynamic Content**:
   Social media platforms often serve the same image with different URL parameters. Our advanced URL normalization algorithm addresses this:

   ```javascript
   normalizeImageUrl(url) {
     try {
       // For Facebook images, strip out changing parameters but keep essential ones
       if (url.includes('fbcdn.net') || url.includes('facebook.com')) {
         const urlObj = new URL(url);

         // Keep only essential parameters
         const essentialParams = ['stp', 'dst-jpg', 'set'];
         const searchParams = new URLSearchParams();

         for (const param of essentialParams) {
           if (urlObj.searchParams.has(param)) {
             searchParams.set(param, urlObj.searchParams.get(param));
           }
         }

         // Build normalized URL
         let normalizedUrl = urlObj.origin + urlObj.pathname;
         if (searchParams.toString()) {
           normalizedUrl += '?' + searchParams.toString();
         }

         return normalizedUrl;
       }

       return url;
     } catch (e) {
       return url; // Return original if parsing fails
     }
   }
   ```

3. **Dynamic CSS Rule Injection**:
   To overcome limitations of direct DOM manipulation, we implemented dynamic CSS rule injection:

   ```javascript
   function injectDynamicCssRules() {
     // Create selectors from stored URLs
     let cssRules = "";
     for (const [url, _] of blurTracker.blurredImages.entries()) {
       try {
         const urlObj = new URL(url);
         const pathParts = urlObj.pathname.split("/");
         const filename = pathParts[pathParts.length - 1].split(".")[0];

         if (filename && filename.length > 5) {
           cssRules += `img[src*="${filename}"] { filter: blur(10px) !important; }\n`;
         }
       } catch (e) {
         console.error("Error generating CSS for URL", url, e);
       }
     }

     // Apply the rules
     styleEl.textContent = cssRules;
   }
   ```

### Multi-layered Persistence Architecture

The enhanced blur persistence system uses a multi-layered approach:

1. **Layer 1: Direct DOM Modification**

   - Applied immediately to visible images
   - Provides immediate visual feedback
   - Most vulnerable to DOM changes

2. **Layer 2: MutationObserver-based Reapplication**

   - Monitors DOM for newly added images
   - Checks URLs against blur list
   - Reapplies blur effects as needed

   ```javascript
   setupMutationObserver() {
     const observer = new MutationObserver((mutations) => {
       for (const mutation of mutations) {
         if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
           this.processAddedNodes(mutation.addedNodes);
         }
       }
     });

     observer.observe(document.body, {
       childList: true,
       subtree: true,
       attributes: false
     });
   }
   ```

3. **Layer 3: Dynamic CSS Rules**

   - Pattern-based URL matching via CSS selectors
   - Survives complete DOM replacement
   - Updated periodically to catch new entries

4. **Layer 4: Persistent Storage**
   - Chrome storage API for cross-session persistence
   - Automatic saving on page unload
   - Periodic backup during browsing session

### Time-based Management System

The system implements sophisticated time-based data management:

1. **Timestamp-based Freshness**:

   - Each URL stores a last-accessed timestamp
   - Automatic refreshing when URLs are accessed
   - Gradual decay of unused entries

2. **Configurable Retention Policies**:

   - Default 30-minute retention for blurred URLs
   - Automatic cleanup of expired entries
   - Size-based pruning (max 10,000 entries)

3. **Prioritized Storage**:
   - Most recently used entries preserved
   - Least recently used entries removed first
   - Background cleanup to maintain performance

### Virtual DOM Compatibility

Special considerations were implemented for modern framework compatibility:

1. **React-specific Optimizations**:

   - Component re-render detection
   - Key-based element tracking
   - Efficient prop comparison

2. **Framework-agnostic Selectors**:

   - Attribute selectors resilient to class changes
   - Multiple selector strategies for redundancy
   - High-specificity CSS rules to override framework styles

3. **Performance Considerations**:
   - Throttled DOM operations
   - Batched style updates
   - Optimized selector generation

The enhanced blur persistence system ensures that once an image is marked for blurring, it remains blurred even through page navigation, scrolling, or dynamic content updates, providing a seamless user experience while maintaining performance and privacy.

---

This extension demonstrates advanced browser capabilities for machine learning and computer vision tasks while maintaining performance and security.
