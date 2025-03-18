# FaceOne: Protective Face Recognition System

<div align="center">

![FaceOne Logo](https://via.placeholder.com/500x150?text=FaceOne+Logo)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

**Protecting vulnerable individuals from unwanted digital encounters through advanced facial recognition**

[Problem](#problem-description) • [Solution](#our-solution) • [Components](#system-components) • [Quick Start](#quick-start) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [About](#about-the-project)

</div>

## 🌟 Problem Description

Recently, a woman shared her story of being sexually assaulted by someone she knew. Despite blocking her attacker on social media, she continued to encounter his images through shared content or untagged photos. Each unexpected encounter retraumatized her, leaving her feeling unsafe even in digital spaces.

This situation underscores a broader problem: existing social media and browser tools are inadequate for shielding users from harmful content. Blocking features and generic image-replacement extensions are either insufficient or too broad to address users' specific needs.

There is a clear need for a more effective solution that uses advanced facial recognition technology to automatically detect and replace images of specific individuals. Such a tool would empower users to navigate the digital world safely, free from the fear of encountering specific and unwanted faces.

## 🚀 Quick Start

Getting started with FaceOne is now easier than ever:

```bash
# 1. Clone the repository
git clone https://github.com/lironfarzam/FaceOne.git
cd FaceOne

# 2. Run the main script (automatically handles dependencies and model setup)
python main.py
```

That's it! The main script will:

- Check for required dependencies and install them if needed
- Automatically download or assemble required model files
- Guide you through the entire pipeline

## 💡 Our Solution

FaceOne is a specialized Chrome extension designed to help users avoid unwanted encounters with specific individuals online by replacing their images in web content. The extension uses advanced facial recognition technology powered by neural networks to identify the target individual's face across a wide range of images and contexts.

With minimal input from the user (such as a link to the individual's profile or a small set of photos), FaceOne automatically scans and processes HTML content on web pages, replacing identified faces with positive or neutral images selected by the user or from a default library.

<div align="center">

![FaceOne Architecture](https://via.placeholder.com/800x400?text=FaceOne+Architecture+Diagram)
_FaceOne's system architecture showing key components and data flow_

</div>

## ✨ Key Features

### 🛡️ Protective Face Recognition

- **Targeted Protection**: Specifically designed to shield users from seeing images of individuals who have caused them trauma
- **Minimal Input Required**: Functions with as few as 5-10 reference images
- **High Accuracy**: Advanced neural networks ensure reliable detection across different angles, lighting conditions, and contexts
- **Real-time Processing**: Scans and processes web content as you browse

### 🔒 Privacy-Focused Design

- **Client-side Processing**: All face detection and recognition happens locally in your browser
- **No Data Collection**: Your sensitive information never leaves your device
- **Full User Control**: You decide which faces to detect and how to replace them

### ⚙️ Customizable Experience

- **Adjustable Sensitivity**: Fine-tune detection confidence thresholds to your needs
- **Replacement Options**: Choose to blur, replace, or hide detected images
- **Visual Indicators**: Optional highlighting of detected faces for verification
- **Live Portrait Generation**: Create animated versions of static face images for additional training and visualization

<div align="center">

![Demo](https://via.placeholder.com/800x400?text=FaceOne+Demo+GIF)
_FaceOne in action: Detecting and blurring a face in real-time_

</div>

## 🧩 System Components

FaceOne consists of four main components, which work together in a pipeline to provide complete face protection:

### 1. Facebook Profile Handler 📱

The first step in building a custom face model:

- Connects to Facebook using your credentials (requires manual login)
- Downloads photos from specified profiles automatically
- Collects images for both targets (faces to detect) and diverse training data
- Handles various photo formats and quality issues automatically

### 2. Face Processing Pipeline 🔍

Processes the collected images to prepare them for model training:

- Detects and extracts faces from the downloaded images
- Enhances image quality for better recognition accuracy
- Creates face embeddings using DeepFace's Facenet512 model
- Clusters similar faces for more efficient training
- Implements parallel processing with unique temporary files to prevent race conditions

### 3. Face Model Creator 🧠

Creates a personalized face recognition model:

- Generates high-dimensional (512D) face embeddings from input images
- Trains on minimal data (5-10 images) with high accuracy
- Uses Siamese network architecture for effective face verification
- Exports models in formats compatible with the Chrome extension

### 4. Chrome Extension 🌐

The user-facing component that provides real-time face detection and replacement:

- Detects faces in web images using FaceAPI.js and FaceNet
- Compares detected faces against known embeddings
- Applies customizable blur or replacement effects to matched faces
- Operates entirely client-side for maximum privacy and security

## 📁 Project Structure

<div align="center">

```
FaceOne/
├── Facebook_profile_handling/    # Facebook image collection
│   ├── download_images.py        # Downloads images from Facebook profiles
│   ├── face_processing.py        # Processes and analyzes downloaded faces
│   └── utils.py                  # Helper functions for profile handling
│
├── create_face_model/            # Face model training
│   ├── create_model.py           # Creates Siamese network for face recognition
│   ├── model_utils.py            # Utilities for model creation and training
│   └── data_generator.py         # Generates training data pairs
│
├── Live_portrait/                # Live portrait generation
│   ├── live_portrait_generator.py # Main script for creating animated portraits
│   ├── LivePortrait/             # Deep learning model for face animation
│   └── utils.py                  # Utilities for portrait generation
│
├── chrome_extensions/            # Chrome extension for face detection/blurring
│   ├── manifest.json             # Extension configuration
│   ├── popup.html                # User interface
│   ├── js/                       # JavaScript functionality
│   │   ├── content.js            # Main content script for web page processing
│   │   ├── sandbox.js            # Isolated ML processing environment
│   │   └── blurTracker.js        # Tracks blurred images across pages
│   ├── css/                      # Styling for the extension
│   ├── lib/                      # Third-party libraries (face-api.js, TensorFlow.js)
│   └── models/                   # Pre-trained and custom face models
│
├── docs/                         # Documentation
│   ├── chrome_extensions.md      # Chrome extension documentation
│   ├── create_face_model.md      # Face model creation guide
│   ├── facebook_profile_handling.md  # Profile handling documentation
│   └── live_portrait.md          # Live portrait generation guide
│
├── main.py                       # Main script to run complete pipeline
├── requirements.txt              # Python dependencies
├── config.json                   # Configuration settings
└── README.md                     # Project documentation
```

_FaceOne Project Structure_

</div>

## 🔄 Execution Flow

The FaceOne system follows a specific workflow to create and deploy a face protection solution:

<div align="center">

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │     │                     │     │                     │     │                     │     │                     │
│  download_images.py │──▶ │  face_processing.py │──▶ │live_portrait_gen.py │──▶ │   download_lfw.py   │──▶ │   create_model.py   │──▶ │  Chrome Extension   │
│                     │     │                     │     │                     │     │                     │     │                     │     │                     │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘     └─────────────────────┘     └─────────────────────┘     └─────────────────────┘
       Step 1                      Step 2                      Step 3                      Step 4                      Step 5                      Step 6
   Collect Images             Process & Extract           Generate Live              Download Negative            Train Custom              Deploy Protection
                                   Faces                    Portraits                    Examples                   Face Model                   Solution
```

_FaceOne execution flow from data collection to deployment_

</div>

### Processing Steps:

1. **Image Collection** (download_images.py):

   - Collects face images from specified Facebook profiles
   - Downloads photos to a local directory for processing
   - Requires Facebook login credentials (via interactive browser session)

2. **Face Processing** (face_processing.py):

   - Detects and extracts faces from downloaded images
   - Filters faces based on quality and size
   - Clusters similar faces to identify the target individual
   - Prepares face data for model training

3. **Live Portrait Generation** (live_portrait_generator.py):

   - Creates animated versions of the processed face images
   - Applies facial movements to static photos for more realistic representation
   - Generates additional training data through varied expressions and angles
   - Creates a more comprehensive face dataset for improved recognition

4. **Negative Examples Collection** (download_lfw.py):

   - Downloads the Labeled Faces in the Wild (LFW) dataset
   - Processes over 13,000 diverse face images
   - Prepares negative examples to improve model discrimination
   - Organizes images for use in model training

5. **Model Creation** (create_model.py):

   - Extracts face embeddings using DeepFace
   - Creates a Siamese neural network for face verification
   - Trains the model on positive and negative face pairs
   - Exports the model in formats compatible with browser execution

6. **Protection Deployment** (Chrome Extension):
   - Loads the trained model in the browser
   - Scans web pages for images containing faces
   - Compares detected faces against the target's embeddings
   - Applies blur or replacement to matching faces

### Automated Execution

The `main.py` script automates the entire pipeline, allowing you to run all components in sequence:

```bash
# Run the complete pipeline
python main.py

# Skip specific steps
python main.py --skip-portraits  # Skip Live Portrait generation
python main.py --skip-lfw        # Skip LFW dataset download

# Model file handling options
python main.py --force-model-assembly  # Force reassembly of model files
python main.py --skip-model-assembly   # Skip model file assembly
python main.py --setup-model-files     # Just set up model files and exit

# Cleanup options
python main.py --skip-cleanup           # Skip cleanup of temporary files
python main.py --preserve-facebook      # Preserve Facebook images during cleanup
python main.py --preserve-lfw           # Preserve LFW dataset during cleanup
python main.py --clean-portrait-dirs    # Clean LivePortrait directories
python main.py --clean-imgs             # Clean imgs directory

# Run individual components as needed
python Facebook_profile_handling/download_images.py
python Facebook_profile_handling/face_processing.py
python Live_portrait/live_portrait_generator.py
python create_face_model/download_lfw.py
python create_face_model/create_model.py
```

## 📂 Folder Explanations

### Facebook_profile_handling/

This folder contains the components for collecting face data from Facebook:

- **Purpose**: Automates the collection of face images for model training
- **Key Features**:
  - Selenium-based Facebook navigation and image collection
  - Intelligent image filtering to find high-quality face images
  - Support for downloading photos from various Facebook sections
- **Why It's Important**: Getting high-quality training data is essential for model accuracy

### create_face_model/

This folder contains the face recognition model training components:

- **Purpose**: Creates a custom face recognition model for a specific individual
- **Key Features**:
  - Face embedding generation using DeepFace
  - Siamese network architecture for face comparison
  - Advanced training techniques like data augmentation and embedding caching
- **Why It's Important**: The custom model ensures high accuracy for the specific target face

### Live_portrait/

This folder contains components for generating animated face portraits:

- **Purpose**: Creates dynamic, animated versions of static face images
- **Key Features**:
  - Deep learning-based face animation
  - Transfers facial expressions from driving videos to target face images
  - Extracts image frames for additional training data
  - GPU-accelerated processing for faster generation
- **Why It's Important**: Provides more varied facial expressions and angles for improved model training and creates engaging visual outputs

### chrome_extensions/

This folder contains the browser extension that provides the actual protection:

- **Purpose**: Detects and blurs/replaces faces in web browsing
- **Key Features**:
  - Client-side face detection and recognition
  - TensorFlow.js model execution in the browser
  - Real-time scanning of web content
  - Customizable face replacement options
- **Why It's Important**: This is the user-facing component that delivers the protection

### docs/

This folder contains comprehensive documentation for all system components:

- **Purpose**: Provides detailed guides for understanding and using the system
- **Key Files**:
  - chrome_extensions.md: Guide to using the Chrome extension
  - create_face_model.md: Details on creating custom face models
  - facebook_profile_handling.md: Guide to collecting Facebook photos
  - live_portrait.md: Information on the portrait animation feature
- **Why It's Important**: Documentation ensures users can effectively use the system

## 🚀 Installation

### Prerequisites

- Chrome browser (version 88 or higher)
- For model training (optional):
  - Python 3.7+
  - TensorFlow 2.x
  - CUDA-compatible GPU (recommended)

### Chrome Extension Setup

```bash
# 1. Download the extension
git clone https://github.com/lironfarzam/FaceOne.git
cd FaceOne

# 2. Open Chrome and navigate to chrome://extensions
# 3. Enable Developer Mode (toggle in top-right)
# 4. Click "Load unpacked" and select the chrome_extensions folder
# 5. The FaceOne icon should appear in your toolbar
```

### Model File Management

FaceOne uses large model files for face recognition and portrait generation. To facilitate distribution and version control, these files are split into smaller chunks:

#### Automatic Model Setup

The main script automatically handles model file setup:

```bash
# Run with automatic model setup (recommended for most users)
python main.py
```

#### Manual Model Setup

You can also set up model files manually:

```bash
# Set up model files using the setup script
./setup_model_files.sh

# Or reassemble model files directly
./reassemble_model_files.sh
```

These scripts:

- Check for required model files
- Assemble models from split chunks if needed
- Download missing models if necessary
- Verify successful installation

For detailed information about model file management, including how to add new model files or troubleshoot issues, please refer to [README_MODEL_FILES.md](README_MODEL_FILES.md).

### Complete System Setup

For users who want to create their own custom face models:

```bash
# 1. Install required dependencies
pip install -r requirements.txt

# 2. Configure the system
# Edit config.json with your settings:
# - Facebook profile URLs to download from
# - Directories for storing images and models
# - Training parameters

# 3. Run the full pipeline
python main.py
```

## 📖 Usage

### Command-Line Arguments

FaceOne's main script accepts several command-line arguments to customize its behavior:

| Argument                  | Description                                                   |
| ------------------------- | ------------------------------------------------------------- |
| `--skip-download`         | Skip downloading images from Facebook                         |
| `--skip-processing`       | Skip face processing step                                     |
| `--skip-portraits`        | Skip live portrait generation                                 |
| `--skip-lfw`              | Skip LFW dataset download                                     |
| `--skip-model`            | Skip model creation step                                      |
| `--chrome-only`           | Only prepare Chrome extension files                           |
| `--force-model-assembly`  | Force reassembly of model files                               |
| `--skip-model-assembly`   | Skip model file assembly                                      |
| `--setup-model-files`     | Run model files setup and exit                                |
| `--skip-dependency-check` | Skip checking dependencies                                    |
| `--skip-cleanup`          | Skip cleanup of temporary files                               |
| `--preserve-facebook`     | Preserve Facebook images during cleanup (default: True)       |
| `--preserve-lfw`          | Preserve LFW dataset during cleanup (default: True)           |
| `--clean-portrait-dirs`   | Clean LivePortrait directories during cleanup (default: True) |
| `--clean-imgs`            | Clean imgs directory during cleanup (default: True)           |

### Model File Scripts

FaceOne includes two scripts to help manage the large model files required for face recognition and portrait generation:

#### setup_model_files.sh

This is the primary script for setting up model files:

```bash
./setup_model_files.sh
```

- **Purpose**: Complete setup of all required model files
- **Functions**:
  - Creates necessary directory structure
  - Checks for existing model files
  - Verifies split files if models need to be assembled
  - Calls reassemble_model_files.sh when necessary
  - Provides comprehensive output and error handling
- **When to use**: When setting up FaceOne for the first time or when model files are missing

#### reassemble_model_files.sh

This script focuses specifically on reassembling split model files:

```bash
./reassemble_model_files.sh
```

- **Purpose**: Reassemble split model files into their original form
- **Functions**:
  - Locates all split file chunks (_.aa, _.ab, etc.)
  - Determines correct output path for each model file
  - Concatenates chunks to recreate original files
  - Reports success/failure for each reassembled file
- **When to use**: When you have split files but the assembled model files are missing

These scripts work together to ensure that the large model files (often several hundred MB each) can be properly version-controlled using split files, while still being usable by the application.

### Automated Cleanup

FaceOne includes an automated cleanup system that runs at the end of the pipeline to remove temporary files and directories:

- **Temporary Files**: Removes process-specific temporary files created during face processing (`temp_safe_detect_*.jpg`, `temp_safe_represent_*.jpg`)
- **Temporary Directories**: Cleans the `split_files_temp` and `temp` directories
- **Optional Cleaning**: Depending on your preferences, can also clean:
  - Facebook output folder (unless `--preserve-facebook` is set)
  - LivePortrait source and output directories (controlled by `--clean-portrait-dirs`)
  - Image directory (controlled by `--clean-imgs`)

This helps maintain a clean workspace after processing and reduces disk space usage. The cleanup process preserves important data like Facebook images and the LFW dataset by default, but can be configured to remove these if needed.

To skip the cleanup entirely:

```bash
python main.py --skip-cleanup
```

To preserve Facebook images but clean everything else:

```bash
python main.py --preserve-facebook
```

To clean everything including Facebook images:

```bash
python main.py --preserve-facebook=False
```

### Basic Usage

1. Click the FaceOne icon in your Chrome toolbar
2. For first-time setup:
   - Provide input for the person you want to avoid seeing (profile link or upload photos)
   - Grant necessary permissions for the extension to modify webpage content
3. Browse the web normally - FaceOne will automatically detect and replace images

### Advanced Settings

Access advanced settings through the extension popup:

- **Detection Sensitivity**: Adjust the confidence threshold (higher = fewer false positives)
- **Replacement Style**: Choose between blur, overlay, or removal
- **Processing Mode**: Select between automatic (all images) or manual (on-demand) processing
- **Visual Feedback**: Toggle detection indicators for verification purposes

<div align="center">

![Chrome Extension](https://via.placeholder.com/400x300?text=Chrome+Extension+Screenshot)
_FaceOne Chrome extension interface_

</div>

## 📚 Documentation

For detailed documentation of each component, please refer to:

- [Chrome Extension Documentation](docs/chrome_extensions.md)
- [Face Model Creation Guide](docs/create_face_model.md)
- [Facebook Profile Handling Documentation](docs/facebook_profile_handling.md)
- [Live Portrait Documentation](docs/live_portrait.md)
- [Model Files Management Guide](README_MODEL_FILES.md) - Detailed instructions for handling large model files, including:
  - How to reassemble split model files after cloning the repository
  - Troubleshooting missing model files
  - Adding new model files to the repository (for developers)
  - Using model file scripts like `reassemble_model_files.sh` and `download_models.py`

## 🧪 Technical Details

### Face Detection and Recognition Pipeline

FaceOne implements a sophisticated multi-stage approach:

1. **Image Detection**: Identifies images on web pages as they load
2. **Face Detection**: Locates faces within images using FaceAPI.js
3. **Face Alignment**: Normalizes detected faces for consistent processing
4. **Embedding Generation**: Creates 512-dimensional face embeddings using FaceNet
5. **Similarity Comparison**: Compares embeddings against known faces using cosine similarity
6. **Action Application**: Applies the user's chosen action (blur/replace) to matching images

### Performance Optimizations

- **Worker Threading**: Processes images in background threads to maintain browsing performance
- **Batch Processing**: Groups similar operations for efficiency
- **Priority Queuing**: Processes visible content before off-screen images
- **Caching**: Stores results to prevent redundant processing

### Race Condition Prevention

FaceOne implements several strategies to prevent race conditions during parallel processing:

- **Process-Specific Temporary Files**: Uses unique filenames based on process ID to prevent conflicts
  - Each process creates files with format `temp_safe_detect_{process_id}.jpg` and `temp_safe_represent_{process_id}.jpg`
  - This prevents processes from overwriting each other's temporary files during face detection and embedding
- **Robust Error Handling**: Gracefully recovers from errors with detailed logging
- **Automatic Cleanup**: Ensures no leftover temporary files remain after processing completes

These improvements enable reliable parallel processing of faces, significantly improving performance on multi-core systems without introducing synchronization errors.

### Privacy and Security Considerations

- **Sandbox Isolation**: TensorFlow.js execution in isolated context
- **Content Security Policy**: Strict CSP to prevent external access
- **Local Storage**: All sensitive data remains on the user's device
- **Permission Minimization**: Only essential Chrome APIs are used

## 📜 About the Project

### Project Origin

FaceOne was conceived and developed by **Liron Farzam** after hearing the story of a sexual assault survivor who continued to be traumatized by unexpected encounters with her abuser's images online. Recognizing the inadequacy of existing tools to address this specific need, Liron developed FaceOne as a targeted solution to help vulnerable individuals regain control of their digital experiences.

### Development Approach

The development of FaceOne followed a user-centered approach:

1. **Problem Identification**: Understanding the specific challenges faced by trauma survivors
2. **Solution Design**: Creating a system that provides protection while respecting privacy
3. **Technical Implementation**: Leveraging advanced ML techniques for accurate face recognition
4. **User Testing**: Refining the system based on feedback from potential users
5. **Continuous Improvement**: Ongoing development to enhance accuracy and performance

### Ethical Considerations

FaceOne was built with strong ethical principles in mind:

- **Trauma-Informed Design**: Created with sensitivity to the needs of trauma survivors
- **User Autonomy**: Giving users control over their digital experiences
- **Privacy Protection**: Ensuring all processing happens locally
- **Transparency**: Clear documentation about how the system works

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [FaceAPI.js](https://github.com/justadudewhohacks/face-api.js)
- [MediaPipe](https://mediapipe.dev/)
- [DeepFace](https://github.com/serengil/deepface)
- Special thanks to all contributors and testers who helped shape this project

---

<div align="center">
<p>Designed and developed by <strong>Liron Farzam</strong></p>
<p>Made with ❤️ by the FaceOne Team</p>
</div>
