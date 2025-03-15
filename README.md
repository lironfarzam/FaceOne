# FaceOne: Protective Face Recognition System

<div align="center">

![FaceOne Logo](https://via.placeholder.com/500x150?text=FaceOne+Logo)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

**Protecting vulnerable individuals from unwanted digital encounters through advanced facial recognition**

[Problem](#problem-description) • [Solution](#our-solution) • [Components](#system-components) • [Project Structure](#project-structure) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [About](#about-the-project)

</div>

## 🌟 Problem Description

Recently, a woman shared her story of being sexually assaulted by someone she knew. Despite blocking her attacker on social media, she continued to encounter his images through shared content or untagged photos. Each unexpected encounter retraumatized her, leaving her feeling unsafe even in digital spaces.

This situation underscores a broader problem: existing social media and browser tools are inadequate for shielding users from harmful content. Blocking features and generic image-replacement extensions are either insufficient or too broad to address users' specific needs.

There is a clear need for a more effective solution that uses advanced facial recognition technology to automatically detect and replace images of specific individuals. Such a tool would empower users to navigate the digital world safely, free from the fear of encountering specific and unwanted faces.

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
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │     │                     │     │                     │     │                     │
│  download_images.py │──▶ │  face_processing.py │──▶ │   download_lfw.py   │──▶ │   create_model.py   │──▶ │  Chrome Extension   │
│                     │     │                     │     │                     │     │                     │     │                     │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘     └─────────────────────┘     └─────────────────────┘
       Step 1                      Step 2                      Step 3                      Step 4                      Step 5
   Collect Images             Process & Extract          Download Negative            Train Custom              Deploy Protection
                                   Faces                     Examples                   Face Model                   Solution
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

3. **Negative Examples Collection** (download_lfw.py):

   - Downloads the Labeled Faces in the Wild (LFW) dataset
   - Processes over 13,000 diverse face images
   - Prepares negative examples to improve model discrimination
   - Organizes images for use in model training

4. **Model Creation** (create_model.py):

   - Extracts face embeddings using DeepFace
   - Creates a Siamese neural network for face verification
   - Trains the model on positive and negative face pairs
   - Exports the model in formats compatible with browser execution

5. **Protection Deployment** (Chrome Extension):
   - Loads the trained model in the browser
   - Scans web pages for images containing faces
   - Compares detected faces against the target's embeddings
   - Applies blur or replacement to matching faces

### Automated Execution

The `main.py` script automates the entire pipeline, allowing you to run all components in sequence:

```bash
# Run the complete pipeline
python main.py

# Or run with specific steps skipped
python main.py --skip-lfw  # Skip LFW dataset download

# Or run individual components as needed
python Facebook_profile_handling/download_images.py
python Facebook_profile_handling/face_processing.py
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

# Or run individual components as needed:
# Step 1: Download Facebook photos
python Facebook_profile_handling/download_images.py

# Step 2: Process the downloaded faces
python Facebook_profile_handling/face_processing.py

# Step 3: Download negative examples
python create_face_model/download_lfw.py

# Step 4: Create and train the face model
python create_face_model/create_model.py

# Step 5: Load the extension in Chrome
# (Follow Chrome Extension Setup steps above)
```

## 📖 Usage

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
