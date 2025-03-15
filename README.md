# FaceOne: Advanced Face Recognition & Privacy System

<div align="center">

![FaceOne Logo](https://via.placeholder.com/500x150?text=FaceOne+Logo)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

**Powerful face recognition, privacy protection, and face animation in one comprehensive platform**

[Features](#key-features) • [Components](#system-components) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [About](#about-the-project) • [Contributing](#contributing)

</div>

## 🌟 Introduction

FaceOne is a comprehensive face recognition and privacy protection system designed to give users control over their digital presence. Built using advanced machine learning techniques including convolutional neural networks (CNNs), 3D face modeling, and Siamese networks, FaceOne offers powerful tools to identify, protect, and animate facial images.

Created by **Liron Farzam**, this project emerged from the need to provide a solution for privacy protection in the digital age, particularly for vulnerable individuals. The system allows users to identify and manage how their likeness (or others') appears online, giving them unprecedented control over their digital presence.

### 🎯 Primary Goals

- **Privacy Protection**: Identify and replace unwanted images of specific faces online
- **Face Recognition**: Achieve accurate recognition with minimal input images
- **Content Control**: Give users power over how their likeness appears on the web
- **Animation**: Transform still photos into realistic animated portraits

<div align="center">

![FaceOne Architecture](https://via.placeholder.com/800x400?text=FaceOne+Architecture+Diagram)
_FaceOne's system architecture showing key components and data flow_

</div>

## ✨ Key Features

### 🛡️ Privacy Protection

- Block or blur specific faces across websites
- Protect victims of harassment from seeing images of their abusers
- Real-time scanning of web content using the Chrome extension
- Ethical approach to privacy preservation and consent management

### 🔍 Advanced Face Recognition

- High-accuracy recognition with as few as 5-10 input images
- Robust against lighting variations, angles, and facial expressions
- 3D face modeling for enhanced recognition accuracy
- DeepFace integration for state-of-the-art embedding generation

### 🎭 Face Animation

- Transform still photos into realistic talking videos
- Use any driving video to animate portrait photos
- Customize settings for optimal results
- GPU-accelerated rendering for fast processing

### 📊 Social Media Integration

- Automatic Facebook profile image collection
- Bulk face detection and clustering
- Profile analysis and visualization
- Privacy-focused API interactions

<div align="center">

![Demo](https://via.placeholder.com/800x400?text=FaceOne+Demo+GIF)
_FaceOne in action: Detecting and blurring a face in real-time_

</div>

## 🧩 System Components

FaceOne consists of four main components:

### 1. Chrome Extension 🌐

A powerful browser extension that provides real-time face detection and blur capabilities:

- Detects faces in web images using FaceAPI.js and FaceNet
- Compares detected faces against known embeddings
- Applies customizable blur effects to matched faces
- Operates entirely client-side for maximum privacy

### 2. Face Model Creator 🧠

Create personalized face recognition models using Siamese neural networks:

- Generate high-dimensional (512D) face embeddings
- Train on minimal data (5-10 images) with high accuracy
- Calculate optimal similarity thresholds
- Export models for use with the Chrome extension

### 3. Live Portrait Generator 🎬

Transform still photos into animated videos:

- Animate any portrait photo with realistic movements
- Use any driving video as a motion source
- Customize settings for optimal results
- GPU-accelerated processing

### 4. Facebook Profile Handler 📱

Extract and process facial data from Facebook profiles:

- Authenticate and navigate Facebook programmatically
- Download profile and album photos
- Process images through face detection pipeline
- Cluster and analyze face data

## 🚀 Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Chrome browser (for the extension)
- CUDA-compatible GPU (recommended for face model training)

### Dependencies

FaceOne relies on several powerful libraries to provide its functionality:

```
# Core ML and Computer Vision
tensorflow>=2.4.0
tensorflowjs>=3.0.0
opencv-python>=4.5.0
numpy>=1.19.0
deepface>=0.0.75
mediapipe>=0.8.0

# Web and Automation
selenium>=3.0.0
requests>=2.25.0

# Utilities and Visualization
tqdm>=4.0.0
matplotlib>=3.0.0
scikit-learn>=0.0.0
scipy>=1.0.0
pillow>=8.0.0
rich>=10.0.0
```

You can install all dependencies using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/FaceOne.git
cd FaceOne

# Install dependencies
pip install -r requirements.txt

# Set up the Chrome extension
# 1. Open Chrome and navigate to chrome://extensions
# 2. Enable Developer Mode
# 3. Click "Load unpacked" and select the chrome_extensions folder
```

### Component-Specific Setup

<details>
<summary><b>Face Model Creator Setup</b></summary>

```bash
# Prepare your face images
mkdir -p data/face_images/source

# Place 5-10 clear face images in the source directory
# Then run the model creation script
python scripts/create_face_model.py
```

</details>

<details>
<summary><b>Live Portrait Generator Setup</b></summary>

```bash
# Install additional dependencies
pip install -r NN_LivePortrait/requirements.txt

# Prepare your data
mkdir -p NN_LivePortrait/Source_Images
mkdir -p NN_LivePortrait/Source_Video

# Place portrait images and driving videos in their respective folders
# Then run the generator
python NN_LivePortrait/live_portrait_generator.py
```

</details>

<details>
<summary><b>Facebook Profile Handler Setup</b></summary>

```bash
# Install additional dependencies
pip install -r facebook_handler/requirements.txt

# Configure credentials (edit config file)
cp facebook_handler/config.example.json facebook_handler/config.json

# Run the profile handler
python facebook_handler/profile_downloader.py
```

</details>

## 📖 Usage

### Chrome Extension

1. Click the FaceOne icon in your Chrome toolbar
2. Toggle between Face Detection and Blur modes
3. Adjust the confidence threshold as needed
4. Browse the web with automatic face detection/blurring active

<div align="center">

![Chrome Extension](https://via.placeholder.com/400x300?text=Chrome+Extension+Screenshot)
_FaceOne Chrome extension interface_

</div>

### Face Model Creation

```python
# Example: Create a face model from a set of input images
from faceone import FaceModelCreator

# Initialize the creator
creator = FaceModelCreator()

# Add positive samples (the person to recognize)
creator.add_positive_samples("path/to/positive/images/")

# Add negative samples (other people)
creator.add_negative_samples("path/to/negative/images/")

# Train the model
model = creator.train(epochs=50)

# Save the model and embeddings
creator.save("models/my_face_model")
```

### Live Portrait Animation

```python
# Example: Animate a portrait using a driving video
from faceone import LivePortrait

# Initialize the generator
generator = LivePortrait()

# Set source image and driving video
generator.set_source_image("path/to/portrait.jpg")
generator.set_driving_video("path/to/driving_video.mp4")

# Generate the animation
output_path = generator.generate()
print(f"Animation saved to: {output_path}")
```

<div align="center">

![Live Portrait Example](https://via.placeholder.com/800x250?text=Before+%E2%86%92+After+Animation)
_Left: Original portrait photo. Right: Animated portrait_

</div>

### Facebook Profile Handling

```python
# Example: Download and process images from a Facebook profile
from faceone import FacebookHandler

# Initialize with credentials
handler = FacebookHandler(config_path="config.json")

# Connect and navigate to profile
handler.login()
handler.navigate_to_profile("profile_url")

# Download images
image_paths = handler.download_images(limit=100)

# Process downloaded images
face_clusters = handler.process_faces(image_paths)
```

## 📚 Documentation

For detailed documentation of each component, please refer to:

- [Chrome Extension Documentation](docs/chrome_extensions.md)
- [Face Model Creation Guide](docs/create_face_model.md)
- [Live Portrait Generator Documentation](docs/live_portrait.md)
- [Facebook Profile Handling Documentation](docs/facebook_profile_handling.md)

## 🧪 Technical Details

### Face Detection and Recognition

FaceOne uses a multi-stage approach for face processing:

1. **Face Detection**: Locates faces in images using FaceAPI.js
2. **Landmark Detection**: Identifies 68 facial landmarks for precise face alignment
3. **Face Alignment**: Normalizes face orientation for consistent embedding generation
4. **Embedding Generation**: Creates 512-dimensional face embeddings using FaceNet
5. **Similarity Comparison**: Computes cosine similarity between embeddings

```python
# Example: Core face recognition process
def recognize_face(image, known_embeddings, threshold=0.6):
    # Detect face in image
    face = face_detector.detect(image)

    # Generate embedding
    embedding = embedding_generator.generate(face)

    # Compare to known embeddings
    matches = []
    for name, known_embedding in known_embeddings.items():
        similarity = cosine_similarity(embedding, known_embedding)
        if similarity > threshold:
            matches.append((name, similarity))

    return sorted(matches, key=lambda x: x[1], reverse=True)
```

### Siamese Network Architecture

The face recognition model uses a Siamese network architecture:

```python
def make_siamese_model():
    # Input layers for two face images
    input_image = Input(name="input_img", shape=(100, 100, 3))
    validation_image = Input(name="validation_img", shape=(100, 100, 3))

    # Shared embedding network
    embedding = make_embedding()

    # Get embeddings for both images
    input_embedding = embedding(input_image)
    validation_embedding = embedding(validation_image)

    # Calculate L1 distance between embeddings
    distances = L1Dist()(input_embedding, validation_embedding)

    # Final classification layer
    classifier = Dense(1, activation="sigmoid")(distances)

    return Model(inputs=[input_image, validation_image],
                outputs=classifier,
                name="SiameseNetwork")
```

## 📜 About the Project

### Project Origin

FaceOne was conceived and developed by **Liron Farzam** in response to the growing need for digital privacy tools in an increasingly connected world. The project began as a research initiative exploring how facial recognition technology could be used to protect individuals rather than expose them.

### Development Journey

The development of FaceOne followed a methodical approach:

1. **Research Phase**: Studied existing facial recognition systems and privacy concerns
2. **Prototype Development**: Built initial models for face detection and embedding generation
3. **Chrome Extension Creation**: Developed browser-based solution for real-time protection
4. **Integration Phase**: Combined various components into a cohesive system
5. **Optimization**: Enhanced performance and accuracy across all modules

### Ethical Considerations

FaceOne was built with strong ethical principles in mind:

- **Privacy-First**: All processing happens locally where possible
- **User Control**: Individuals maintain control over their facial data
- **Transparency**: Clear documentation about how the system works
- **Protection Focus**: Designed to shield vulnerable individuals

### Current Status and Future Directions

The project is actively maintained and being developed with several planned enhancements:

- Cross-browser extension support
- Mobile application development
- Enhanced animation capabilities
- Improved social media integration
- Advanced 3D modeling techniques

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

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
