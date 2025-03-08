# Live Portrait Generator

## Friendly Overview

Welcome to the Live Portrait Generator! This tool brings your still photos to life by animating them using AI technology. It's like magic - you provide a photo of a face and a driving video, and the system creates a realistic animation where the person in your photo mimics the movements from the video.

### What You Can Do

- **Animate Photos**: Turn any portrait photo into a talking, moving video
- **Use Any Driving Video**: The movements can come from any video with a face
- **Extract Frames**: Automatically save frames from generated videos for further use
- **Customize Settings**: Control various aspects of the animation through the config file
- **Parallel Processing**: Generate multiple videos simultaneously for faster results
- **GPU Acceleration**: Utilize NVIDIA, AMD, or Apple Silicon GPUs for faster processing

### Quick Start

1. Place your portrait photos in the `NN_LiveProtrait/Source_Images/` folder
2. Place your driving videos in the `NN_LiveProtrait/Source_Video/` folder
3. Run the script: `python live_portrait_generator.py`
4. Find your animated videos in the `NN_LiveProtrait/Output/` folder

### GPU Support

The Live Portrait Generator now includes comprehensive GPU support for faster processing:

- **Automatic Detection**: Automatically detects and uses available GPU hardware
- **Multiple GPU Types**: Supports NVIDIA (CUDA), AMD (ROCm), and Apple Silicon (MPS) GPUs
- **Half Precision**: Uses FP16 (half precision) for faster GPU processing when available
- **Memory Management**: Configurable GPU memory usage to prevent out-of-memory errors
- **Fallback Options**: Gracefully falls back to CPU when GPU is unavailable or disabled

To control GPU usage:

```
# Force CPU usage (even if GPU is available)
python live_portrait_generator.py --cpu

# Force GPU usage (will still fall back to CPU if no GPU is available)
python live_portrait_generator.py --gpu
```

Configure GPU behavior in your config.json:

```json
{
  "force_cpu": false,           // Set to true to force CPU usage
  "enable_mps": true,           // Enable Apple Silicon GPU support
  "use_half_precision": true,   // Use FP16 for faster GPU processing
  "cuda_memory_fraction": 0.8,  // Limit GPU memory usage to 80%
  ...
}
```

### Parallel Processing

The Live Portrait Generator now supports parallel processing to significantly speed up video generation:

- **Multiple Workers**: Automatically uses all available CPU cores for maximum performance
- **Configurable**: Set the number of parallel workers in the config file or command line
- **Progress Tracking**: Shows real-time progress for all parallel operations
- **Error Handling**: Continues processing even if some combinations fail

To specify the number of workers manually:

```
python live_portrait_generator.py --workers 8
```

Or set it in your config.json:

```json
{
  "num_of_workers": 8,
  ...
}
```

### Progress Tracking

The Live Portrait Generator uses Rich's progress tracking to provide beautiful, informative progress bars during processing. These progress bars show:

- Current operation description
- Percentage complete
- Visual progress indicator
- Estimated time remaining
- Processing speed

This makes it easy to monitor long-running operations like video generation and frame extraction.

#### Rich's `track` vs `tqdm`

This project uses Rich's `track` function instead of the more common `tqdm` for progress tracking. Key differences include:

- **Visual Appeal**: Rich provides more visually appealing progress bars with better colors and formatting
- **Information Density**: Rich progress bars show more detailed information about the ongoing process
- **Terminal Integration**: Rich handles terminal resizing and different terminal types better
- **Consistency**: All progress indicators in the project now have the same look and feel

The `track` function is used as a direct replacement for `tqdm`, making it easy to integrate into existing code:

```python
# Before: Using tqdm
for item in tqdm(items, desc="Processing items"):
    process_item(item)

# After: Using Rich's track
for item in track(items, description="Processing items"):
    process_item(item)
```

## Command Line Arguments

The Live Portrait Generator supports the following command line arguments:

```
python live_portrait_generator.py [--config CONFIG] [--workers WORKERS] [--cpu] [--gpu]

Options:
  --config CONFIG    Path to config file
  --workers WORKERS  Number of parallel workers
  --cpu              Force CPU usage
  --gpu              Force GPU usage
```

For backward compatibility, you can also use positional arguments:

```
python live_portrait_generator.py config.json 8
```

## Technical Details

### How It Works

The Live Portrait Generator uses a sophisticated AI model to animate still portrait images. The process involves several key steps:

1. **Face Analysis**: The system detects and analyzes facial landmarks in both the source image and driving video
2. **Feature Extraction**: Deep neural networks extract appearance features from the source image and motion features from the driving video
3. **Motion Transfer**: The system transfers the motion from the driving video to the source image while preserving the identity of the person in the source image
4. **Image Generation**: A specialized generator creates realistic frames that combine the appearance of the source with the motion of the driving video
5. **Post-processing**: The generated frames are assembled into a video, and audio from the driving video is added

### Key Components

#### 1. LivePortrait Model Architecture

The LivePortrait model consists of several neural network components:

- **Appearance Feature Extractor**: Extracts identity-preserving features from the source image
- **Motion Extractor**: Captures motion patterns from the driving video
- **Warping Module**: Applies motion to the source features
- **SPADE Generator**: Generates realistic output frames
- **Stitching & Retargeting Module**: Controls specific facial regions like eyes and lips

#### 2. Process Flow

```
Source Image → Appearance Feature Extraction →
                                              ↘
                                                Motion Transfer → Image Generation → Video Assembly
                                              ↗
Driving Video → Motion Feature Extraction →
```

### Implementation Details

The implementation uses several key technologies:

- **PyTorch**: For deep learning model implementation
- **OpenCV**: For image and video processing
- **FFmpeg**: For video encoding and audio processing

### Configuration Options

The system can be configured through the `config.json` file with these key parameters:

- `input_source_folder`: Directory containing source portrait images
- `input_video_folder`: Directory containing driving videos
- `output_video_folder`: Directory for saving output videos
- `use_cpu`: Whether to use CPU instead of GPU (useful for compatibility)
- `frame_interval`: Interval for extracting frames from videos
- `animation_region`: Which facial regions to animate ("all", "face", etc.)
- `audio_priority`: Source of audio in the output video
- `do_pasteback`: Whether to paste the animated face back onto the original image
- `do_crop`: Whether to crop the face before processing

## Scientific Explanation

### Neural Network Architecture

The LivePortrait system employs a specialized neural network architecture designed for one-shot face reenactment. Unlike traditional computer graphics approaches that require extensive 3D modeling, this approach uses deep learning to directly map between source and target domains.

#### Appearance Feature Extraction

The appearance feature extractor is a convolutional neural network (CNN) that creates a latent representation of the source image. This network is designed to capture identity-preserving features while discarding pose-specific information. The network architecture includes:

- Multiple convolutional layers with instance normalization
- Residual connections to preserve fine details
- Downsampling layers to create a compact feature representation

The extracted features encode facial structure, skin tone, and other identity-specific attributes that should be preserved during animation.

#### Motion Extraction and Transfer

The motion extractor analyzes the driving video to capture facial movements. It uses:

- Facial landmark detection to track key points
- Optical flow estimation to capture motion between frames
- A specialized keypoint detector that identifies motion-relevant facial features

The motion information is then transferred to the source image through a warping process that respects the facial geometry of the source image.

#### Image Generation with SPADE

The Spatially-Adaptive Denormalization (SPADE) generator creates the output frames. This architecture:

- Uses the warped features as input
- Applies spatially-adaptive normalization to preserve spatial information
- Employs multiple upsampling layers to generate high-resolution output
- Includes skip connections to maintain fine details

#### Stitching and Retargeting Control

A key innovation in LivePortrait is the stitching and retargeting module that provides fine-grained control over specific facial regions:

- **Stitching**: Allows selective animation of specific facial regions while keeping others static
- **Retargeting**: Adapts the intensity of motion for different facial parts (e.g., eyes, lips)

This enables more natural animations by respecting the physical constraints of facial movements.

### Mathematical Formulation

The core of the animation process can be formulated as:

$$I_{animated} = G(F_s(I_{source}), W(F_m(I_{driving}), F_s(I_{source})))$$

Where:

- $G$ is the generator function
- $F_s$ is the source feature extractor
- $F_m$ is the motion feature extractor
- $W$ is the warping function
- $I_{source}$ is the source image
- $I_{driving}$ is the driving video frame

### Performance Considerations

The system balances quality and performance through:

- Efficient network architectures with separable convolutions
- Batch processing for parallel computation
- Optional half-precision (FP16) computation
- CPU fallback for systems without compatible GPUs

## Troubleshooting

### Common Issues

1. **Missing Pretrained Weights**: Ensure all model weights are downloaded to the `LivePortrait/pretrained_weights/` directory
2. **CUDA Compatibility**: If using GPU, ensure your CUDA version is compatible with the PyTorch version
3. **Memory Issues**: For high-resolution videos, consider reducing the resolution or using CPU mode
4. **Module Import Errors**: Ensure all dependencies are installed and the directory structure is maintained

### Solutions

- Run with `--flag-force-cpu` for compatibility with systems lacking GPU support
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon Macs
- Check the LivePortrait documentation for detailed troubleshooting steps

## References

1. LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control
2. First Order Motion Model for Image Animation
3. SPADE: Spatially-Adaptive Normalization for Image Synthesis
4. Face Analysis using Deep Learning Approaches
