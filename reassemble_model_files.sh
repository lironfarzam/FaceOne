#!/bin/bash

# Create the output directories
mkdir -p Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models
mkdir -p Live_portrait/LivePortrait/pretrained_weights/liveportrait/retargeting_models
mkdir -p Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models
mkdir -p Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models_v1.1
mkdir -p Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/retargeting_models
mkdir -p Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l

# Track success and failure counts
success_count=0
failure_count=0

# Function to reassemble a file
reassemble_file() {
  local pattern=$1
  local output_file=$2
  
  # Check if pattern matches any files
  if ls $pattern 1> /dev/null 2>&1; then
    echo "Reassembling $output_file..."
    
    # Make sure the output directory exists
    mkdir -p "$(dirname "$output_file")"
    
    # Concatenate the split files
    cat $pattern > "$output_file"
    
    echo "✅ Successfully reassembled $output_file"
    ((success_count++))
  else
    echo "❌ Error: Split files not found for $output_file"
    echo "   Pattern searched: $pattern"
    echo "   You may need to run split_and_upload.sh first for this file."
    ((failure_count++))
  fi
}

echo "===== Starting reassembly process ====="
echo "Looking for split files in split_files directory..."

# List available split files
echo "Found these split files:"
find split_files -type f | sort

echo ""
echo "===== Reassembling files ====="

# Reassemble liveportrait base_models files
reassemble_file "split_files/liveportrait/base_models/appearance_feature_extractor.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth"
reassemble_file "split_files/liveportrait/base_models/motion_extractor.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/motion_extractor.pth"
reassemble_file "split_files/liveportrait/base_models/spade_generator.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/spade_generator.pth"
reassemble_file "split_files/liveportrait/base_models/warping_module.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/warping_module.pth"

# Reassemble liveportrait retargeting_models files
reassemble_file "split_files/liveportrait/retargeting_models/stitching_retargeting_module.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth"

# Reassemble liveportrait landmark.onnx
reassemble_file "split_files/liveportrait/landmark.onnx.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/landmark.onnx"

# Reassemble liveportrait_animals base_models files
reassemble_file "split_files/liveportrait_animals/base_models/appearance_feature_extractor.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models/appearance_feature_extractor.pth"
reassemble_file "split_files/liveportrait_animals/base_models/motion_extractor.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models/motion_extractor.pth"
reassemble_file "split_files/liveportrait_animals/base_models/spade_generator.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models/spade_generator.pth"
reassemble_file "split_files/liveportrait_animals/base_models/warping_module.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models/warping_module.pth"

# Reassemble liveportrait_animals base_models_v1.1 files
reassemble_file "split_files/liveportrait_animals/base_models_v1.1/appearance_feature_extractor.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models_v1.1/appearance_feature_extractor.pth"
reassemble_file "split_files/liveportrait_animals/base_models_v1.1/motion_extractor.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models_v1.1/motion_extractor.pth"
reassemble_file "split_files/liveportrait_animals/base_models_v1.1/spade_generator.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models_v1.1/spade_generator.pth"
reassemble_file "split_files/liveportrait_animals/base_models_v1.1/warping_module.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models_v1.1/warping_module.pth"

# Reassemble liveportrait_animals retargeting_models files
reassemble_file "split_files/liveportrait_animals/retargeting_models/stitching_retargeting_module.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/retargeting_models/stitching_retargeting_module.pth"

# Reassemble liveportrait_animals xpose.pth
reassemble_file "split_files/liveportrait_animals/xpose.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/xpose.pth"

# Reassemble insightface models
reassemble_file "split_files/insightface/models/buffalo_l/2d106det.onnx.*" "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx"
reassemble_file "split_files/insightface/models/buffalo_l/det_10g.onnx.*" "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx"

echo ""
echo "===== Reassembly process completed ====="
echo "✅ Successfully reassembled: $success_count files"
echo "❌ Failed to reassemble: $failure_count files"

if [ $failure_count -gt 0 ]; then
  echo ""
  echo "Some files could not be reassembled because the split files were not found."
  echo "You may need to run the split_and_upload.sh script or check if all files were properly uploaded."
  echo "If you're still encountering issues, you might need to:"
  echo "1. Look for missing model files at the original source"
  echo "2. Download them directly and place them in the appropriate directories"
  echo "3. Check if the split_files directory has all the necessary files"
else
  echo ""
  echo "All files have been reassembled successfully!"
fi 