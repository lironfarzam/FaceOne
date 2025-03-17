#!/bin/bash

# Create the output directories
mkdir -p Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models
mkdir -p Live_portrait/LivePortrait/pretrained_weights/liveportrait/retargeting_models
mkdir -p Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models
mkdir -p Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models_v1.1
mkdir -p Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/retargeting_models
mkdir -p Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l

# Function to reassemble a file
reassemble_file() {
  local pattern=$1
  local output_file=$2
  
  echo "Reassembling $output_file..."
  
  # Make sure the output directory exists
  mkdir -p "$(dirname "$output_file")"
  
  # Concatenate the split files
  cat $pattern > "$output_file"
  
  echo "Successfully reassembled $output_file"
}

# Reassemble liveportrait base_models files
reassemble_file "split_files/liveportrait/base_models/appearance_feature_extractor.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth"
reassemble_file "split_files/liveportrait/base_models/motion_extractor.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/motion_extractor.pth"
reassemble_file "split_files/liveportrait/base_models/spade_generator.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/spade_generator.pth"
reassemble_file "split_files/liveportrait/base_models/warping_module.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/warping_module.pth"

# Reassemble liveportrait retargeting_models files
reassemble_file "split_files/liveportrait/retargeting_models/stitching_retargeting_module.pth.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth"

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

# Reassemble liveportrait landmark.onnx
reassemble_file "split_files/liveportrait/landmark.onnx.*" "Live_portrait/LivePortrait/pretrained_weights/liveportrait/landmark.onnx"

# Reassemble insightface models
reassemble_file "split_files/insightface/models/buffalo_l/2d106det.onnx.*" "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx"
reassemble_file "split_files/insightface/models/buffalo_l/det_10g.onnx.*" "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx"

echo "All files have been reassembled successfully!" 