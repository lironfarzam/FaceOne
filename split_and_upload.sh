#!/bin/bash

# Create a directory for the split files
mkdir -p split_files

# Track progress
total_files=0
processed_files=0

# Count the total number of files to process
echo "Counting files to process..."
total_files=$(find Live_portrait/LivePortrait/pretrained_weights -type f -name "*.pth" -o -name "*.onnx" | wc -l)
echo "Found $total_files files to process"

# Function to split and upload a file
split_and_upload() {
  local file=$1
  local base_name=$(basename "$file")
  local dir_name=$(dirname "$file")
  local relative_dir=${dir_name#"Live_portrait/LivePortrait/pretrained_weights/"}
  
  ((processed_files++))
  echo "[$processed_files/$total_files] Processing $file..."
  
  # Create the output directory
  mkdir -p "split_files/$relative_dir"
  
  # Get file size
  local file_size=$(du -h "$file" | cut -f1)
  echo "File size: $file_size"
  
  # Split the file into 50MB chunks
  echo "Splitting file into 50MB chunks..."
  split -b 50m "$file" "split_files/$relative_dir/${base_name}."
  
  # Count the number of chunks
  local chunk_count=$(ls "split_files/$relative_dir/${base_name}."* | wc -l)
  echo "Created $chunk_count chunks"
  
  # Add the split files to git
  echo "Adding split files to git..."
  git add "split_files/$relative_dir"
  
  # Commit the split files
  echo "Committing split files..."
  git commit -m "Add split files for $relative_dir/$base_name ($file_size, $chunk_count chunks)"
  
  # Push the commit
  echo "Pushing commit to GitHub..."
  git push
  
  echo "Completed processing $file"
  echo "--------------------------------------------"
}

# Process each file one by one
find Live_portrait/LivePortrait/pretrained_weights -type f -name "*.pth" -o -name "*.onnx" | sort | while read file; do
  split_and_upload "$file"
done

echo "All files have been split and uploaded successfully!"
echo "Total files processed: $processed_files out of $total_files" 