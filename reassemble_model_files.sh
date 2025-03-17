#!/bin/bash

# reassemble_model_files.sh
# Script to reassemble split model files into their original form

echo "===== Starting reassembly process ====="
echo "Looking for split files in split_files and split_files_temp directories..."

# Check if split_files and split_files_temp directories exist
if [ ! -d "split_files" ] && [ ! -d "split_files_temp" ]; then
    echo "❌ Error: Neither split_files nor split_files_temp directories found."
    exit 1
fi

# Find all split files (files with .aa extension, which is the first chunk)
SPLIT_FILES=$(find split_files split_files_temp -name "*.aa" 2>/dev/null | sort)

if [ -z "$SPLIT_FILES" ]; then
    echo "❌ Error: No split files found."
    exit 1
fi

echo "Found these split files:"
echo "$SPLIT_FILES"

echo ""
echo "===== Reassembling files ====="

# Track success and failure counts
SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_FILES=()

# Process each .aa file (first chunk of each split file)
for first_chunk in $SPLIT_FILES; do
    echo "$first_chunk"
    
    # Extract base path without the .aa extension
    base_path=${first_chunk%.aa}
    
    # Extract directory and filename
    dir_path=$(dirname "$first_chunk")
    filename=$(basename "$base_path")
    
    # Determine the output path based on the structure
    # The directory structure in split_files matches the structure in Live_portrait/LivePortrait/pretrained_weights
    rel_path=${dir_path#split_files/}
    rel_path=${rel_path#split_files_temp/}
    
    output_path="Live_portrait/LivePortrait/pretrained_weights/$rel_path"
    output_file="$output_path/$filename"
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_path"
    
    echo "Reassembling $output_file..."
    
    # Find all chunks for this file (aa, ab, ac, etc.) in order
    chunks=$(find "$dir_path" -name "$(basename $base_path).*" | sort)
    echo "$chunks"
    
    # Concatenate all chunks to reassemble the original file
    cat $chunks > "$output_file"
    
    # Check if the reassembly was successful
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        echo "✅ Successfully reassembled $output_file"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Failed to reassemble $output_file"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        FAILED_FILES+=("$output_file")
    fi
done

echo ""
echo "===== Reassembly process completed ====="
echo "✅ Successfully reassembled: $SUCCESS_COUNT files"
echo "❌ Failed to reassemble: $FAILURE_COUNT files"

if [ $FAILURE_COUNT -gt 0 ]; then
    echo "The following files failed to reassemble:"
    for file in "${FAILED_FILES[@]}"; do
        echo "  - $file"
    done
    echo "Please check if all chunks for these files are available."
else
    echo "All files have been reassembled successfully!"
fi

exit 0 