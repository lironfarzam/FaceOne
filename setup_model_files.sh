#!/bin/bash

# setup_model_files.sh
# A script to set up model files for FaceOne
# This script handles checking, splitting, and reassembling model files

set -e  # Exit on any error

# Text formatting
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
RESET="\033[0m"

# Print a header
print_header() {
    echo -e "\n${BOLD}${MAGENTA}=============================================================================${RESET}"
    echo -e "${BOLD}${MAGENTA} $1 ${RESET}"
    echo -e "${BOLD}${MAGENTA}=============================================================================${RESET}\n"
}

# Print a step
print_step() {
    echo -e "\n${BOLD}${BLUE}[Step $1/$2] $3${RESET}\n"
}

# Print success message
print_success() {
    echo -e "${GREEN}✅ $1${RESET}"
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}⚠️ $1${RESET}"
}

# Print error message
print_error() {
    echo -e "${RED}❌ $1${RESET}"
}

# Check if model files exist and have the right size
check_model_files() {
    print_header "Checking Model Files"
    
    # Define model file paths to check
    model_paths=(
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/landmark.onnx"
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth"
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/motion_extractor.pth"
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/spade_generator.pth"
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models/warping_module.pth"
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth"
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/2d106det.onnx"
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l/det_10g.onnx"
    )
    
    missing_files=()
    for path in "${model_paths[@]}"; do
        if [ ! -f "$path" ] || [ $(stat -f%z "$path" 2>/dev/null || echo 0) -lt 10000 ]; then
            missing_files+=("$path")
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        print_warning "Found ${#missing_files[@]} missing or invalid model files."
        for path in "${missing_files[@]}"; do
            print_warning "  - $path"
        done
        return 1
    else
        print_success "All ${#model_paths[@]} model files exist and have proper size."
        return 0
    fi
}

# Check if split files exist
check_split_files() {
    print_header "Checking Split Files"
    
    # Check if either directory exists
    if [ ! -d "split_files" ] && [ ! -d "split_files_temp" ]; then
        print_error "Neither split_files nor split_files_temp directories found."
        return 1
    fi
    
    # Count files in split_files directory
    split_files_count=0
    if [ -d "split_files" ]; then
        split_files_count=$(find split_files -type f | wc -l | xargs)
    fi
    
    # Count files in split_files_temp directory
    temp_files_count=0
    if [ -d "split_files_temp" ]; then
        temp_files_count=$(find split_files_temp -type f | wc -l | xargs)
    fi
    
    total_files=$((split_files_count + temp_files_count))
    
    if [ $total_files -eq 0 ]; then
        print_error "No split files found in either split_files or split_files_temp directories."
        return 1
    fi
    
    print_success "Found $split_files_count files in split_files directory."
    print_success "Found $temp_files_count files in split_files_temp directory."
    print_success "Total of $total_files split files found."
    return 0
}

# Create directories needed for model files
create_directories() {
    print_header "Creating Directories"
    
    # Create model directory structure
    model_dirs=(
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/base_models"
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait/retargeting_models"
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models"
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/base_models_v1.1"
        "Live_portrait/LivePortrait/pretrained_weights/liveportrait_animals/retargeting_models"
        "Live_portrait/LivePortrait/pretrained_weights/insightface/models/buffalo_l"
    )
    
    for dir_path in "${model_dirs[@]}"; do
        mkdir -p "$dir_path"
        print_success "Created directory: $dir_path"
    done
    
    # Create directories for split files if they don't exist
    mkdir -p "split_files"
    mkdir -p "split_files_temp"
    
    print_success "All directories created successfully."
}

# Main script execution
print_header "FaceOne Model Files Setup"
echo "This script will help you set up the necessary model files for FaceOne."

# Create directories
create_directories

# Check if model files already exist
if check_model_files; then
    print_success "Model files are already set up correctly."
    
    # Ask if user wants to force reassembly
    read -p "Do you want to force model reassembly anyway? (y/N): " force_reassembly
    if [[ "$force_reassembly" =~ ^[Yy]$ ]]; then
        if [ -f "reassemble_model_files.sh" ]; then
            print_step "1" "1" "Reassembling model files"
            bash ./reassemble_model_files.sh
        else
            print_error "reassemble_model_files.sh script not found."
            exit 1
        fi
    else
        print_success "Setup complete. No reassembly needed."
    fi
else
    # Check if split files exist
    if check_split_files; then
        if [ -f "reassemble_model_files.sh" ]; then
            print_step "1" "1" "Reassembling model files from split chunks"
            bash ./reassemble_model_files.sh
            
            # Verify successful assembly
            if check_model_files; then
                print_success "Model files were successfully reassembled."
            else
                print_error "Some model files are still missing after reassembly."
                print_warning "You may need to download them or check your split files."
            fi
        else
            print_error "reassemble_model_files.sh script not found."
            print_warning "You need this script to reassemble the model files."
            exit 1
        fi
    else
        print_error "No model files or split files found."
        print_warning "You need to:"
        print_warning "1. Clone the repository with the --recursive flag, or"
        print_warning "2. Download the split files from the original source, or"
        print_warning "3. Run the split_and_upload.sh script if you have the original files."
        exit 1
    fi
fi

print_header "Setup Complete"
echo "You should now be ready to use FaceOne with all required model files."
echo "Run the main script with: python main.py" 