# Model Files Management

This repository contains large model files that exceed GitHub's file size limit (100MB). To handle these large files, we've split them into smaller chunks that can be pushed to GitHub normally without using Git LFS.

## Overview

- Model files are stored in the `Live_portrait/LivePortrait/pretrained_weights` directory
- These files have been split into 50MB chunks and stored in the `split_files` directory
- The split files are organized in the same directory structure as the original files

## For Users: How to Reassemble the Model Files

If you've cloned this repository, you need to run the reassembly script to combine the split files back into the original model files:

1. Clone the repository:

```bash
git clone https://github.com/lironfarzam/FaceOne.git
cd FaceOne
```

2. Make sure you're on the right branch:

```bash
git checkout model-files-normal
```

3. Run the reassembly script:

```bash
./reassemble_model_files.sh
```

4. The script will:

   - Show you which split files are available
   - Attempt to reassemble each model file
   - Report which files were successfully reassembled and which ones failed
   - Create the model files in their correct locations

5. After running the script, you should be able to use the model as normal.

## Missing Model Files

If you encounter missing model files after running the reassembly script, we've provided a script to download them:

```bash
# First check which model files are missing
./download_models.py --check

# Then download the missing model files
./download_models.py
```

If the download script fails to find some model files, you may need to look for them from the original sources:

1. LivePortrait models: https://github.com/TalkUHulk/liveportrait
2. InsightFace models: https://github.com/deepinsight/insightface

## For Developers: How to Add New Model Files

If you need to add new large model files to the repository:

1. Place the model file in its correct location in the `Live_portrait/LivePortrait/pretrained_weights` directory

2. Run the splitting script:

```bash
./split_and_upload.sh
```

3. The script will:
   - Find all `.pth` and `.onnx` files in the pretrained_weights directory
   - Split each file into 50MB chunks
   - Save the chunks in the corresponding location in the `split_files` directory
   - Add, commit, and push the chunks to GitHub

## Troubleshooting

If you encounter issues with the reassembly process:

1. Make sure you have all the necessary split files:

```bash
find split_files -type f | sort
```

2. If some split files are missing, you can:

   - Pull the latest changes: `git pull`
   - Run the reassembly script again: `./reassemble_model_files.sh`

3. If the model files are corrupted or invalid after reassembly:

   - Use the `check_onnx.py` script to check ONNX files: `./check_onnx.py path/to/file.onnx`
   - Download the models directly using the `download_models.py` script

4. If you're still missing files, you may need to:
   - Download the original model files from their source
   - Place them in the correct directories manually
   - Consider running the `split_and_upload.sh` script to contribute the missing files to the repository

## File Structure

```
FaceOne/
├── split_files/                          # Contains the split model files
│   ├── liveportrait/
│   │   ├── base_models/
│   │   ├── retargeting_models/
│   │   └── ...
│   └── liveportrait_animals/
│       ├── base_models/
│       ├── base_models_v1.1/
│       ├── retargeting_models/
│       └── ...
├── Live_portrait/
│   └── LivePortrait/
│       └── pretrained_weights/           # This is where the reassembled files go
│           ├── liveportrait/
│           └── liveportrait_animals/
├── split_and_upload.sh                   # Script to split and upload model files
├── reassemble_model_files.sh             # Script to reassemble the split files
├── check_onnx.py                         # Script to check ONNX model files
└── download_models.py                    # Script to download missing model files
```
