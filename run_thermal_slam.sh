#!/bin/bash

# Configuration file for thermal SLAM
CONFIG_FILE="configs/Custom/thermal_slam.yaml"

# Dataset path (edit this placeholder to your dataset path)
DATASET_PATH="/path/to/your/thermal_rgb_dataset"

# Output directory for results
OUTPUT_DIR="output/thermal_slam_results"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the SLAM system with thermal prior and debug mode enabled
python3 run.py "$CONFIG_FILE" -d "$DATASET_PATH" -o "$OUTPUT_DIR" --use_thermal_prior True --debug_mode True

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Thermal SLAM completed successfully."
    echo "Results saved in: $OUTPUT_DIR"
    echo "Check debug images and logs in the output directory for more details."
else
    echo "[ERROR] Thermal SLAM failed. Please check the configuration and dataset paths."
    echo "Debug logs and outputs are available in: $OUTPUT_DIR"
fi