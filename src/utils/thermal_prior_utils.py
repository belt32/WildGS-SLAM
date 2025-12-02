"""
thermal_prior_utils.py

This module provides utilities for processing thermal images and calibrating them to the RGB coordinate system.
"""

import numpy as np
import cv2
import os
import datetime


def detect_heat_sources(thermal_image, threshold=50):
    """
    Detect heat sources in a thermal image.

    Args:
        thermal_image (np.ndarray): Input thermal image (grayscale).
        threshold (float): Temperature threshold for detecting heat sources.

    Returns:
        np.ndarray: Binary mask of detected heat sources.
    """
    # Normalize the thermal image to the range [0, 255]
    normalized_image = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Threshold the image to detect heat sources
    _, heat_mask = cv2.threshold(normalized_image, threshold, 255, cv2.THRESH_BINARY)
    
    return heat_mask


def get_calibrated_thermal_mask(thermal_image, T_rgb_thermal, K_thermal, rgb_shape, threshold=50):
    """
    Generate a calibrated binary mask of heat sources in the RGB image coordinate system.

    Args:
        thermal_image (np.ndarray): Input thermal image (grayscale).
        T_rgb_thermal (np.ndarray): 4x4 transformation matrix from thermal to RGB coordinate system.
        K_thermal (np.ndarray): 3x3 intrinsic matrix of the thermal camera.
        rgb_shape (tuple): Shape of the RGB image (height, width).
        threshold (float): Temperature threshold for detecting heat sources.

    Returns:
        np.ndarray: Calibrated binary mask in the RGB image coordinate system.
    """
    # Step 1: Detect heat sources in the thermal image
    heat_mask = detect_heat_sources(thermal_image, threshold)

    # Step 2: Get pixel coordinates of detected heat sources
    heat_coords = np.column_stack(np.where(heat_mask > 0))  # Nx2 array of (row, col)

    if heat_coords.size == 0:
        # If no heat sources are detected, return an empty mask
        return np.zeros(rgb_shape, dtype=np.uint8)

    # Step 3: Convert pixel coordinates to homogeneous coordinates
    heat_coords_h = np.hstack((heat_coords[:, ::-1], np.ones((heat_coords.shape[0], 1))))  # Nx3 (x, y, 1)

    # Step 4: Project thermal coordinates to the RGB coordinate system
    K_thermal_inv = np.linalg.inv(K_thermal)
    thermal_camera_coords = (K_thermal_inv @ heat_coords_h.T).T  # Nx3
    thermal_camera_coords_h = np.hstack((thermal_camera_coords, np.ones((thermal_camera_coords.shape[0], 1))))  # Nx4

    rgb_camera_coords_h = (T_rgb_thermal @ thermal_camera_coords_h.T).T  # Nx4
    rgb_camera_coords = rgb_camera_coords_h[:, :3] / rgb_camera_coords_h[:, 3, None]  # Nx3 (x, y, z)

    # Step 5: Project back to the RGB image plane
    rgb_coords_h = rgb_camera_coords[:, :2] / rgb_camera_coords[:, 2, None]  # Nx2 (x/z, y/z)
    rgb_coords = np.round(rgb_coords_h).astype(int)  # Convert to integer pixel coordinates

    # Step 6: Create the calibrated binary mask in the RGB image space
    calibrated_mask = np.zeros(rgb_shape, dtype=np.uint8)
    valid_coords = (0 <= rgb_coords[:, 0]) & (rgb_coords[:, 0] < rgb_shape[1]) & \
                   (0 <= rgb_coords[:, 1]) & (rgb_coords[:, 1] < rgb_shape[0])
    rgb_coords = rgb_coords[valid_coords]

    calibrated_mask[rgb_coords[:, 1], rgb_coords[:, 0]] = 255

    # Debug: Save the calibrated mask as a PNG file
    debug_dir = './debug_output'
    os.makedirs(debug_dir, exist_ok=True)

    # Add color to the mask for visualization (255 -> green)
    color_mask = np.zeros((*calibrated_mask.shape, 3), dtype=np.uint8)
    color_mask[calibrated_mask > 0] = [0, 255, 0]  # Green color for mask

    # Generate timestamped filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_path = os.path.join(debug_dir, f'calib_mask_{timestamp}.png')

    # Save the colorized mask
    cv2.imwrite(debug_path, color_mask)

    # Print the debug path
    print(f"Calibrated mask saved to: {debug_path}")

    return calibrated_mask