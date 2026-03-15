import os
import cv2
import time

import imageio.v2 as imageio
import numpy as np

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Create a `RealSenseCameraConfig` specifying your camera’s serial number and enabling depth.
config = RealSenseCameraConfig(
    serial_number_or_name="207122078979",
    fps=30,
    width=1280,
    height=720,
    color_mode=ColorMode.RGB,
    use_depth=True,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect a `RealSenseCamera` with warm-up read (default).
camera = RealSenseCamera(config)
camera.connect()

# Capture a color frame via `read()` and a depth map via `read_depth()`.
try:
    color_frame = camera.read()
    depth_map = camera.read_depth()
    print("Color frame shape:", color_frame.shape)
    print("Depth map shape:", depth_map.shape)

    print(depth_map.dtype)
    print("min depth:", depth_map.min())
    print("max depth:", depth_map.max())
    print("mean depth:", depth_map.mean())

    valid = depth_map[depth_map > 0]

    min_d = np.percentile(valid, 20)
    max_d = np.percentile(valid, 80)

    depth_clipped = np.clip(depth_map, min_d, max_d)

    depth_vis = ((depth_clipped - min_d) / (max_d - min_d) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    imageio.imwrite("depth_color.png", depth_color)

finally:
    camera.disconnect()
