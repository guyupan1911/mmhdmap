"""Crop nuScenes basemaps for visualization outside the dataset pipeline."""

import json
from typing import Tuple

import numpy as np
from PIL import Image


def crop_nusc_basemap(basemap_path: str, map_path: str, ego2global: np.ndarray,
                      roi_size: Tuple[float, float]) -> np.ndarray:
    """Return a forward-up RGB uint8 crop at the basemap's native resolution.

    Args:
        basemap_path (str): Path to a trusted nuScenes basemap PNG.
        map_path (str): Corresponding map JSON containing ``canvas_edge``
            (global map width and height in meters).
        ego2global (np.ndarray): 4x4 ego pose. Only XY translation and yaw
            are used; ego X points forward and ego Y points left.
        roi_size (tuple[float, float]): Lateral width and longitudinal length
            in meters, rounded to whole source pixels. The ROI is symmetric
            about the ego origin. Output dimensions follow the source map's
            pixels per meter; no resizing is performed.

    The PNG spans [0, map_width] x [0, map_height], with its top row at
    maximum global Y, matching the nuScenes SDK bitmap rendering convention.
    Out-of-map pixels are filled with gray (32). Only the local crop is
    converted to RGB. Rotation and subpixel translation use bilinear sampling.
    PNG decoding still loads the source image into memory;
    this function is intended for occasional visualization, not training.
    """
    roi = np.asarray(roi_size, dtype=np.float64)
    if roi.shape != (2, ) or not np.all(np.isfinite(roi) & (roi > 0)):
        raise ValueError('roi_size must contain two positive finite lengths')
    pose = np.asarray(ego2global, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('ego2global must be a finite 4x4 matrix')
    if np.linalg.norm(pose[:2, 0]) < 1e-8:
        raise ValueError('ego forward axis must have a nonzero XY projection')

    with open(map_path, encoding='utf-8') as file:
        canvas = np.asarray(json.load(file)['canvas_edge'], dtype=np.float64)
    if canvas.shape != (2, ) or not np.all(np.isfinite(canvas) & (canvas > 0)):
        raise ValueError(
            'canvas_edge must contain two positive finite lengths')

    # Official basemaps exceed Pillow's generic decompression-bomb threshold.
    # Relax it only while opening this explicitly supplied, trusted map file.
    max_pixels = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        source = Image.open(basemap_path)
    finally:
        Image.MAX_IMAGE_PIXELS = max_pixels

    with source:
        if source.mode not in ('L', 'RGB', 'RGBA'):
            raise ValueError(f'Unsupported basemap image mode: {source.mode}')
        yaw = np.arctan2(pose[1, 0], pose[0, 0])
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        scale_x, scale_y = np.asarray(source.size) / canvas
        if not np.isclose(scale_x, scale_y, rtol=1e-6, atol=0):
            raise ValueError(
                'Basemap must have equal X and Y pixels per meter')
        width, height = (max(1, round(size * scale_x)) for size in roi)
        # Quantize the ROI extent instead of changing the native pixel scale.
        lateral, longitudinal = width / scale_x, height / scale_x
        step_left = step_forward = 1 / scale_x

        # Pillow uses pixel-edge coordinates and samples output pixel centers.
        # Local X decreases downwards; local Y decreases towards the right.
        # Map output pixels -> local meters -> global meters -> source pixels.
        affine = (
            sin_yaw * step_left * scale_x,
            -cos_yaw * step_forward * scale_x,
            (pose[0, 3] + cos_yaw * longitudinal / 2 - sin_yaw * lateral / 2) *
            scale_x,
            cos_yaw * step_left * scale_y,
            sin_yaw * step_forward * scale_y,
            (canvas[1] - pose[1, 3] - sin_yaw * longitudinal / 2 -
             cos_yaw * lateral / 2) * scale_y,
        )
        fill = {'L': 32, 'RGB': (32, 32, 32), 'RGBA': (32, 32, 32, 255)}
        crop = source.transform((width, height),
                                Image.Transform.AFFINE,
                                affine,
                                resample=Image.Resampling.BILINEAR,
                                fillcolor=fill[source.mode])
        return np.array(crop.convert('RGB'))
