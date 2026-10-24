"""Project planar map outlines using explicit poses and camera calibration."""

from typing import Tuple

import numpy as np


def _geometry_segments(geometry):
    """Yield independent XY segments, preserving multipart and ring breaks."""
    if geometry.is_empty:
        return
    if hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from _geometry_segments(part)
    elif geometry.geom_type == 'Polygon':
        yield from _geometry_segments(geometry.exterior)
        for ring in geometry.interiors:
            yield from _geometry_segments(ring)
    elif geometry.geom_type in ('LineString', 'LinearRing'):
        points = np.asarray(geometry.coords, dtype=np.float64)[:, :2]
        if len(points) > 1:
            yield np.stack((points[:-1], points[1:]), axis=1)


def _clip_camera_segments(segments, intrinsic, image_shape, near_clip):
    """Clip segments against the near plane and four image boundary planes.

    Clipping before perspective division avoids unbounded pixel coordinates
    and retains visible portions whose endpoints lie outside the image.
    """
    height, width = image_shape
    planes = np.zeros((5, 4), dtype=np.float64)
    planes[0] = (0, 0, 1, -near_clip)
    planes[1, :3] = intrinsic[0]
    planes[2, :3] = (width - 1) * intrinsic[2] - intrinsic[0]
    planes[3, :3] = intrinsic[1]
    planes[4, :3] = (height - 1) * intrinsic[2] - intrinsic[1]
    enter = np.zeros(len(segments))
    leave = np.ones(len(segments))
    visible = np.ones(len(segments), dtype=bool)
    for plane in planes:
        distances = segments @ plane[:3] + plane[3]
        start, end = distances[:, 0], distances[:, 1]
        visible &= ~((start < 0) & (end < 0))
        crossing = (start < 0) != (end < 0)
        fraction = np.zeros(len(segments))
        np.divide(start, start - end, out=fraction, where=crossing)
        entering = crossing & (start < 0)
        leaving = crossing & (end < 0)
        enter[entering] = np.maximum(enter[entering], fraction[entering])
        leave[leaving] = np.minimum(leave[leaving], fraction[leaving])
    visible &= enter <= leave
    start = segments[visible, 0]
    delta = segments[visible, 1] - start
    return np.stack((start + enter[visible, None] * delta,
                     start + leave[visible, None] * delta),
                    axis=1)


def project_map_geometries(geometries: dict,
                           local2global: np.ndarray,
                           global2cam: np.ndarray,
                           cam2img: np.ndarray,
                           image_shape: Tuple[int, int],
                           ground_z: float,
                           near_clip: float = 0.1) -> dict:
    """Return layer -> (N, 2, 2) float pixel segments for a single camera.

    Geometries use local XY meters, with a 3x3 planar ``local2global`` pose.
    They are lifted onto the horizontal global plane ``Z = ground_z``.
    The caller supplies one common ground_z for all cameras and a global2cam
    matrix evaluated at each camera's capture time. This keeps the assumed
    map elevation independent of the camera pose, including pitch and roll.

    Camera calibration must match the supplied image dimensions and pixel
    coordinates. Pinhole intrinsics and camera Z-forward coordinates are
    expected. Timestamp compensation is encoded in the supplied global2cam;
    occlusion is not handled. Polygons are rendered as outlines, including
    holes and ROI clipping edges; multipart geometries are never connected.
    """
    matrices = []
    for name, value, size in (
        ('local2global', local2global, 3),
        ('global2cam', global2cam, 4),
        ('cam2img', cam2img, 3),
    ):
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (size, size) or not np.isfinite(matrix).all():
            raise ValueError(f'{name} must be a finite {size}x{size} matrix')
        expected_row = np.zeros(size)
        expected_row[-1] = 1
        if not np.allclose(matrix[-1], expected_row):
            raise ValueError(f'{name} has an invalid homogeneous last row')
        matrices.append(matrix)
    local2global, global2cam, intrinsic = matrices
    if len(image_shape) != 2 or any(
            not isinstance(size, (int, np.integer)) or size <= 0
            for size in image_shape):
        raise ValueError('image_shape must contain positive integer H and W')
    if not np.isfinite(ground_z):
        raise ValueError('ground_z must be finite')
    if not np.isfinite(near_clip) or near_clip <= 0:
        raise ValueError('near_clip must be positive and finite')

    map2global = np.eye(4)
    map2global[:2, :2] = local2global[:2, :2]
    map2global[:2, 3] = local2global[:2, 2]
    map2global[2, 3] = ground_z
    map2cam = global2cam @ map2global
    projected = {}
    for name, layer in geometries.items():
        parts = [
            part for geometry in layer for part in _geometry_segments(geometry)
        ]
        if not parts:
            projected[name] = np.empty((0, 2, 2), dtype=np.float64)
            continue
        xy = np.concatenate(parts)
        if not np.isfinite(xy).all():
            raise ValueError(f'{name}: geometry coordinates must be finite')
        camera_segments = xy @ map2cam[:3, :2].T + map2cam[:3, 3]
        clipped = _clip_camera_segments(camera_segments, intrinsic,
                                        image_shape, near_clip)
        pixels = clipped @ intrinsic.T
        pixels = pixels[..., :2] / pixels[..., 2:3]
        # Bound floating-point roundoff at the image edges before drawing.
        projected[name] = np.clip(pixels, (0, 0),
                                  (image_shape[1] - 1, image_shape[0] - 1))
    return projected
