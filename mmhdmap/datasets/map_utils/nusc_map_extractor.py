"""SDK-backed nuScenes map extraction shared by inspection and transforms."""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.map_expansion.map_api import NuScenesMap
from PIL import Image


class _DirectoryBitMap(BitMap):
    """Adapt the SDK bitmap loader to an explicitly configured PNG directory."""

    def __init__(self, data_root: str, map_name: str, directory: str) -> None:
        self.directory = Path(directory)
        super().__init__(data_root, map_name, 'basemap')

    def load_bitmap(self) -> np.ndarray:
        with Image.open(self.directory / f'{self.map_name}.png') as image:
            return np.array(image)


class NuScenesMapExtractor:
    """Extract basemaps and raw annotations for one nuScenes map region.

    SDK map and bitmap objects are loaded separately on first use and cached
    for this instance. Geometry-only extraction never opens a PNG. Create one
    extractor per region as needed; call ``clear_cache`` to release resources.

    Coordinates use an ego-centered horizontal frame: X forward, Y left,
    with XY translation and yaw taken from ``ego2global``. ROI sizes are
    (lateral width, longitudinal length), in meters. Returned ``local2global``
    matrices are 3x3 planar homogeneous transforms. These are raw SDK layers,
    not task-specific vector or segmentation GT.

    Args:
        data_root (str): nuScenes root with ``maps/expansion/<map_name>.json``.
        map_name (str): Region name, such as ``singapore-onenorth``.
        basemap_dir (str, optional): Directory containing trusted basemap PNGs.
            If omitted, use the SDK's ``data_root/maps/basemap`` convention.
    """

    DEFAULT_LAYERS = ('road_divider', 'lane_divider', 'ped_crossing')
    # get_map_geom does not implement traffic_light or lane_connector extraction.
    SUPPORTED_LAYERS = ('drivable_area', 'road_segment', 'road_block', 'lane',
                        'ped_crossing', 'walkway', 'stop_line', 'carpark_area',
                        'road_divider', 'lane_divider')

    def __init__(self,
                 data_root: str,
                 map_name: str,
                 basemap_dir: Optional[str] = None) -> None:
        self.data_root = str(data_root)
        self.map_name = map_name
        self.basemap_dir = basemap_dir
        self._map = None
        self._bitmap = None

    def clear_cache(self) -> None:
        """Release the SDK objects, including the full-size decoded bitmap."""
        self._bitmap = None
        self._map = None

    def _get_map(self) -> NuScenesMap:
        if self._map is None:
            self._map = NuScenesMap(self.data_root, self.map_name)
        return self._map

    def _get_bitmap(self) -> BitMap:
        if self._bitmap is None:
            # Official maps exceed Pillow's generic image-size limit. Restore
            # the setting even when loading the trusted dataset image fails.
            max_pixels = Image.MAX_IMAGE_PIXELS
            try:
                Image.MAX_IMAGE_PIXELS = None
                if self.basemap_dir is None:
                    self._bitmap = BitMap(self.data_root, self.map_name,
                                          'basemap')
                else:
                    self._bitmap = _DirectoryBitMap(self.data_root,
                                                    self.map_name,
                                                    self.basemap_dir)
            finally:
                Image.MAX_IMAGE_PIXELS = max_pixels
        return self._bitmap

    @staticmethod
    def _get_patch_info(ego2global: np.ndarray,
                        roi_size: Tuple[float, float]) -> dict:
        """Build shared planar pose and ROI information without reading files."""
        roi = np.asarray(roi_size, dtype=np.float64)
        if roi.shape != (2, ) or not np.all(np.isfinite(roi) & (roi > 0)):
            raise ValueError(
                'roi_size must contain two positive finite lengths')
        pose = np.asarray(ego2global, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('ego2global must be a finite 4x4 matrix')
        if np.linalg.norm(pose[:2, 0]) < 1e-8:
            raise ValueError(
                'ego forward axis must have a nonzero XY projection')
        yaw = np.arctan2(pose[1, 0], pose[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        local2global = np.array([[c, -s, pose[0, 3]], [s, c, pose[1, 3]],
                                 [0, 0, 1]])
        return dict(
            roi_size=tuple(roi),
            local2global=local2global,
            yaw_degrees=float(np.degrees(yaw)))

    def extract_basemap(self, ego2global: np.ndarray,
                        roi_size: Tuple[float, float]) -> dict:
        """Return an RGB crop, resolution (m/px), ROI and planar pose.

        The crop is forward-up and retains native pixel density. ROI extents
        are rounded to whole pixels; pass the returned ``roi_size`` to
        ``extract_geometries`` for exact alignment. Rotation and subpixel
        translation use bilinear sampling. Out-of-map pixels are gray (32).
        Full PNG decoding still consumes memory; only the local window is
        converted to a Pillow image and then RGB.
        """
        patch = self._get_patch_info(ego2global, roi_size)
        canvas = np.asarray(self._get_map().canvas_edge, dtype=np.float64)
        if canvas.shape != (
                2, ) or not np.all(np.isfinite(canvas) & (canvas > 0)):
            raise ValueError(
                'canvas_edge must contain two positive finite lengths')
        source = self._get_bitmap().image
        if (source.dtype != np.uint8 or source.ndim not in (2, 3)
                or (source.ndim == 3 and source.shape[2] not in (3, 4))):
            raise ValueError(
                'Basemap must be a uint8 grayscale, RGB or RGBA image')
        scale_x, scale_y = np.array(source.shape[1::-1]) / canvas
        if not np.isclose(scale_x, scale_y, rtol=1e-6, atol=0):
            raise ValueError(
                'Basemap must have equal X and Y pixels per meter')
        width, height = (max(1, round(size * scale_x)) for size in roi_size)
        lateral, longitudinal = width / scale_x, height / scale_x
        patch['roi_size'] = (lateral, longitudinal)
        patch['resolution'] = 1 / scale_x
        transform = patch['local2global']
        c, s = transform[0, 0], transform[1, 0]
        tx, ty = transform[:2, 2]
        # Inverse sampling: output pixels -> local meters -> source pixels.
        # Pillow uses pixel-edge coordinates, sampling each pixel center.
        affine = np.array([
            [s, -c, (tx + c * longitudinal / 2 - s * lateral / 2) * scale_x],
            [
                c * scale_y / scale_x, s * scale_y / scale_x,
                (canvas[1] - ty - s * longitudinal / 2 - c * lateral / 2) *
                scale_y
            ],
        ])
        corners = affine @ np.array([[0, width, 0, width],
                                     [0, 0, height, height], [1, 1, 1, 1]])
        # Slice a small source window before RGB conversion; include an
        # interpolation margin and clip against the full bitmap dimensions.
        left, top = np.maximum(
            np.floor(corners.min(axis=1)).astype(int) - 2, 0)
        right, bottom = np.minimum(
            np.ceil(corners.max(axis=1)).astype(int) + 2, source.shape[1::-1])
        if left >= right or top >= bottom:
            patch['image'] = np.full((height, width, 3), 32, dtype=np.uint8)
            return patch
        affine[:, 2] -= (left, top)
        with Image.fromarray(source[top:bottom, left:right]) as window:
            fill = {'L': 32, 'RGB': (32, 32, 32), 'RGBA': (32, 32, 32, 255)}
            crop = window.transform((width, height),
                                    Image.Transform.AFFINE,
                                    affine.ravel(),
                                    resample=Image.Resampling.BILINEAR,
                                    fillcolor=fill[window.mode])
            patch['image'] = np.array(crop.convert('RGB'))
        return patch

    def extract_geometries(self,
                           ego2global: np.ndarray,
                           roi_size: Tuple[float, float],
                           layers: Sequence[str] = DEFAULT_LAYERS) -> dict:
        """Return raw SDK layer geometries in local meters plus patch metadata.

        Polygon outlines may contain ROI clipping edges. These are display
        geometries, not derived road-boundary GT. Separate multipart objects
        and polygon holes are preserved. No fixed-point sampling is applied.
        """
        patch = self._get_patch_info(ego2global, roi_size)
        layers = list(dict.fromkeys(layers))
        invalid = set(layers) - set(self.SUPPORTED_LAYERS)
        if invalid:
            raise ValueError(
                f'Unsupported map geometry layers: {sorted(invalid)}')
        x, y = patch['local2global'][:2, 2]
        lateral, longitudinal = patch['roi_size']
        geometries = self._get_map().get_map_geom(
            patch_box=(x, y, lateral, longitudinal),
            patch_angle=patch['yaw_degrees'],
            layer_names=layers)
        patch['geometries'] = dict(geometries)
        return patch
