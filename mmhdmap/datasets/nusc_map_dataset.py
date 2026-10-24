from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np

from mmhdmap.registry import DATASETS
from .base_map_dataset import BaseMapDataset


@DATASETS.register_module()
class NuscMapDataset(BaseMapDataset):
    """Read mmhdmap nuScenes infos in MMEngine's standard dataset format.

    BaseDataset loads ``metainfo`` and ``data_list``. This adapter resolves
    nested paths and validates calibration without loading sensor payloads
    or extracting map geometry. Only info protocol version 1.0 is supported.

    Poses use column vectors and meters. Top-level ``ego2global`` is the
    LiDAR-time ego pose; each camera's ``ego2global`` is its capture-time
    ego pose. Camera ``ego2cam`` remains a static extrinsic. All timestamps
    are integer microseconds.

    Args:
        ann_file (str): Path to a mmhdmap nuScenes infos PKL.
        data_root (str): Root directory of nuScenes data.
        data_prefix (dict, optional): Prefixes for ``img_path``, ``lidar_path``
            and ``map_path``, relative to data_root unless absolute. All
            default to data_root; stored paths already include their sensor
            or map subdirectory.
        **kwargs: Arguments forwarded to BaseMapDataset.
    """

    METAINFO = dict(classes=('ped_crossing', 'divider', 'boundary'))
    CAMERA_NAMES = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')

    def __init__(self,
                 ann_file: str,
                 data_root: str = '',
                 data_prefix: Optional[dict] = None,
                 **kwargs) -> None:
        prefixes = dict(img_path='', lidar_path='', map_path='')
        if data_prefix is not None:
            prefixes.update(data_prefix)
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=prefixes,
            **kwargs)

    def _resolve_path(self, filename: str, prefix_key: str) -> str:
        """Join a portable info path to its configured data prefix."""
        path = Path(filename)
        if not filename or path.is_absolute() or '..' in path.parts:
            raise ValueError(f'{prefix_key} must be relative: {filename!r}')
        return str(Path(self.data_prefix[prefix_key]) / path)

    @staticmethod
    def _matrix(value, size: int, name: str) -> np.ndarray:
        """Validate a finite homogeneous matrix without reducing precision."""
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (size, size) or not np.isfinite(matrix).all():
            raise ValueError(f'{name}: expected a finite {size}x{size} matrix')
        last_row = np.zeros(size)
        last_row[-1] = 1
        if not np.allclose(matrix[-1], last_row):
            raise ValueError(f'{name}: invalid homogeneous last row')
        return matrix

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Resolve paths and preserve capture poses and token-based links.

        ``sample_idx`` is assigned by BaseDataset when retrieving a sample;
        ``frame_idx`` remains the original frame number within its scene.
        """
        if self.metainfo.get('info_version') != '1.0':
            raise ValueError(
                'NuscMapDataset requires mmhdmap info_version 1.0')
        info = deepcopy(raw_data_info)
        for key in ('ego2global', 'lidar2ego'):
            info[key] = self._matrix(info[key], 4, key)
        if set(info['images']) != set(self.CAMERA_NAMES):
            raise ValueError('images must contain the six nuScenes cameras')
        images = {}
        for name in self.CAMERA_NAMES:
            camera = info['images'][name]
            camera['img_path'] = self._resolve_path(camera['img_path'],
                                                    'img_path')
            for key, size in (('cam2img', 3), ('ego2cam', 4), ('ego2global',
                                                               4)):
                camera[key] = self._matrix(camera[key], size, f'{name}/{key}')
            images[name] = camera
        info['images'] = images
        lidar = info['lidar_points']
        lidar['lidar_path'] = self._resolve_path(lidar['lidar_path'],
                                                 'lidar_path')
        info['map_path'] = self._resolve_path(info['map_path'], 'map_path')
        return info
