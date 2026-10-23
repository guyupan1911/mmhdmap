from pathlib import Path
from typing import List, Optional

import numpy as np
from mmengine.fileio import load
from pyquaternion import Quaternion

from mmhdmap.registry import DATASETS
from .base_map_dataset import BaseMapDataset


@DATASETS.register_module()
class NuscMapDataset(BaseMapDataset):
    """Read HRDX nuScenes map infos before pipeline processing.

    Each annotation is a list of frame dictionaries in HRDX's format, not
    MMDetection3D's nuScenes info format. Images, point clouds and map geometry
    are loaded by subsequent transforms.

    Parsed poses use homogeneous matrices with column vectors and translations
    in meters. ``ego2global`` is the LiDAR-time ego pose. Camera ``ego2cam``
    matrices are static extrinsics from the source annotations; they do not
    compensate for camera/LiDAR timestamp differences. Timestamps remain in
    microseconds. Camera order follows the HRDX converter.

    Args:
        ann_file (str): Path to an HRDX map infos file.
        data_root (str): Root directory of nuScenes data.
        ann_path_prefix (str, optional): Old path prefix to remove from relative
            image and LiDAR paths before joining their data prefixes. Absolute
            paths are preserved.
        data_prefix (dict, optional): Prefixes for ``img_path``, ``lidar_path``
            and ``map_path``, relative to data_root unless absolute. Defaults
            to the data root for sensors and ``maps/expansion`` for maps.
        **kwargs: Arguments forwarded to BaseMapDataset.
    """

    METAINFO = dict(classes=('ped_crossing', 'divider', 'boundary'))
    CAMERA_NAMES = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')

    def __init__(self,
                 ann_file: str,
                 data_root: str = '',
                 ann_path_prefix: Optional[str] = None,
                 data_prefix: Optional[dict] = None,
                 **kwargs) -> None:
        self.ann_path_prefix = (
            Path(ann_path_prefix) if ann_path_prefix is not None else None)
        prefixes = dict(img_path='', lidar_path='', map_path='maps/expansion')
        if data_prefix is not None:
            prefixes.update(data_prefix)
        # BaseDataset may call parse_data_info during its initialization.
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=prefixes,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load and parse the list of HRDX frame annotations."""
        annotations = load(self.ann_file)
        if not isinstance(annotations, list):
            raise TypeError('HRDX map infos must be a list of sample dicts')

        data_list = []
        for index, raw_data_info in enumerate(annotations):
            if not isinstance(raw_data_info, dict):
                raise TypeError(f'Annotation at index {index} must be a dict')
            data_list.append(self.parse_data_info(raw_data_info))
        return data_list

    def _resolve_sensor_path(self, path: str, prefix_key: str) -> str:
        """Resolve a sensor path without guessing its original data root."""
        path = Path(path)
        if path.is_absolute():
            return str(path)
        # TODO: Update the nuScenes converter to save paths relative to
        # data_root and regenerate the PKLs, then remove ann_path_prefix
        # and this legacy prefix compatibility logic.
        if self.ann_path_prefix is not None:
            try:
                path = path.relative_to(self.ann_path_prefix)
            except ValueError:
                # Other relative paths are already relative to the data prefix.
                pass
        return str(Path(self.data_prefix[prefix_key]) / path)

    @staticmethod
    def _pose_matrix(rotation, translation) -> np.ndarray:
        """Convert a (w, x, y, z) quaternion and translation to a 4x4 pose."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Quaternion(rotation).rotation_matrix
        matrix[:3, 3] = translation
        return matrix

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Normalize paths and calibration without loading sensor or map data.

        Source indices refer to the original annotation list. BaseDataset adds
        ``sample_idx`` for the current dataset when get_data_info is called.
        No map reference frame, GT or MapDataSample is constructed here.
        """
        images = {}
        for name in self.CAMERA_NAMES:
            camera = raw_data_info['cams'][name]
            intrinsic = np.array(camera['intrinsics'], dtype=np.float64)
            extrinsic = np.array(camera['extrinsics'], dtype=np.float64)
            if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
                raise ValueError(
                    f'{name}: expected 3x3 intrinsics and 4x4 extrinsics')
            images[name] = dict(
                img_path=self._resolve_sensor_path(camera['img_fpath'],
                                                   'img_path'),
                cam2img=intrinsic,
                ego2cam=extrinsic)

        location = raw_data_info['location']
        data_info = dict(
            token=raw_data_info['token'],
            scene_name=raw_data_info['scene_name'],
            timestamp=raw_data_info['timestamp'],
            location=location,
            source_sample_idx=raw_data_info['sample_idx'],
            ego2global=self._pose_matrix(raw_data_info['e2g_rotation'],
                                         raw_data_info['e2g_translation']),
            lidar2ego=self._pose_matrix(
                raw_data_info['lidar2ego_rotation'],
                raw_data_info['lidar2ego_translation']),
            images=images,
            lidar_points=dict(
                lidar_path=self._resolve_sensor_path(
                    raw_data_info['lidar_path'], 'lidar_path')),
            map_path=str(
                Path(self.data_prefix['map_path']) / f'{location}.json'))
        for key in ('prev', 'next'):
            if key in raw_data_info:
                data_info[f'source_{key}_idx'] = raw_data_info[key]
        return data_info
