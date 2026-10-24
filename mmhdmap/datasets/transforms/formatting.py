from typing import Sequence

import numpy as np
import torch
from mmcv.transforms import BaseTransform

from mmhdmap.registry import TRANSFORMS
from mmhdmap.structures import MapDataSample


@TRANSFORMS.register_module()
class PackMapInputs(BaseTransform):
    """Pack loaded camera images and metadata at the end of the pipeline.

    Required keys are ``imgs`` (RGB HWC numpy arrays) and ``camera_names``
    (aligned with the images). Returns ``inputs`` and ``data_samples``:

    - ``inputs['imgs']`` is a list of contiguous CHW tensors in camera order.
      Image dtype and values are preserved. Different view sizes are allowed;
      batching, padding and normalization belong to the data preprocessor.
    - ``data_samples`` is a MapDataSample with a copy of available metadata.

    Optional ``gt_instances`` (InstanceData) and ``gt_sem_seg`` (PixelData)
    are attached if previous transforms supplied them. This transform does
    not generate annotations or infer their coordinate conventions.

    Args:
        meta_keys (Sequence[str]): Metadata keys to copy when present.
    """

    def __init__(
        self,
        meta_keys: Sequence[str] = ('token', 'sample_idx', 'scene_token',
                                    'scene_name', 'frame_idx', 'timestamp',
                                    'prev_token', 'next_token', 'location',
                                    'camera_names', 'images', 'ori_shapes',
                                    'img_shapes', 'ego2global', 'lidar2ego',
                                    'lidar_points', 'map_path')
    ) -> None:
        self.meta_keys = tuple(meta_keys)

    def transform(self, results: dict) -> dict:
        images = results['imgs']
        camera_names = results['camera_names']
        if not images or len(images) != len(camera_names):
            raise ValueError('imgs and camera_names must be nonempty and '
                             'have the same length')
        if len(set(camera_names)) != len(camera_names):
            raise ValueError('camera_names must be unique')

        tensors = []
        for name, image in zip(camera_names, images):
            if (not isinstance(image, np.ndarray) or image.ndim != 3
                    or image.shape[2] != 3 or image.size == 0):
                raise ValueError(f'{name}: expected a nonempty HWC RGB array')
            tensors.append(
                torch.from_numpy(
                    np.ascontiguousarray(image.transpose(2, 0, 1))))

        data_sample = MapDataSample(metainfo={
            key: results[key]
            for key in self.meta_keys if key in results
        })
        for key in ('gt_instances', 'gt_sem_seg'):
            if key in results:
                setattr(data_sample, key, results[key])

        return dict(inputs=dict(imgs=tensors), data_samples=data_sample)
