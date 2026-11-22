from copy import deepcopy

import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.fileio import get
from mmengine.structures import InstanceData

from mmhdmap.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadVectorMapAnnotations(BaseTransform):
    """Load prepared vector instances without generating model targets.

    Requires ``instances`` (an empty list denotes annotated negative data).
    Adds ``gt_instances`` with labels (int64 tensor), polylines (a list of
    float32 [Ki, 2] arrays), and attributes (a list of copied dictionaries).
    Coordinates retain their input frame and point order. No normalization,
    resampling, closure or attribute encoding is performed. Omit this
    transform from pipelines for unannotated inference images.
    """

    def transform(self, results: dict) -> dict:
        if 'instances' not in results:
            raise KeyError(
                'Missing instances: unannotated data is not empty GT')
        instances = results['instances']
        if not isinstance(instances, list):
            raise TypeError('instances must be a list')
        labels, polylines, attributes = [], [], []
        for instance in instances:
            label = instance['label']
            if type(label) is not int or label < 0:
                raise ValueError('label must be a nonnegative integer')
            points = np.array(instance['points'], dtype=np.float32, copy=True)
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                    or not np.isfinite(points).all()
                    or not np.any(np.diff(points, axis=0))):
                raise ValueError(
                    'points must be finite [K, 2] with nonzero length')
            attrs = instance.get('attrs', {})
            if not isinstance(attrs, dict):
                raise TypeError('attrs must be a dictionary')
            labels.append(label)
            polylines.append(points)
            attributes.append(deepcopy(attrs))
        results['gt_instances'] = InstanceData(
            labels=torch.tensor(labels, dtype=torch.long),
            polylines=polylines,
            attributes=attributes)
        return results


@TRANSFORMS.register_module()
class LoadMultiViewImagesFromFiles(BaseTransform):
    """Load camera images as RGB uint8 arrays in dataset camera order.

    Required Keys:
        images (dict): Ordered camera information, with an ``img_path`` for
            each camera. Calibration and other existing fields are preserved.

    Added Keys:
        imgs (list[np.ndarray]): RGB images, each with shape (H, W, 3).
            Images may have different sizes and are not stacked or resized.
        camera_names (list[str]): Camera names aligned with ``imgs``.
        ori_shapes (list[tuple[int, int]]): Original (H, W) image sizes.
        img_shapes (list[tuple[int, int]]): Current (H, W) image sizes.
    """

    def transform(self, results: dict) -> dict:
        """Load all views, reporting the camera and path on failure."""
        cameras = results['images']
        if not cameras:
            raise ValueError('images must contain at least one camera')

        imgs = []
        for name, camera in cameras.items():
            path = camera['img_path']
            try:
                # Ignore EXIF rotation to preserve calibration pixel coordinates.
                img = mmcv.imfrombytes(
                    get(path),
                    flag='color_ignore_orientation',
                    channel_order='bgr',
                    backend='cv2')
                if img is None:
                    raise ValueError('Image decoding returned None')
                # MMCV 2.1 does not convert channels with the EXIF-ignore flag.
                img = mmcv.bgr2rgb(img)
            except Exception as exc:
                raise RuntimeError(
                    f'Failed to load image for camera {name}: {path}') from exc
            imgs.append(img)

        results['imgs'] = imgs
        results['camera_names'] = list(cameras)
        results['ori_shapes'] = [img.shape[:2] for img in imgs]
        results['img_shapes'] = results['ori_shapes'].copy()
        return results
