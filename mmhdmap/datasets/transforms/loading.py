import mmcv
from mmcv.transforms import BaseTransform
from mmengine.fileio import get

from mmhdmap.registry import TRANSFORMS


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
