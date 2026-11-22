from pathlib import Path
from typing import Optional

import cv2
import mmcv
import numpy as np
from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from mmhdmap.registry import VISUALIZERS
from mmhdmap.structures import MapDataSample


@VISUALIZERS.register_module()
class BevMapVisualizer(Visualizer):
    """Display a prepared BEV image at its original size and orientation.

    Images must be unnormalized RGB uint8 HWC arrays or uint8 intensity
    HW/HW1 arrays. Intensity is replicated into RGB for colored GT overlays;
    its numerical values are not stretched or log-mapped. Image preparation is
    the caller's responsibility. GT polylines use current-image pixel
    coordinates; no crop offset or projection is applied. Set dataset_meta
    with classes and optionally an RGB palette. Predictions are not supported.
    """

    PALETTE = [(255, 80, 80), (80, 220, 100), (80, 160, 255), (255, 200, 60),
               (210, 100, 255), (60, 220, 220), (255, 140, 60),
               (240, 100, 180)]

    def _draw_gt(self, image, instances, draw_labels, draw_points,
                 draw_endpoints, draw_instance_ids, draw_attributes):
        """Draw on a private RGB copy, without changing GT arrays."""
        classes = self.dataset_meta.get('classes', [])
        palette = self.dataset_meta.get('palette', self.PALETTE)
        if not classes or not palette:
            raise ValueError(
                'GT visualization requires classes and a nonempty palette')
        height, width = image.shape[:2]
        for index, (label, points) in enumerate(
                zip(instances.labels, instances.polylines)):
            label = int(label)
            if not 0 <= label < len(classes):
                raise ValueError(f'Invalid GT label: {label}')
            if hasattr(points, 'detach'):
                points = points.detach().cpu().numpy()
            points = np.asarray(points)
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                    or not np.isfinite(points).all()):
                raise ValueError(
                    'GT polylines must contain finite [K, 2] vertices')
            color = tuple(int(c) for c in palette[label % len(palette)])
            # OpenCV draws the supplied tuple directly into the RGB array.
            # Fixed-point vertices preserve subpixel coordinates for rendering.
            vertices = np.rint(points * 256).astype(np.int32)
            cv2.polylines(
                image, [vertices], False, color, 2, cv2.LINE_AA, shift=8)
            pixels = np.rint(points).astype(np.int32)
            if draw_points:
                for point in pixels:
                    cv2.circle(image, tuple(point), 3, color, -1, cv2.LINE_AA)
            if draw_endpoints:
                cv2.circle(image, tuple(pixels[0]), 6, (255, 255, 255), 2,
                           cv2.LINE_AA)
                cv2.drawMarker(image, tuple(pixels[-1]), (255, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)
            text = classes[label] if draw_labels else ''
            if draw_instance_ids:
                text = f'#{index} {text}'.rstrip()
            if draw_attributes and 'attributes' in instances:
                attrs = instances.attributes[index]
                text = ' '.join(
                    filter(None, [
                        text, ', '.join(f'{k}={v}' for k, v in attrs.items())
                    ]))
            if text:
                (text_width,
                 text_height), _ = cv2.getTextSize(text,
                                                   cv2.FONT_HERSHEY_SIMPLEX,
                                                   0.45, 1)
                x = int(
                    np.clip(pixels[0, 0], 0, max(0, width - text_width - 1)))
                y = int(
                    np.clip(pixels[0, 1] - 8, text_height + 2,
                            max(text_height + 2, height - 2)))
                cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1, cv2.LINE_AA)
        return image

    @master_only
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: Optional[MapDataSample] = None,
                       draw_gt: bool = False,
                       draw_pred: bool = False,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       step: int = 0,
                       draw_labels: bool = True,
                       draw_points: bool = False,
                       draw_endpoints: bool = False,
                       draw_instance_ids: bool = False,
                       draw_attributes: bool = False) -> None:
        """Show the image and save to out_file or configured vis backends.

        Prefer PNG for lossless inspection. No resizing, flipping, axes or
        margins are added. Polylines are never automatically closed or filled.
        Endpoint markers are a circle (start) and cross (end); instance IDs
        are indices within this crop. If supplied,
        data_sample.img_shape must match the displayed image's (H, W).
        """
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.size == 0
                or not (image.ndim == 2 or
                        (image.ndim == 3 and image.shape[2] in (1, 3)))):
            raise ValueError(
                'image must be nonempty uint8 RGB HWC or intensity HW/HW1')
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if data_sample is not None:
            if not isinstance(data_sample, MapDataSample):
                raise TypeError('data_sample must be a MapDataSample')
            shape = data_sample.get('img_shape')
            if shape is not None and tuple(shape[:2]) != image.shape[:2]:
                raise ValueError('data_sample.img_shape does not match image')
        if draw_pred:
            raise NotImplementedError('Prediction drawing is not implemented')
        if draw_gt and (data_sample is None
                        or 'gt_instances' not in data_sample):
            raise ValueError('draw_gt requires data_sample.gt_instances')
        if not np.isfinite(wait_time) or wait_time < 0:
            raise ValueError('wait_time must be finite and nonnegative')

        drawn_image = image.copy()
        if draw_gt:
            drawn_image = self._draw_gt(drawn_image, data_sample.gt_instances,
                                        draw_labels, draw_points,
                                        draw_endpoints, draw_instance_ids,
                                        draw_attributes)
        self.set_image(drawn_image)
        if show:
            self.show(drawn_image, win_name=name, wait_time=wait_time)
        if out_file is not None:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            # MMCV's OpenCV writer expects BGR, unlike this public interface.
            if not mmcv.imwrite(drawn_image[..., ::-1], str(out_file)):
                raise OSError(f'Failed to save visualization: {out_file}')
        else:
            self.add_image(name, drawn_image, step)
