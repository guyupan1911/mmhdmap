from pathlib import Path
from typing import Optional, Sequence, Tuple

import mmcv
import numpy as np
from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from mmhdmap.registry import VISUALIZERS
from mmhdmap.structures import MapDataSample


@VISUALIZERS.register_module()
class MapLocalVisualizer(Visualizer):
    """Display nuScenes camera views beside a BEV map panel.

    Front cameras occupy the top row and rear cameras the bottom row. Left
    and right refer to the vehicle's sides. Camera images remain upright and
    are never mirrored. The right panel contains a forward-facing vehicle
    icon over an optional local basemap. The map is displayed pixel for pixel;
    the canvas expands to fit it. All input and output images use RGB uint8
    arrays.

    Args:
        name (str): Visualizer instance name.
        vis_backends (list, optional): Visualization backend configurations.
        save_dir (str, optional): Root directory for visualization backends.
        image_size (tuple[int, int]): Display area (width, height) per camera,
            excluding its name bar. Images are resized with aspect preserved.
        car_icon_path (str, optional): Path to a right-facing RGBA icon.
            Defaults to the bundled car.png. Rotated counterclockwise by 90
            degrees for display.
        icon_height (int): Maximum display height of the vehicle icon.
    """

    CAMERA_LAYOUT = (
        ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'),
        ('CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'),
    )

    def __init__(self,
                 name: str = 'map_visualizer',
                 vis_backends: Optional[list] = None,
                 save_dir: Optional[str] = None,
                 image_size: Tuple[int, int] = (480, 270),
                 car_icon_path: Optional[str] = None,
                 icon_height: int = 120) -> None:
        if len(image_size) != 2 or any(not isinstance(size, int) or size <= 0
                                       for size in image_size):
            raise ValueError('image_size must contain two positive integers')
        if not isinstance(icon_height, int) or icon_height <= 0:
            raise ValueError('icon_height must be a positive integer')
        icon_path = (
            Path(car_icon_path) if car_icon_path is not None else
            Path(__file__).parent / 'assets' / 'car.png')
        icon = mmcv.imread(str(icon_path), flag='unchanged', backend='cv2')
        if icon is None or icon.ndim != 3 or icon.shape[2] != 4:
            raise ValueError(f'Expected an RGBA vehicle icon: {icon_path}')
        # OpenCV returns BGRA; preserve alpha while converting to RGBA.
        self._car_icon = np.ascontiguousarray(
            np.rot90(icon[..., [2, 1, 0, 3]]))
        self.image_size = tuple(image_size)
        self.icon_height = icon_height
        super().__init__(
            name=name, vis_backends=vis_backends, save_dir=save_dir)

    def _draw_multiview_images(
            self,
            images: Sequence[np.ndarray],
            camera_names: Sequence[str],
            bev_image: Optional[np.ndarray] = None) -> np.ndarray:
        """Compose RGB camera views without modifying input images."""
        expected = {name for row in self.CAMERA_LAYOUT for name in row}
        if len(images) != len(camera_names):
            raise ValueError('images and camera_names must have equal lengths')
        if len(set(camera_names)) != len(camera_names):
            raise ValueError('camera_names must not contain duplicates')
        if set(camera_names) != expected:
            raise ValueError(
                f'Expected cameras {sorted(expected)}, got {list(camera_names)}'
            )

        cameras = dict(zip(camera_names, images))
        width, height = self.image_size
        margin, label_height = 12, 32
        cell_height = height + label_height
        map_width = width
        map_height = 2 * cell_height + margin - label_height
        if bev_image is not None:
            if (not isinstance(bev_image, np.ndarray)
                    or bev_image.dtype != np.uint8 or bev_image.ndim != 3
                    or bev_image.shape[2] != 3 or bev_image.size == 0):
                raise ValueError(
                    'bev_image must be a nonempty RGB uint8 image')
            map_width = max(map_width, bev_image.shape[1])
            map_height = max(map_height, bev_image.shape[0])
        canvas = np.full((map_height + label_height + 2 * margin,
                          3 * width + map_width + 5 * margin, 3),
                         32,
                         dtype=np.uint8)
        names, positions = [], []
        for row, camera_row in enumerate(self.CAMERA_LAYOUT):
            for col, name in enumerate(camera_row):
                img = cameras[name]
                if (not isinstance(img, np.ndarray) or img.dtype != np.uint8
                        or img.ndim != 3 or img.shape[2] != 3
                        or min(img.shape[:2]) == 0):
                    raise ValueError(
                        f'{name}: expected a nonempty RGB uint8 image')
                x = margin + col * (width + margin)
                y = margin + row * (cell_height + margin)
                canvas[y:y + cell_height, x:x + width] = 16
                scale = min(width / img.shape[1], height / img.shape[0])
                resized = mmcv.imresize(
                    img, (max(1, round(img.shape[1] * scale)),
                          max(1, round(img.shape[0] * scale))),
                    interpolation='bilinear',
                    backend='cv2')
                left = x + (width - resized.shape[1]) // 2
                top = y + label_height + (height - resized.shape[0]) // 2
                canvas[top:top + resized.shape[0],
                       left:left + resized.shape[1]] = resized
                names.append(name)
                positions.append((x + width / 2, y + label_height / 2))

        map_left = margin + 3 * (width + margin)
        map_top = margin + label_height
        canvas[margin:map_top, map_left:map_left + map_width] = 16
        names.append('BEV MAP')
        positions.append((map_left + map_width / 2, margin + label_height / 2))

        if bev_image is not None:
            # Copy the crop directly; smaller maps are centered without scaling.
            left = map_left + (map_width - bev_image.shape[1]) // 2
            top = map_top + (map_height - bev_image.shape[0]) // 2
            canvas[top:top + bev_image.shape[0],
                   left:left + bev_image.shape[1]] = bev_image

        icon = self._car_icon
        scale = min(self.icon_height / icon.shape[0],
                    map_height / icon.shape[0], map_width / icon.shape[1])
        icon = mmcv.imresize(
            icon, (max(1, round(
                icon.shape[1] * scale)), max(1, round(icon.shape[0] * scale))),
            interpolation='bilinear',
            backend='cv2')
        top = map_top + (map_height - icon.shape[0]) // 2
        left = map_left + (map_width - icon.shape[1]) // 2
        region = canvas[top:top + icon.shape[0], left:left + icon.shape[1]]
        alpha = icon[..., 3:4].astype(np.float32) / 255
        region[:] = np.round(icon[..., :3] * alpha + region * (1 - alpha))

        self.set_image(canvas)
        self.draw_texts(
            names,
            np.array(positions),
            font_sizes=min(13, width / 30),
            colors='white',
            horizontal_alignments='center',
            vertical_alignments='center')
        drawn_image = self.get_image()
        if bev_image is not None:
            # Preserve exact map pixels after Matplotlib renders the labels.
            drawn_image[map_top:map_top + map_height, map_left:map_left +
                        map_width] = canvas[map_top:map_top + map_height,
                                            map_left:map_left + map_width]
        return drawn_image

    @master_only
    def add_datasample(self,
                       name: str,
                       images: Sequence[np.ndarray],
                       data_sample: MapDataSample,
                       out_file: Optional[str] = None,
                       show: bool = False,
                       wait_time: float = 0,
                       step: int = 0,
                       bev_image: Optional[np.ndarray] = None) -> None:
        """Visualize one sample using separately supplied camera images.

        Args:
            name (str): Name used by the display window and backends.
            images (Sequence[np.ndarray]): RGB uint8 HWC images, aligned with
                ``data_sample.metainfo['camera_names']``. The caller prepares
                display images from unnormalized inputs or source files.
            data_sample (MapDataSample): Sample metadata and annotations.
                Currently only camera names are used; map GT and prediction
                overlays are not implemented yet.
            out_file (str, optional): Output image path. If omitted, use the
                configured visualization backends.
            show (bool): Whether to display the composed image.
            wait_time (float): Display wait time in seconds.
            step (int): Step used when recording to backends.
            bev_image (np.ndarray, optional): Vehicle-centered, forward-up
                RGB uint8 local basemap, prepared by the caller. Its aspect
                ratio must reflect the physical ROI. Displayed at native size
                without resizing; the canvas expands as needed. The icon is
                a position marker, not a drawing of the vehicle to scale.
        """
        if not isinstance(data_sample, MapDataSample):
            raise TypeError('data_sample must be a MapDataSample')
        if 'camera_names' not in data_sample.metainfo:
            raise ValueError('data_sample.metainfo must contain camera_names')
        self.add_multiview_images(
            name=name,
            images=images,
            camera_names=data_sample.camera_names,
            out_file=out_file,
            show=show,
            wait_time=wait_time,
            step=step,
            bev_image=bev_image)

    @master_only
    def add_multiview_images(self,
                             name: str,
                             images: Sequence[np.ndarray],
                             camera_names: Sequence[str],
                             out_file: Optional[str] = None,
                             show: bool = False,
                             wait_time: float = 0,
                             step: int = 0,
                             bev_image: Optional[np.ndarray] = None) -> None:
        """Draw camera views and save to out_file or configured backends."""
        drawn_image = self._draw_multiview_images(images, camera_names,
                                                  bev_image)
        self.set_image(drawn_image)
        if show:
            self.show(drawn_image, win_name=name, wait_time=wait_time)
        if out_file is not None:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            if not mmcv.imwrite(drawn_image[..., ::-1], str(out_file)):
                raise OSError(f'Failed to save visualization: {out_file}')
        else:
            self.add_image(name, drawn_image, step)
