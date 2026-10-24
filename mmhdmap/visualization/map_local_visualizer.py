from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import mmcv
import numpy as np
from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from mmhdmap.registry import VISUALIZERS
from mmhdmap.structures import MapDataSample
from .map_projection import project_map_geometries


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
    # RGB colors for raw nuScenes SDK layers, separate from task class labels.
    MAP_LAYER_COLORS = {
        'road_divider': (255, 100, 60),
        'lane_divider': (40, 160, 255),
        'ped_crossing': (60, 210, 100),
        'drivable_area': (190, 150, 90),
        'road_segment': (180, 120, 220),
        'road_block': (240, 150, 180),
        'lane': (80, 200, 200),
        'walkway': (230, 200, 90),
        'stop_line': (240, 80, 180),
        'carpark_area': (140, 150, 230),
    }

    def draw_map_geometries(self, bev_image: np.ndarray, geometries: dict,
                            resolution: float) -> np.ndarray:
        """Overlay raw local-meter SDK geometry on a copy of an RGB basemap.

        Local X is forward and Y left; the image is ego-centered and forward-up.
        Multipart geometries are drawn separately and polygon holes remain
        unfilled. Polygon clipping edges are raw ROI outlines, not boundary GT.
        """
        if (not isinstance(bev_image, np.ndarray)
                or bev_image.dtype != np.uint8 or bev_image.ndim != 3
                or bev_image.shape[2] != 3 or bev_image.size == 0):
            raise ValueError(
                'Map overlays require a nonempty RGB uint8 basemap')
        if resolution is None or not np.isfinite(
                resolution) or resolution <= 0:
            raise ValueError('map_resolution must be a positive value in m/px')
        unknown = set(geometries) - set(self.MAP_LAYER_COLORS)
        if unknown:
            raise ValueError(f'Unknown map layer colors: {sorted(unknown)}')
        image = bev_image.copy()
        height, width = image.shape[:2]

        def pixels(coords):
            xy = np.asarray(coords)[:, :2]
            uv = np.column_stack(((width - 1) / 2 - xy[:, 1] / resolution,
                                  (height - 1) / 2 - xy[:, 0] / resolution))
            return np.rint(uv).astype(np.int32)

        def draw(geometry, color):
            if geometry.is_empty:
                return
            if hasattr(geometry, 'geoms'):
                for part in geometry.geoms:
                    draw(part, color)
            elif geometry.geom_type == 'Polygon':
                rings = [pixels(geometry.exterior.coords)]
                rings.extend(
                    pixels(ring.coords) for ring in geometry.interiors)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [rings[0]], 1)
                if len(rings) > 1:
                    cv2.fillPoly(mask, rings[1:], 0)
                inside = mask.astype(bool)
                image[inside] = np.rint(image[inside] * 0.75 +
                                        np.asarray(color) * 0.25)
                cv2.polylines(image, rings, True, color, 2, cv2.LINE_AA)
            elif geometry.geom_type in ('LineString', 'LinearRing'):
                cv2.polylines(image, [pixels(geometry.coords)], False, color,
                              2, cv2.LINE_AA)
            # Point intersections at the ROI boundary have no drawable length.

        # Fill polygon layers first so they do not obscure divider lines.
        layers = sorted(geometries, key=lambda name: name.endswith('divider'))
        for name in layers:
            for geometry in geometries[name]:
                draw(geometry, self.MAP_LAYER_COLORS[name])
        return image

    def draw_projected_map(self,
                           images: Sequence[np.ndarray],
                           data_sample: MapDataSample,
                           geometries: dict,
                           local2global: np.ndarray,
                           ground_height: float = 0.0) -> list:
        """Draw map outlines on copies of calibrated, unnormalized RGB images.

        The caller must supply images matching the camera intrinsics. Drawing
        happens before display resizing. Each camera uses its capture-time
        ego pose. All views share a horizontal ground plane at the reference
        LiDAR-time ego elevation plus ground_height. Occlusion is not handled.
        """
        names = data_sample.camera_names
        if len(images) != len(names):
            raise ValueError('images and camera_names must have equal lengths')
        if len(set(names)) != len(names):
            raise ValueError('camera_names must not contain duplicates')
        unknown = set(geometries) - set(self.MAP_LAYER_COLORS)
        if unknown:
            raise ValueError(f'Unknown map layer colors: {sorted(unknown)}')
        if 'images' not in data_sample or 'ego2global' not in data_sample:
            raise ValueError('Map projection requires images calibration and '
                             'ego2global in data_sample')
        ground_z = data_sample.ego2global[2, 3] + ground_height
        outputs = []
        for name, image in zip(names, images):
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[2] != 3
                    or image.size == 0):
                raise ValueError(
                    f'{name}: expected a nonempty RGB uint8 image')
            camera = data_sample.images[name]
            if 'ego2global' not in camera:
                raise ValueError(
                    f'{name}: camera capture-time ego2global is required')
            global2cam = camera['ego2cam'] @ np.linalg.inv(
                camera['ego2global'])
            projected = project_map_geometries(
                geometries,
                local2global,
                global2cam,
                camera['cam2img'],
                image.shape[:2],
                ground_z=ground_z)
            output = image.copy()
            for layer, segments in projected.items():
                if len(segments):
                    cv2.polylines(output,
                                  np.rint(segments).astype(np.int32), False,
                                  self.MAP_LAYER_COLORS[layer], 3, cv2.LINE_AA)
            outputs.append(output)
        return outputs

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
        bev_image: Optional[np.ndarray] = None,
        map_layers: Sequence[str] = ()
    ) -> np.ndarray:
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
        if map_layers:
            # Put the legend below the canvas, never over native map pixels.
            legend = np.full((24 * len(map_layers) + 12, canvas.shape[1], 3),
                             32,
                             dtype=np.uint8)
            for index, name in enumerate(map_layers):
                y = 12 + index * 24
                color = self.MAP_LAYER_COLORS[name]
                cv2.line(legend, (map_left, y), (map_left + 24, y), color, 3)
                cv2.putText(legend, name, (map_left + 32, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            cv2.LINE_AA)
            drawn_image = np.concatenate((drawn_image, legend), axis=0)
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
                       bev_image: Optional[np.ndarray] = None,
                       map_geometries: Optional[dict] = None,
                       map_resolution: Optional[float] = None,
                       project_map: bool = True,
                       map_local2global: Optional[np.ndarray] = None,
                       ground_height: float = 0.0) -> None:
        """Visualize one sample using separately supplied camera images.

        Args:
            name (str): Name used by the display window and backends.
            images (Sequence[np.ndarray]): RGB uint8 HWC images, aligned with
                ``data_sample.metainfo['camera_names']``. The caller prepares
                display images from unnormalized inputs or source files.
            data_sample (MapDataSample): Sample metadata and annotations.
                Camera names, calibration and capture-time ego poses are used. Training
                GT and prediction overlays are not implemented yet.
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
            map_geometries (dict, optional): Raw SDK layer geometries in local
                meters, sharing the basemap's pose and ROI. Kept separate from
                training GT and model predictions in ``data_sample``.
            map_resolution (float, optional): Basemap resolution in meters per
                pixel, required for a basemap geometry overlay.
            project_map (bool): Also draw raw map outlines on camera images.
                Defaults to True. Set False for camera-only or BEV-only views.
            map_local2global (np.ndarray, optional): 3x3 planar transform from
                the map extractor, required for camera projection.
            ground_height (float): Assumed horizontal ground plane offset in
                meters along global Z relative to the reference LiDAR-time ego
                origin. Default zero
                assumes the ground passes through that origin.
        """
        if not isinstance(data_sample, MapDataSample):
            raise TypeError('data_sample must be a MapDataSample')
        if 'camera_names' not in data_sample.metainfo:
            raise ValueError('data_sample.metainfo must contain camera_names')
        if project_map:
            if map_geometries is None or map_local2global is None:
                raise ValueError('project_map requires map_geometries and '
                                 'map_local2global')
            images = self.draw_projected_map(
                images,
                data_sample,
                map_geometries,
                map_local2global,
                ground_height=ground_height)
        self.add_multiview_images(
            name=name,
            images=images,
            camera_names=data_sample.camera_names,
            out_file=out_file,
            show=show,
            wait_time=wait_time,
            step=step,
            bev_image=bev_image,
            map_geometries=map_geometries,
            map_resolution=map_resolution)

    @master_only
    def add_multiview_images(self,
                             name: str,
                             images: Sequence[np.ndarray],
                             camera_names: Sequence[str],
                             out_file: Optional[str] = None,
                             show: bool = False,
                             wait_time: float = 0,
                             step: int = 0,
                             bev_image: Optional[np.ndarray] = None,
                             map_geometries: Optional[dict] = None,
                             map_resolution: Optional[float] = None) -> None:
        """Draw camera views and save to out_file or configured backends."""
        if map_geometries is not None and bev_image is not None:
            bev_image = self.draw_map_geometries(bev_image, map_geometries,
                                                 map_resolution)
        drawn_image = self._draw_multiview_images(
            images,
            camera_names,
            bev_image,
            map_layers=tuple(map_geometries or ()))
        self.set_image(drawn_image)
        if show:
            self.show(drawn_image, win_name=name, wait_time=wait_time)
        if out_file is not None:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            if not mmcv.imwrite(drawn_image[..., ::-1], str(out_file)):
                raise OSError(f'Failed to save visualization: {out_file}')
        else:
            self.add_image(name, drawn_image, step)
