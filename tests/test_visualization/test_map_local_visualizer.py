import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from mmhdmap.registry import VISUALIZERS
from mmhdmap.structures import MapDataSample
from mmhdmap.visualization.map_projection import project_map_geometries


class TestMapLocalVisualizer(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        icon = np.zeros((5, 5, 4), dtype=np.uint8)
        icon[2, 4] = (0, 0, 255, 255)
        icon[2, 2] = (255, 0, 0, 128)
        self.icon_path = self.root / 'car.png'
        cv2.imwrite(str(self.icon_path), icon)
        self.visualizer = VISUALIZERS.build(
            dict(
                type='MapLocalVisualizer',
                name=self.id(),
                image_size=(80, 50),
                car_icon_path=str(self.icon_path),
                icon_height=5,
                vis_backends=[dict(type='LocalVisBackend')],
                save_dir=str(self.root)))
        self.addCleanup(self.visualizer.close)
        # Deliberately use an order different from the display layout.
        self.names = [
            'CAM_BACK', 'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT',
            'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT'
        ]
        self.colors = [(40 + i * 30, 50, 200 - i * 20) for i in range(6)]
        self.images = [
            np.full((10, 20, 3), color, dtype=np.uint8)
            for color in self.colors
        ]

    def test_layout_rotation_alpha_and_input_preservation(self):
        originals = [image.copy() for image in self.images]
        output = self.root / 'nested' / 'surround.png'
        self.visualizer.add_multiview_images(
            'sample', self.images, self.names, out_file=str(output))
        result = cv2.cvtColor(cv2.imread(str(output)), cv2.COLOR_BGR2RGB)
        self.assertEqual(result.shape, (200, 380, 3))
        expected_cells = {
            'CAM_FRONT_LEFT': (0, 0),
            'CAM_FRONT': (0, 1),
            'CAM_FRONT_RIGHT': (0, 2),
            'CAM_BACK_LEFT': (1, 0),
            'CAM_BACK': (1, 1),
            'CAM_BACK_RIGHT': (1, 2),
        }
        for name, color in zip(self.names, self.colors):
            row, col = expected_cells[name]
            x = 12 + col * 92 + 40
            y = 12 + row * 94 + 32 + 25
            np.testing.assert_array_equal(result[y, x], color)
        # Check orientation and blending without assuming exact canvas rounding.
        red = np.argwhere(np.all(result == [255, 0, 0], axis=-1))
        blue = np.argwhere(np.all(result == [16, 16, 144], axis=-1))
        self.assertEqual(red.shape, (1, 2))
        self.assertEqual(blue.shape, (1, 2))
        np.testing.assert_array_equal(blue[0] - red[0], [2, 0])
        # The vehicle belongs in the right map area, not between camera rows.
        np.testing.assert_allclose(blue[0], [116, 328], atol=2)
        np.testing.assert_array_equal(result[80, 320], [32, 32, 32])
        # The wide source image leaves letterboxing inside the taller cell.
        np.testing.assert_array_equal(result[45, 52], [16, 16, 16])
        for original, image in zip(originals, self.images):
            np.testing.assert_array_equal(original, image)
        self.assertFalse((self.root / 'vis_data' / 'vis_image').exists())

    def test_backend_output(self):
        self.visualizer.add_multiview_images(
            'sample', self.images, self.names, step=3)
        self.assertTrue(
            (self.root / 'vis_data' / 'vis_image' / 'sample_3.png').is_file())

    def test_datasample_entry_point(self):
        sample = MapDataSample(metainfo=dict(camera_names=self.names))
        originals = [image.copy() for image in self.images]
        reference = self.root / 'reference.png'
        self.visualizer.add_multiview_images(
            'reference', self.images, self.names, out_file=str(reference))
        expected = cv2.cvtColor(cv2.imread(str(reference)), cv2.COLOR_BGR2RGB)
        output = self.root / 'sample.png'
        self.visualizer.add_datasample(
            'sample',
            self.images,
            sample,
            out_file=str(output),
            project_map=False)
        result = cv2.cvtColor(cv2.imread(str(output)), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(sample.camera_names, self.names)
        self.assertEqual(list(sample.keys()), [])
        for original, image in zip(originals, self.images):
            np.testing.assert_array_equal(original, image)
        self.visualizer.add_datasample(
            'sample', self.images, sample, step=7, project_map=False)
        self.assertTrue(
            (self.root / 'vis_data' / 'vis_image' / 'sample_7.png').is_file())

    def test_invalid_datasample(self):
        with self.assertRaisesRegex(TypeError, 'MapDataSample'):
            self.visualizer.add_datasample('sample', self.images, {})
        with self.assertRaisesRegex(ValueError, 'camera_names'):
            self.visualizer.add_datasample('sample', self.images,
                                           MapDataSample())

    def test_small_bev_background_is_not_enlarged(self):
        sample = MapDataSample(metainfo=dict(camera_names=self.names))
        basemap = np.full((20, 40, 3), [70, 90, 110], dtype=np.uint8)
        original = basemap.copy()
        output = self.root / 'basemap.png'
        self.visualizer.add_datasample(
            'map',
            self.images,
            sample,
            out_file=str(output),
            bev_image=basemap,
            project_map=False)
        result = cv2.cvtColor(cv2.imread(str(output)), cv2.COLOR_BGR2RGB)
        # The 40x20 map stays at native size in the 80x144 BEV panel.
        np.testing.assert_array_equal(result[110, 312], [70, 90, 110])
        np.testing.assert_array_equal(result[104, 312], [32, 32, 32])
        np.testing.assert_array_equal(result[110, 300], [32, 32, 32])
        np.testing.assert_array_equal(result[80, 300], [32, 32, 32])
        np.testing.assert_array_equal(result[150, 300], [32, 32, 32])
        # The existing forward-facing icon is composited over the map.
        red = np.argwhere(np.all(result == [255, 0, 0], axis=-1))
        self.assertEqual(red.shape, (1, 2))
        np.testing.assert_allclose(red[0], [114, 328], atol=2)
        np.testing.assert_array_equal(basemap, original)
        with self.assertRaisesRegex(ValueError, 'bev_image'):
            self.visualizer.add_datasample(
                'map',
                self.images,
                sample,
                bev_image=basemap.astype(float),
                project_map=False)

    def test_large_bev_expands_canvas_and_preserves_pixels(self):
        sample = MapDataSample(metainfo=dict(camera_names=self.names))
        basemap = np.random.default_rng(0).integers(
            0, 256, size=(180, 140, 3), dtype=np.uint8)
        output = self.root / 'large_basemap.png'
        self.visualizer.add_datasample(
            'map',
            self.images,
            sample,
            out_file=str(output),
            bev_image=basemap,
            project_map=False)
        result = cv2.cvtColor(cv2.imread(str(output)), cv2.COLOR_BGR2RGB)
        self.assertEqual(result.shape, (236, 440, 3))
        displayed_map = result[44:224, 288:428]
        # Every background pixel is unchanged except the centered 5x5 icon.
        background = np.ones((180, 140), dtype=bool)
        background[87:92, 67:72] = False
        np.testing.assert_array_equal(displayed_map[background],
                                      basemap[background])

    def test_invalid_camera_inputs(self):
        with self.assertRaisesRegex(ValueError, 'equal lengths'):
            self.visualizer.add_multiview_images('sample', [], self.names)
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            self.visualizer.add_multiview_images('sample', self.images,
                                                 ['CAM_FRONT'] * 6)
        with self.assertRaisesRegex(ValueError, 'Expected cameras'):
            self.visualizer.add_multiview_images('sample', self.images[:-1],
                                                 self.names[:-1])
        with self.assertRaisesRegex(ValueError, 'RGB uint8'):
            self.visualizer.add_multiview_images(
                'sample', [image.astype(np.float32) for image in self.images],
                self.names)

    def test_sdk_geometry_overlay_holes_and_disconnected_lines(self):
        basemap = np.zeros((80, 80, 3), dtype=np.uint8)
        polygon = Polygon([(-15, -15), (15, -15), (15, 15), (-15, 15)],
                          holes=[[(-5, -5), (5, -5), (5, 5), (-5, 5)]])
        lines = MultiLineString([[(25, -15), (25, -5)], [(25, 5), (25, 15)]])
        geometries = dict(
            ped_crossing=[MultiPolygon([polygon])], lane_divider=[lines])
        result = self.visualizer.draw_map_geometries(basemap, geometries, 1)
        np.testing.assert_array_equal(result[40, 40], [0, 0, 0])
        fill = np.rint(
            np.array(self.visualizer.MAP_LAYER_COLORS['ped_crossing']) * 0.25)
        np.testing.assert_array_equal(result[30, 40], fill)
        # The two line parts are forward of the polygon with a gap at Y=0.
        np.testing.assert_array_equal(
            result[14, 28], self.visualizer.MAP_LAYER_COLORS['lane_divider'])
        np.testing.assert_array_equal(result[14, 40], [0, 0, 0])
        self.assertFalse(basemap.any())
        self.assertEqual(len(polygon.interiors), 1)
        sample = MapDataSample(metainfo=dict(camera_names=self.names))
        output = self.root / 'annotated.png'
        self.visualizer.add_datasample(
            'map',
            self.images,
            sample,
            out_file=str(output),
            bev_image=basemap,
            map_geometries=geometries,
            map_resolution=1,
            project_map=False)
        saved = cv2.imread(str(output))
        # Two legend rows are appended below the canvas, leaving the map size.
        self.assertEqual(saved.shape, (260, 380, 3))
        self.assertNotIn('gt_instances', sample)
        with self.assertRaisesRegex(ValueError, 'map_resolution'):
            self.visualizer.draw_map_geometries(basemap, geometries, None)

    def test_capture_time_poses_share_one_ground_plane(self):
        extrinsic = np.array(
            [[0, -1, 0, 0], [0, 0, -1, 1.5], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=float)
        intrinsic = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]])
        shifted_pose = np.eye(4)
        shifted_pose[:3, 3] = [1, 0, 0.5]
        names = ['CAM_FRONT', 'CAM_FRONT_LEFT']
        cameras = {
            name: dict(ego2cam=extrinsic, cam2img=intrinsic, ego2global=pose)
            for name, pose in zip(names, [np.eye(4), shifted_pose])
        }
        sample = MapDataSample(
            metainfo=dict(
                camera_names=names, images=cameras, ego2global=np.eye(4)))
        geometries = dict(lane_divider=[LineString([(5, -2), (5, 2)])])
        images = [np.zeros((101, 101, 3), dtype=np.uint8) for _ in names]
        projections = []

        def record_projection(*args, **kwargs):
            result = project_map_geometries(*args, **kwargs)
            projections.append(result['lane_divider'])
            return result

        with patch(
                'mmhdmap.visualization.map_local_visualizer.project_map_geometries',
                side_effect=record_projection) as project:
            self.visualizer.draw_projected_map(images, sample, geometries,
                                               np.eye(3))
        np.testing.assert_allclose(projections[0], [[[70, 65], [30, 65]]])
        np.testing.assert_allclose(projections[1], [[[75, 75], [25, 75]]])
        self.assertEqual(
            [call.kwargs['ground_z'] for call in project.call_args_list],
            [0, 0])
        del sample.images['CAM_FRONT']['ego2global']
        with self.assertRaisesRegex(ValueError, 'capture-time ego2global'):
            self.visualizer.draw_projected_map(images, sample, geometries,
                                               np.eye(3))

    def test_camera_projection_uses_named_calibration_and_preserves_inputs(
            self):
        front = np.array(
            [[0, -1, 0, 0], [0, 0, -1, 1], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=float)
        back = front.copy()
        back[0] *= -1
        back[2] *= -1
        intrinsic = np.array([[5, 0, 10], [0, 5, 5], [0, 0, 1]])
        cameras = {
            name:
            dict(
                ego2cam=front if name == 'CAM_FRONT' else back,
                ego2global=np.eye(4),
                cam2img=intrinsic)
            for name in self.names
        }
        sample = MapDataSample(
            metainfo=dict(
                camera_names=self.names, images=cameras, ego2global=np.eye(4)))
        geometries = dict(lane_divider=[LineString([(5, -3), (5, 3)])])
        originals = [image.copy() for image in self.images]
        result = self.visualizer.draw_projected_map(self.images, sample,
                                                    geometries, np.eye(3))
        for name, image, drawn in zip(self.names, self.images, result):
            if name == 'CAM_FRONT':
                self.assertFalse(np.array_equal(image, drawn))
                np.testing.assert_array_equal(
                    drawn[6, 10],
                    self.visualizer.MAP_LAYER_COLORS['lane_divider'])
            else:
                np.testing.assert_array_equal(image, drawn)
        output = self.root / 'projected.png'
        self.visualizer.add_datasample(
            'map',
            self.images,
            sample,
            out_file=str(output),
            map_geometries=geometries,
            map_local2global=np.eye(3))
        self.assertTrue(output.is_file())
        for original, image in zip(originals, self.images):
            np.testing.assert_array_equal(original, image)
        np.testing.assert_array_equal(sample.images['CAM_FRONT']['ego2cam'],
                                      front)
        self.assertNotIn('gt_instances', sample)
        with self.assertRaisesRegex(ValueError, 'map_local2global'):
            self.visualizer.add_datasample(
                'map',
                self.images,
                sample,
                project_map=True,
                map_geometries=geometries)


if __name__ == '__main__':
    unittest.main()
