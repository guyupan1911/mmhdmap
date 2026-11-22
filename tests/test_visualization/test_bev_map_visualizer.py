import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from mmengine.structures import InstanceData
from PIL import Image

from mmhdmap.registry import VISUALIZERS
from mmhdmap.structures import MapDataSample


class TestBevMapVisualizer(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.visualizer = VISUALIZERS.build(
            dict(
                type='BevMapVisualizer',
                name=self.id(),
                vis_backends=[dict(type='LocalVisBackend')],
                save_dir=str(self.root)))
        self.addCleanup(self.visualizer.close)
        # Asymmetric dimensions and colors detect axis and channel swaps.
        self.image = np.arange(12 * 17 * 3, dtype=np.uint16).reshape(12, 17, 3)
        self.image = (self.image % 256).astype(np.uint8)
        self.sample = MapDataSample(metainfo=dict(img_shape=(12, 17)))

    def test_lossless_save_orientation_and_no_mutation(self):
        image = self.image[:, ::-1]  # Negative-stride input is supported.
        original = image.copy()
        output = self.root / 'nested' / 'image.png'
        self.visualizer.add_datasample(
            'sample', image, self.sample, out_file=str(output))
        with Image.open(output) as saved:
            np.testing.assert_array_equal(np.array(saved), original)
        np.testing.assert_array_equal(image, original)
        self.assertNotIn('gt_instances', self.sample)
        self.assertFalse((self.root / 'vis_data/vis_image').exists())

    def test_backend_save(self):
        self.visualizer.add_datasample('sample', self.image, step=3)
        path = self.root / 'vis_data/vis_image/sample_3.png'
        with Image.open(path) as saved:
            np.testing.assert_array_equal(np.array(saved), self.image)

    def test_display_and_save_can_be_combined(self):
        with patch.object(self.visualizer, 'show') as show:
            self.visualizer.add_datasample(
                'sample',
                self.image,
                show=True,
                wait_time=0.1,
                out_file=str(self.root / 'shown.png'))
        show.assert_called_once()
        np.testing.assert_array_equal(show.call_args.args[0], self.image)
        self.assertEqual(show.call_args.kwargs['wait_time'], 0.1)

    def test_reject_invalid_inputs_and_unsupported_overlays(self):
        for image in (self.image.astype(float), self.image[:, :, :2],
                      np.zeros((0, 17, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                self.visualizer.add_datasample('bad', image)
        with self.assertRaisesRegex(ValueError, 'img_shape'):
            self.visualizer.add_datasample(
                'bad', self.image,
                MapDataSample(metainfo=dict(img_shape=(17, 12))))
        with self.assertRaises(TypeError):
            self.visualizer.add_datasample('bad', self.image, {})
        with self.assertRaises(NotImplementedError):
            self.visualizer.add_datasample('bad', self.image, draw_pred=True)
        with self.assertRaisesRegex(ValueError, 'gt_instances'):
            self.visualizer.add_datasample('bad', self.image, draw_gt=True)

    def test_gt_pixel_coordinates_colors_and_open_geometry(self):
        image = np.zeros((100, 160, 3), dtype=np.uint8)
        points = np.array([[20., 20.], [120., 20.], [120., 70.]],
                          dtype=np.float32)
        original_points = points.copy()
        sample = MapDataSample(
            metainfo=dict(img_shape=(100, 160), crop_origin=[5000, 5000]))
        sample.gt_instances = InstanceData(
            labels=torch.tensor([1]),
            polylines=[points],
            attributes=[{
                'type': 'solid'
            }])
        self.visualizer.dataset_meta = dict(
            classes=['lane', 'curb'], palette=[(0, 255, 0), (255, 0, 0)])
        path = self.root / 'gt.png'
        self.visualizer.add_datasample(
            'gt',
            image,
            sample,
            draw_gt=True,
            draw_labels=False,
            out_file=str(path))
        saved = np.array(Image.open(path))
        np.testing.assert_array_equal(saved[20, 60], [255, 0, 0])
        np.testing.assert_array_equal(saved[50, 120], [255, 0, 0])
        # Neither close the last vertex to the first nor fill the polygon.
        np.testing.assert_array_equal(saved[45, 70], [0, 0, 0])
        np.testing.assert_array_equal(saved[30, 100], [0, 0, 0])
        self.assertFalse(image.any())
        np.testing.assert_array_equal(points, original_points)
        self.visualizer.add_datasample(
            'details',
            image,
            sample,
            draw_gt=True,
            draw_points=True,
            draw_endpoints=True,
            draw_instance_ids=True,
            draw_attributes=True,
            out_file=str(self.root / 'details.png'))
        self.assertFalse(
            np.array_equal(saved,
                           np.array(Image.open(self.root / 'details.png'))))

    def test_empty_gt_preserves_image(self):
        self.visualizer.dataset_meta = dict(classes=['lane'])
        self.sample.gt_instances = InstanceData(
            labels=torch.empty(0, dtype=torch.long),
            polylines=[],
            attributes=[])
        self.visualizer.add_datasample(
            'empty',
            self.image,
            self.sample,
            draw_gt=True,
            out_file=str(self.root / 'empty.png'))
        np.testing.assert_array_equal(
            np.array(Image.open(self.root / 'empty.png')), self.image)

    def test_intensity_background_uses_same_gt_coordinates(self):
        gray = np.full((80, 120), 63, dtype=np.uint8)
        self.visualizer.dataset_meta = dict(
            classes=['lane'], palette=[(255, 0, 0)])
        sample = MapDataSample(metainfo=dict(img_shape=gray.shape))
        sample.gt_instances = InstanceData(
            labels=torch.tensor([0]),
            polylines=[np.array([[10, 20], [100, 20]], dtype=np.float32)],
            attributes=[{}])
        for image in (gray, gray[..., None]):
            path = self.root / 'intensity.png'
            self.visualizer.add_datasample(
                'intensity',
                image,
                sample,
                draw_gt=True,
                draw_labels=False,
                out_file=str(path))
            saved = np.array(Image.open(path))
            np.testing.assert_array_equal(saved[20, 50], [255, 0, 0])
            np.testing.assert_array_equal(saved[60, 50], [63, 63, 63])
        self.assertTrue(np.all(gray == 63))


if __name__ == '__main__':
    unittest.main()
