import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mmhdmap.visualization import crop_nusc_basemap


class TestNuscBasemap(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.map_path = self.root / 'map.json'
        self.map_path.write_text(json.dumps(dict(canvas_edge=[20, 20])))
        self.png_path = self.root / 'map.png'
        rows, cols = np.indices((20, 20))
        image = np.stack((cols * 10, rows * 10, np.full_like(rows, 77)),
                         axis=-1)
        Image.fromarray(image.astype(np.uint8)).save(self.png_path)
        self.pose = np.eye(4)
        # Global meters at the center of source pixel (row=10, column=10).
        self.pose[:2, 3] = (10.5, 9.5)

    def crop(self, **kwargs):
        args = dict(
            basemap_path=str(self.png_path),
            map_path=str(self.map_path),
            ego2global=self.pose,
            roi_size=(5, 5))
        args.update(kwargs)
        return crop_nusc_basemap(**args)

    def test_center_forward_and_left_at_zero_yaw(self):
        image = self.crop()
        self.assertEqual(image.shape, (5, 5, 3))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image[2, 2], [100, 100, 77])
        # Facing global +X: forward moves right in the source image.
        np.testing.assert_array_equal(image[0, 2], [120, 100, 77])
        # Vehicle left is global +Y: up in the source image.
        np.testing.assert_array_equal(image[2, 0], [100, 80, 77])

    def test_yaw_and_pose_translation(self):
        self.pose[:2, :2] = [[0, -1], [1, 0]]
        self.pose[:2, 3] = (5.5, 14.5)
        image = self.crop()
        np.testing.assert_array_equal(image[2, 2], [50, 50, 77])
        # Facing global +Y: both map and vehicle left must remain unmirrored.
        np.testing.assert_array_equal(image[0, 2], [50, 30, 77])
        np.testing.assert_array_equal(image[2, 0], [30, 50, 77])

    def test_resolution_comes_from_canvas_dimensions(self):
        self.map_path.write_text(json.dumps(dict(canvas_edge=[10, 10])))
        self.pose[:2, 3] = (5.25, 4.75)
        image = self.crop(roi_size=(2.5, 3.5))
        self.assertEqual(image.shape, (7, 5, 3))
        np.testing.assert_array_equal(image[3, 2], [100, 100, 77])
        np.testing.assert_array_equal(image[0, 2], [130, 100, 77])
        np.testing.assert_array_equal(image[3, 0], [100, 80, 77])

    def test_rounding_roi_preserves_native_pixel_spacing(self):
        np.testing.assert_array_equal(
            self.crop(roi_size=(5.2, 4.8)), self.crop(roi_size=(5, 5)))
        self.map_path.write_text(json.dumps(dict(canvas_edge=[40, 20])))
        with self.assertRaisesRegex(ValueError, 'equal X and Y'):
            self.crop()

    def test_grayscale_and_outside_fill(self):
        Image.fromarray(np.full((20, 20), 180,
                                dtype=np.uint8)).save(self.png_path)
        np.testing.assert_array_equal(self.crop(), np.full((5, 5, 3), 180))
        self.pose[:2, 3] = (0.5, 9.5)
        crop = self.crop()
        np.testing.assert_array_equal(crop[0, 2], [180, 180, 180])
        np.testing.assert_array_equal(crop[-1, 2], [32, 32, 32])
        self.pose[:2, 3] = (100, 100)
        np.testing.assert_array_equal(self.crop(), np.full((5, 5, 3), 32))

    def test_invalid_arguments_and_pillow_limit_restoration(self):
        for overrides in (
                dict(roi_size=(0, 5)),
                dict(roi_size=(float('nan'), 5)),
                dict(ego2global=np.eye(3)),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.crop(**overrides)
        limit = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = 1
            self.crop()
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 1)
            with self.assertRaises(FileNotFoundError):
                self.crop(basemap_path=str(self.root / 'missing.png'))
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 1)
        finally:
            Image.MAX_IMAGE_PIXELS = limit


if __name__ == '__main__':
    unittest.main()
