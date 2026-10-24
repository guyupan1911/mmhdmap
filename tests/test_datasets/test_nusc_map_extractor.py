import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mmhdmap.datasets.map_utils import NuScenesMapExtractor


class TestNuScenesMapExtractor(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.map_name = 'singapore-onenorth'
        self.map_path = self.root / 'maps' / 'expansion' / f'{self.map_name}.json'
        self.map_path.parent.mkdir(parents=True)
        self.png_path = self.root / 'maps' / 'basemap' / f'{self.map_name}.png'
        self.png_path.parent.mkdir(parents=True)
        # Minimal real SDK schema, including a divider crossing the whole ROI
        # and a polygon with a hole. No NuScenes instance or sensor tables.
        self.map_data = dict(
            version='1.3',
            canvas_edge=[20, 20],
            node=[],
            line=[],
            polygon=[],
            lane_connector=[],
            arcline_path_3={},
            connectivity={})
        self.map_data.update(
            {layer: []
             for layer in NuScenesMapExtractor.SUPPORTED_LAYERS})
        self.map_data['traffic_light'] = []
        coords = [(2.5, 9.5), (18.5, 9.5), (9, 8), (12, 8), (12, 11), (9, 11),
                  (10, 9), (11, 9), (11, 10), (10, 10)]
        self.map_data['node'] = [
            dict(token=str(i), x=x, y=y) for i, (x, y) in enumerate(coords)
        ]
        self.map_data['line'] = [dict(token='line', node_tokens=['0', '1'])]
        self.map_data['road_divider'] = [
            dict(token='divider', line_token='line')
        ]
        self.map_data['polygon'] = [
            dict(
                token='polygon',
                exterior_node_tokens=['2', '3', '4', '5'],
                holes=[dict(node_tokens=['6', '7', '8', '9'])])
        ]
        self.map_data['ped_crossing'] = [
            dict(token='crossing', polygon_token='polygon')
        ]
        self.map_path.write_text(json.dumps(self.map_data))
        rows, cols = np.indices((20, 20))
        image = np.stack((cols * 10, rows * 10, np.full_like(rows, 77)),
                         axis=-1)
        Image.fromarray(image.astype(np.uint8)).save(self.png_path)
        self.pose = np.eye(4)
        # Global meters at the center of source pixel (row=10, column=10).
        self.pose[:2, 3] = (10.5, 9.5)
        self.extractor = NuScenesMapExtractor(str(self.root), self.map_name)
        self.addCleanup(self.extractor.clear_cache)

    def set_canvas(self, canvas):
        self.map_data['canvas_edge'] = canvas
        self.map_path.write_text(json.dumps(self.map_data))
        self.extractor.clear_cache()

    def crop(self, **kwargs):
        args = dict(ego2global=self.pose, roi_size=(5, 5))
        args.update(kwargs)
        return self.extractor.extract_basemap(**args)['image']

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
        self.set_canvas([10, 10])
        self.pose[:2, 3] = (5.25, 4.75)
        image = self.crop(roi_size=(2.5, 3.5))
        self.assertEqual(image.shape, (7, 5, 3))
        np.testing.assert_array_equal(image[3, 2], [100, 100, 77])
        np.testing.assert_array_equal(image[0, 2], [130, 100, 77])
        np.testing.assert_array_equal(image[3, 0], [100, 80, 77])

    def test_rounding_roi_preserves_native_pixel_spacing(self):
        np.testing.assert_array_equal(
            self.crop(roi_size=(5.2, 4.8)), self.crop(roi_size=(5, 5)))
        self.set_canvas([40, 20])
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
                NuScenesMapExtractor(
                    str(self.root),
                    self.map_name,
                    basemap_dir=str(self.root / 'missing')).extract_basemap(
                        self.pose, (5, 5))
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 1)
        finally:
            Image.MAX_IMAGE_PIXELS = limit

    def test_sdk_geometry_clipping_and_lazy_bitmap(self):
        # Geometry extraction must work even without a basemap on disk.
        self.png_path.unlink()
        self.assertIsNone(self.extractor._map)
        patch = self.extractor.extract_geometries(self.pose, (5, 5))
        self.assertIsNone(self.extractor._bitmap)
        self.assertEqual(patch['roi_size'], (5, 5))
        geometries = patch['geometries']
        self.assertEqual(list(geometries), list(self.extractor.DEFAULT_LAYERS))
        line = geometries['road_divider'][0]
        np.testing.assert_allclose(line.coords, [[-2.5, 0], [2.5, 0]])
        self.assertEqual(geometries['lane_divider'], [])
        polygon = geometries['ped_crossing'][0].geoms[0]
        self.assertEqual(len(polygon.interiors), 1)
        self.assertAlmostEqual(polygon.area, 8)
        cached_map = self.extractor._map
        self.extractor.extract_geometries(self.pose, (5, 5))
        self.assertIs(cached_map, self.extractor._map)
        self.extractor.clear_cache()
        self.assertIsNone(self.extractor._map)

    def test_sdk_yaw_and_shared_quantized_roi(self):
        self.pose[:2, :2] = [[0, -1], [1, 0]]
        basemap = self.extractor.extract_basemap(self.pose, (5.2, 4.8))
        patch = self.extractor.extract_geometries(self.pose,
                                                  basemap['roi_size'])
        np.testing.assert_array_equal(patch['local2global'],
                                      basemap['local2global'])
        self.assertEqual(basemap['resolution'], 1)
        line = patch['geometries']['road_divider'][0]
        np.testing.assert_allclose(
            line.coords, [[0, 2.5], [0, -2.5]], atol=1e-10)
        self.assertEqual(
            self.extractor.extract_geometries(self.pose, (5, 5),
                                              layers=[])['geometries'], {})
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self.extractor.extract_geometries(
                self.pose, (5, 5), layers=['traffic_light'])

    def test_explicit_bitmap_directory(self):
        external = self.root / 'custom_basemap'
        external.mkdir()
        self.png_path.rename(external / self.png_path.name)
        extractor = NuScenesMapExtractor(
            str(self.root), self.map_name, basemap_dir=str(external))
        patch = extractor.extract_basemap(self.pose, (5, 5))
        np.testing.assert_array_equal(patch['image'][2, 2], [100, 100, 77])
        cached_bitmap = extractor._bitmap
        extractor.extract_basemap(self.pose, (5, 5))
        self.assertIs(cached_bitmap, extractor._bitmap)
        extractor.clear_cache()
        self.assertIsNone(extractor._bitmap)


if __name__ == '__main__':
    unittest.main()
