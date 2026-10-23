import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope

from mmhdmap.registry import TRANSFORMS


class TestLoadMultiViewImagesFromFiles(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        init_default_scope('mmhdmap')

    def test_rgb_order_sizes_and_calibration(self):
        cameras = {}
        for name, shape, bgr in (
            ('CAM_BACK', (3, 5, 3), (0, 0, 255)),
            ('CAM_FRONT', (4, 6, 3), (255, 0, 0)),
        ):
            path = self.root / f'{name}.png'
            img = np.empty(shape, dtype=np.uint8)
            img[:] = bgr
            self.assertTrue(cv2.imwrite(str(path), img))
            cameras[name] = dict(
                img_path=str(path), cam2img=np.eye(3), ego2cam=np.eye(4))

        results = dict(images=cameras, token='sample-token')
        pipeline = Compose([dict(type='LoadMultiViewImagesFromFiles')])
        output = pipeline(results)
        self.assertEqual(output['camera_names'], ['CAM_BACK', 'CAM_FRONT'])
        self.assertEqual(output['ori_shapes'], [(3, 5), (4, 6)])
        self.assertEqual(output['img_shapes'], output['ori_shapes'])
        self.assertEqual(output['imgs'][0].dtype, np.uint8)
        np.testing.assert_array_equal(output['imgs'][0][0, 0], [255, 0, 0])
        np.testing.assert_array_equal(output['imgs'][1][0, 0], [0, 0, 255])
        self.assertIs(output['images'], cameras)
        np.testing.assert_array_equal(cameras['CAM_FRONT']['ego2cam'],
                                      np.eye(4))
        self.assertEqual(output['token'], 'sample-token')
        output['img_shapes'][0] = (1, 1)
        self.assertEqual(output['ori_shapes'][0], (3, 5))

    def test_missing_or_corrupt_image(self):
        transform = TRANSFORMS.build(dict(type='LoadMultiViewImagesFromFiles'))
        path = self.root / 'broken.png'
        for exists in (False, True):
            with self.subTest(exists=exists):
                if exists:
                    path.write_bytes(b'not an image')
                with self.assertRaises(RuntimeError) as error:
                    transform(
                        dict(images={'CAM_FRONT': dict(img_path=str(path))}))
                self.assertIn('CAM_FRONT', str(error.exception))
                self.assertIn(str(path), str(error.exception))

    def test_empty_cameras(self):
        transform = TRANSFORMS.build(dict(type='LoadMultiViewImagesFromFiles'))
        with self.assertRaisesRegex(ValueError, 'at least one camera'):
            transform(dict(images={}))


if __name__ == '__main__':
    unittest.main()
