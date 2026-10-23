import unittest

import numpy as np
import torch
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData, PixelData

from mmhdmap.datasets.transforms import PackMapInputs
from mmhdmap.structures import MapDataSample


class TestPackMapInputs(unittest.TestCase):

    def setUp(self):
        init_default_scope('mmhdmap')

    def test_camera_order_layout_and_metadata(self):
        # Include a negative-stride view and cameras with different sizes.
        first = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)[:, ::-1]
        second = np.ones((4, 5, 3), dtype=np.uint8)
        results = dict(
            imgs=[first, second],
            camera_names=['CAM_BACK', 'CAM_FRONT'],
            token='sample-token',
            images=dict(CAM_BACK=dict(cam2img=np.eye(3))),
            ego2global=np.eye(4),
            ori_shapes=[(2, 3), (4, 5)],
            img_shapes=[(2, 3), (4, 5)])
        pipeline = Compose([dict(type='PackMapInputs')])
        output = pipeline(results)
        self.assertEqual(set(output), {'inputs', 'data_samples'})
        sample = output['data_samples']
        self.assertIsInstance(sample, MapDataSample)
        self.assertEqual(sample.camera_names, ['CAM_BACK', 'CAM_FRONT'])
        self.assertNotIn('imgs', sample)
        self.assertNotIn('gt_instances', sample)
        self.assertNotIn('gt_sem_seg', sample)
        for tensor, image in zip(output['inputs']['imgs'], results['imgs']):
            self.assertEqual(tensor.dtype, torch.uint8)
            self.assertTrue(tensor.is_contiguous())
            np.testing.assert_array_equal(tensor.numpy(),
                                          image.transpose(2, 0, 1))
        sample.images['CAM_BACK']['cam2img'][0, 0] = 42
        self.assertEqual(results['images']['CAM_BACK']['cam2img'][0, 0], 1)
        sample.camera_names.append('NEW_CAMERA')
        self.assertEqual(len(results['camera_names']), 2)

        batch = pseudo_collate([output, pipeline(results)])
        self.assertEqual(len(batch['data_samples']), 2)
        self.assertTrue(
            all(
                isinstance(item, MapDataSample)
                for item in batch['data_samples']))

    def test_existing_ground_truth_and_custom_metadata(self):
        instances = InstanceData(labels=torch.tensor([1]))
        semantic = PixelData(sem_seg=torch.zeros(3, 10, 20))
        output = PackMapInputs(meta_keys=('token', ))(
            dict(
                imgs=[np.zeros((2, 3, 3), dtype=np.float32)],
                camera_names=['CAM_FRONT'],
                token='sample-token',
                gt_instances=instances,
                gt_sem_seg=semantic))
        sample = output['data_samples']
        self.assertIs(sample.gt_instances, instances)
        self.assertIs(sample.gt_sem_seg, semantic)
        self.assertEqual(sample.metainfo, dict(token='sample-token'))
        self.assertEqual(output['inputs']['imgs'][0].dtype, torch.float32)

    def test_invalid_images(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        for images, names in (
            ([], []),
            ([image], []),
            ([image, image], ['CAM_FRONT', 'CAM_FRONT']),
            ([np.zeros((2, 3))], ['CAM_FRONT']),
        ):
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    PackMapInputs()(dict(imgs=images, camera_names=names))


if __name__ == '__main__':
    unittest.main()
