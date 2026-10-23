import unittest

import torch
from mmengine.structures import InstanceData, PixelData

from mmhdmap.structures import MapDataSample


class TestMapDataSample(unittest.TestCase):

    def test_optional_typed_fields(self):
        sample = MapDataSample(metainfo=dict(token='sample-token'))
        self.assertEqual(sample.token, 'sample-token')
        fields = dict(
            gt_instances=InstanceData,
            pred_instances=InstanceData,
            gt_sem_seg=PixelData,
            pred_sem_seg=PixelData)
        for name, field_type in fields.items():
            with self.subTest(field=name):
                self.assertNotIn(name, sample)
                with self.assertRaises(AssertionError):
                    setattr(sample, name, {})
                value = field_type()
                setattr(sample, name, value)
                self.assertIs(getattr(sample, name), value)
                self.assertIn(name, sample.keys())
                delattr(sample, name)
                self.assertNotIn(name, sample)

    def test_nested_tensor_conversion(self):
        sample = MapDataSample(metainfo=dict(token='sample-token'))
        sample.gt_instances = InstanceData(
            labels=torch.tensor([0, 2]), polylines=torch.ones(2, 20, 2))
        sample.gt_sem_seg = PixelData(sem_seg=torch.ones(3, 10, 20))
        converted = sample.to(dtype=torch.float64)
        self.assertEqual(converted.gt_instances.polylines.dtype, torch.float64)
        self.assertEqual(converted.gt_sem_seg.sem_seg.dtype, torch.float64)
        self.assertEqual(sample.gt_instances.polylines.dtype, torch.float32)
        self.assertEqual(converted.token, 'sample-token')


if __name__ == '__main__':
    unittest.main()
