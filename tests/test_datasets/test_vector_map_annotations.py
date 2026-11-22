import copy
import unittest

import numpy as np
import torch

from mmhdmap.datasets.transforms import LoadVectorMapAnnotations, PackBEVMapInputs


class TestVectorMapAnnotations(unittest.TestCase):

    def test_variable_length_instances_and_independent_storage(self):
        raw = [
            dict(
                label=0,
                points=[[1.25, 2.5], [3, 4]],
                attrs={'direction': 'unknown'}),
            dict(label=2, points=[[0, 0], [2, 0], [2, 2], [0, 0]], attrs={})
        ]
        original = copy.deepcopy(raw)
        results = LoadVectorMapAnnotations()(
            dict(instances=raw, img=np.zeros((10, 20, 3), dtype=np.uint8)))
        output = PackBEVMapInputs()(results)
        gt = output['data_samples'].gt_instances
        self.assertEqual(gt.labels.dtype, torch.long)
        self.assertEqual(gt.labels.tolist(), [0, 2])
        self.assertEqual([p.shape for p in gt.polylines], [(2, 2), (4, 2)])
        self.assertEqual(gt.polylines[0].dtype, np.float32)
        np.testing.assert_array_equal(gt.polylines[0], original[0]['points'])
        self.assertEqual(gt.attributes, [i['attrs'] for i in original])
        gt.polylines[0][0] = -1
        gt.attributes[0]['direction'] = 'changed'
        self.assertEqual(raw, original)

    def test_empty_and_unannotated_are_distinct(self):
        gt = LoadVectorMapAnnotations()(dict(instances=[]))['gt_instances']
        self.assertEqual(len(gt), 0)
        self.assertEqual(tuple(gt.labels.shape), (0, ))
        self.assertEqual(gt.polylines, [])
        self.assertEqual(gt.attributes, [])
        with self.assertRaises(KeyError):
            LoadVectorMapAnnotations()({})
        sample = PackBEVMapInputs()(
            dict(img=np.zeros((2, 3, 3), dtype=np.uint8)))
        self.assertNotIn('gt_instances', sample['data_samples'])

    def test_invalid_annotations(self):
        valid = dict(label=0, points=[[0, 0], [1, 1]], attrs={})
        for field, value in [('label', -1), ('label', True),
                             ('points', [[0, 0]]), ('points', [[0, 0], [0,
                                                                        0]]),
                             ('points', [[0, 0], [float('nan'), 1]]),
                             ('points', [[0, 0, 0], [1, 1, 1]])]:
            with self.subTest(field=field, value=value):
                instance = dict(valid, **{field: value})
                with self.assertRaises(ValueError):
                    LoadVectorMapAnnotations()(dict(instances=[instance]))
        with self.assertRaises(TypeError):
            LoadVectorMapAnnotations()(
                dict(instances=[dict(valid, attrs='solid')]))


if __name__ == '__main__':
    unittest.main()
