import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from mmengine.dataset import pseudo_collate
from mmengine.fileio import dump
from mmengine.registry import init_default_scope
from PIL import Image

from mmhdmap.datasets import BevMapDataset
from mmhdmap.registry import DATASETS
from mmhdmap.structures import MapDataSample


class TestBevMapDataset(unittest.TestCase):

    def setUp(self):
        init_default_scope('mmhdmap')
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / 'images').mkdir()
        self.image = np.array(
            [[[10, 90, 240], [30, 70, 200]], [[50, 60, 180], [70, 40, 160]]],
            dtype=np.uint8)
        Image.fromarray(self.image).save(self.root / 'images/sample.png')
        self.infos = dict(
            metainfo=dict(
                dataset_type='bev_image_map',
                info_version='1.0',
                image_size=[2, 2],
                coordinate_system='image_pixel',
                has_annotations=False,
                classes=[]),
            data_list=[
                dict(
                    sample_id='source/0',
                    img_path='images/sample.png',
                    height=2,
                    width=2,
                    source_id='source',
                    crop_origin=[100, 200])
            ])
        self.write_infos()

    def write_infos(self):
        dump(self.infos, str(self.root / 'infos.pkl'))

    def build(self, with_annotations=False, **kwargs):
        pipeline = [
            dict(
                type='mmcv.LoadImageFromFile',
                to_float32=False,
                imdecode_backend='pillow',
                color_type='color_ignore_orientation')
        ]
        if with_annotations:
            pipeline.append(dict(type='LoadVectorMapAnnotations'))
        pipeline.append(dict(type='PackBEVMapInputs'))
        return DATASETS.build(
            dict(
                type='BevMapDataset',
                data_root=str(self.root),
                ann_file='infos.pkl',
                test_mode=True,
                pipeline=pipeline,
                **kwargs))

    def test_bgr_pixels_metadata_and_batch(self):
        for serialize_data in (True, False):
            with self.subTest(serialize_data=serialize_data):
                dataset = self.build(serialize_data=serialize_data)
                self.assertIsInstance(dataset, BevMapDataset)
                self.assertTrue(dataset.modality['use_bev_image'])
                self.assertFalse(dataset.modality['use_camera'])
                sample = dataset[0]
                image = sample['inputs']
                self.assertEqual(image.dtype, torch.uint8)
                self.assertTrue(image.is_contiguous())
                np.testing.assert_array_equal(
                    image.numpy(), self.image[..., ::-1].transpose(2, 0, 1))
                data = sample['data_samples']
                self.assertIsInstance(data, MapDataSample)
                self.assertEqual(data.img_shape, (2, 2))
                self.assertEqual(data.crop_origin, [100, 200])
                self.assertNotIn('gt_instances', data)
                self.assertEqual(data.img_path,
                                 str(self.root / 'images/sample.png'))
                data.crop_origin[0] = -1
                self.assertEqual(dataset[0]['data_samples'].crop_origin,
                                 [100, 200])
                batch = pseudo_collate([sample, dataset[0]])
                self.assertEqual(len(batch['data_samples']), 2)

    def test_lazy_subset_and_image_prefix(self):
        second = copy.deepcopy(self.infos['data_list'][0])
        second['sample_id'] = 'source/1'
        self.infos['data_list'].append(second)
        for sample in self.infos['data_list']:
            sample['img_path'] = 'sample.png'
        self.write_infos()
        dataset = self.build(
            lazy_init=True, indices=[1], data_prefix=dict(img_path='images'))
        self.assertFalse(dataset._fully_initialized)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]['data_samples'].sample_id, 'source/1')

    def test_intensity_path_is_metadata_only(self):
        intensity = np.array([[0, 1], [127, 255]], dtype=np.uint8)
        (self.root / 'intensity_image').mkdir()
        path = self.root / 'intensity_image/sample.png'
        Image.fromarray(intensity).save(path)
        self.infos['metainfo']['has_intensity'] = True
        self.infos['data_list'][0][
            'intensity_img_path'] = 'intensity_image/sample.png'
        self.write_infos()
        dataset = self.build()
        output = dataset[0]
        np.testing.assert_array_equal(output['inputs'].numpy(),
                                      self.image[..., ::-1].transpose(2, 0, 1))
        self.assertEqual(output['data_samples'].intensity_img_path, str(path))
        batch = pseudo_collate([output, dataset[0]])
        self.assertEqual(len(batch['inputs']), 2)
        path.unlink()
        # A missing visualization-only file must not affect model inputs.
        np.testing.assert_array_equal(dataset[0]['inputs'].numpy(),
                                      output['inputs'].numpy())
        for value in ('../escape.png', '/absolute.png', ''):
            self.infos['data_list'][0]['intensity_img_path'] = value
            self.write_infos()
            with self.assertRaises(ValueError):
                self.build()

    def test_raw_gt_survives_dataset_and_packing(self):
        self.infos['metainfo'].update(
            has_annotations=True,
            classes=['curb'],
            attribute_schema=dict(
                curb_type=dict(values=['roadside'], applies_to=['curb'])))
        self.infos['data_list'][0]['instances'] = [
            dict(
                label=0,
                points=[[0., 0.], [1., 1.]],
                attrs=dict(curb_type='roadside'),
                source_instance_id=0)
        ]
        self.write_infos()
        dataset = self.build()
        info = dataset.get_data_info(0)
        self.assertEqual(info['instances'],
                         self.infos['data_list'][0]['instances'])
        # Omitting the annotation loader still supports image-only use.
        self.assertNotIn('gt_instances', dataset[0]['data_samples'])
        packed = self.build(with_annotations=True)[0]['data_samples']
        self.assertEqual(packed.gt_instances.labels.tolist(), [0])
        self.assertEqual(packed.gt_instances.labels.dtype, torch.long)
        np.testing.assert_array_equal(packed.gt_instances.polylines[0],
                                      info['instances'][0]['points'])
        self.assertEqual(packed.gt_instances.attributes,
                         [info['instances'][0]['attrs']])

    def test_invalid_protocol_dimensions_and_path(self):
        original = copy.deepcopy(self.infos)
        for field, value in [('dataset_type', 'nuscenes'),
                             ('info_version', '9.0'), ('image_size', [3, 2])]:
            self.infos = copy.deepcopy(original)
            self.infos['metainfo'][field] = value
            self.write_infos()
            with self.assertRaises(ValueError):
                self.build()
        for path in ('../sample.png', '/sample.png'):
            self.infos = copy.deepcopy(original)
            self.infos['data_list'][0]['img_path'] = path
            self.write_infos()
            with self.assertRaises(ValueError):
                self.build()

    def test_actual_size_and_missing_file_by_loader(self):
        image_path = self.root / 'images/sample.png'
        Image.new('RGB', (3, 2)).save(image_path)
        dataset = self.build()
        output = dataset[0]  # Standard loader reports actual image dimensions.
        self.assertEqual(output['data_samples'].ori_shape, (2, 3))
        self.assertEqual(dataset.get_data_info(0)['width'], 2)
        image_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'sample.png'):
            dataset[0]


if __name__ == '__main__':
    unittest.main()
