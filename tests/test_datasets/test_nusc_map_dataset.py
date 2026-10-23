import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mmengine.fileio import dump

from mmhdmap.datasets import NuscMapDataset
from mmhdmap.registry import DATASETS


class TestNuscMapDataset(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.sample = dict(
            token='frame-token',
            scene_name='scene-0001',
            timestamp=1532402927647951,
            location='singapore-onenorth',
            sample_idx=7,
            prev=6,
            next=8,
            e2g_rotation=[1, 0, 0, 0],
            e2g_translation=[10, 20, 0],
            lidar2ego_rotation=[1, 0, 0, 0],
            lidar2ego_translation=[1, 0, 2],
            lidar_path='old/nuscenes/samples/LIDAR_TOP/frame.bin',
            cams={
                name:
                dict(
                    img_fpath=f'old/nuscenes/samples/{name}/frame.jpg',
                    intrinsics=np.eye(3),
                    extrinsics=np.eye(4))
                for name in reversed(NuscMapDataset.CAMERA_NAMES)
            })
        self.ann_file = self.root / 'infos.pkl'
        dump([self.sample], str(self.ann_file))

    def build(self, **kwargs):
        return DATASETS.build(
            dict(
                type='NuscMapDataset',
                data_root=str(self.root),
                ann_file='infos.pkl',
                ann_path_prefix='old/nuscenes',
                pipeline=[],
                **kwargs))

    def test_paths_poses_and_copy(self):
        for serialize_data in (False, True):
            with self.subTest(serialize_data=serialize_data):
                dataset = self.build(serialize_data=serialize_data)
                info = dataset.get_data_info(0)
                self.assertEqual(tuple(info['images']), dataset.CAMERA_NAMES)
                self.assertEqual(
                    info['images']['CAM_FRONT']['img_path'],
                    str(self.root / 'samples/CAM_FRONT/frame.jpg'))
                self.assertEqual(
                    info['map_path'],
                    str(self.root / 'maps/expansion/singapore-onenorth.json'))
                lidar2global = info['ego2global'] @ info['lidar2ego']
                np.testing.assert_allclose(lidar2global[:3, 3], [11, 20, 2])
                self.assertEqual(info['source_sample_idx'], 7)
                self.assertEqual(info['source_prev_idx'], 6)
                self.assertEqual(info['sample_idx'], 0)
                info['ego2global'][0, 3] = -99
                self.assertEqual(dataset[0]['ego2global'][0, 3], 10)

    def test_lazy_subset_and_metainfo(self):
        second = copy.deepcopy(self.sample)
        second.update(token='second-token', sample_idx=8)
        dump([self.sample, second], str(self.ann_file))
        dataset = self.build(
            lazy_init=True,
            indices=[1],
            metainfo=dict(classes=('boundary', 'divider')))
        self.assertFalse(dataset._fully_initialized)
        self.assertEqual(len(dataset), 1)
        info = dataset.get_data_info(0)
        self.assertEqual(info['token'], 'second-token')
        self.assertEqual(info['source_sample_idx'], 8)
        self.assertEqual(info['sample_idx'], 0)
        self.assertEqual(dataset.cat2id, dict(boundary=0, divider=1))

    def test_path_prefix_boundaries(self):
        dataset = self.build(lazy_init=True)
        resolve = dataset._resolve_sensor_path
        self.assertEqual(
            resolve('/external/frame.jpg', 'img_path'), '/external/frame.jpg')
        self.assertEqual(
            resolve('samples/frame.jpg', 'img_path'),
            str(self.root / 'samples/frame.jpg'))
        self.assertEqual(
            resolve('old/nuscenes_extra/frame.jpg', 'img_path'),
            str(self.root / 'old/nuscenes_extra/frame.jpg'))

    def test_invalid_annotations(self):
        dump(dict(data_list=[self.sample]), str(self.ann_file))
        with self.assertRaisesRegex(TypeError, 'list of sample dicts'):
            self.build()
        self.sample['cams']['CAM_FRONT']['intrinsics'] = np.eye(4)
        dump([self.sample], str(self.ann_file))
        with self.assertRaisesRegex(ValueError, 'CAM_FRONT'):
            self.build()


if __name__ == '__main__':
    unittest.main()
