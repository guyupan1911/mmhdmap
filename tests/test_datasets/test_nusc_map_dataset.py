import copy
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from mmengine.fileio import dump
from mmengine.registry import init_default_scope

from mmhdmap.datasets import NuscMapDataset
from mmhdmap.registry import DATASETS
from mmhdmap.structures import MapDataSample


class TestNuscMapDataset(unittest.TestCase):

    def setUp(self):
        init_default_scope('mmhdmap')
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        ego2global = np.eye(4)
        ego2global[:3, 3] = [10, 20, 0]
        lidar2ego = np.eye(4)
        lidar2ego[:3, 3] = [1, 0, 2]
        camera_pose = ego2global.copy()
        camera_pose[0, 3] -= 0.5
        self.sample = dict(
            token='frame-token',
            scene_token='scene-token',
            scene_name='scene-0001',
            frame_idx=7,
            timestamp=1532402927647951,
            prev_token='previous-token',
            next_token='next-token',
            location='singapore-onenorth',
            map_path='maps/expansion/singapore-onenorth.json',
            ego2global=ego2global,
            lidar2ego=lidar2ego,
            lidar_points=dict(
                lidar_path='samples/LIDAR_TOP/frame.bin',
                sample_data_token='lidar-token',
                timestamp=1532402927647951,
                num_pts_feats=5),
            images={
                name:
                dict(
                    img_path=f'samples/{name}/frame.jpg',
                    sample_data_token=f'{name}-token',
                    timestamp=1532402927612460,
                    width=3,
                    height=2,
                    cam2img=np.eye(3),
                    ego2cam=np.eye(4),
                    ego2global=camera_pose.copy())
                for name in reversed(NuscMapDataset.CAMERA_NAMES)
            })
        self.ann_file = self.root / 'infos.pkl'
        self.write_infos([self.sample])

    def write_infos(self, samples, version='1.0'):
        dump(
            dict(
                metainfo=dict(info_version=version, timestamp_unit='us'),
                data_list=samples), str(self.ann_file))

    def build(self, **kwargs):
        return DATASETS.build(
            dict(
                type='NuscMapDataset',
                data_root=str(self.root),
                ann_file='infos.pkl',
                pipeline=kwargs.pop('pipeline', []),
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
                self.assertEqual(
                    info['lidar_points']['lidar_path'],
                    str(self.root / 'samples/LIDAR_TOP/frame.bin'))
                lidar2global = info['ego2global'] @ info['lidar2ego']
                np.testing.assert_allclose(lidar2global[:3, 3], [11, 20, 2])
                camera = info['images']['CAM_FRONT']
                np.testing.assert_allclose(camera['ego2global'][:3, 3],
                                           [9.5, 20, 0])
                self.assertEqual(camera['timestamp'], 1532402927612460)
                self.assertEqual(camera['sample_data_token'],
                                 'CAM_FRONT-token')
                self.assertEqual(camera['width'], 3)
                self.assertEqual(info['frame_idx'], 7)
                self.assertEqual(info['prev_token'], 'previous-token')
                self.assertEqual(info['sample_idx'], 0)
                info['ego2global'][0, 3] = -99
                camera['ego2global'][0, 3] = -99
                self.assertEqual(dataset[0]['ego2global'][0, 3], 10)
                self.assertEqual(
                    dataset[0]['images']['CAM_FRONT']['ego2global'][0, 3], 9.5)
                self.assertEqual(dataset.metainfo['timestamp_unit'], 'us')

    def test_lazy_subset_and_metainfo(self):
        second = copy.deepcopy(self.sample)
        second.update(token='second-token', frame_idx=8)
        self.write_infos([self.sample, second])
        dataset = self.build(
            lazy_init=True,
            indices=[1],
            metainfo=dict(classes=('boundary', 'divider')))
        self.assertFalse(dataset._fully_initialized)
        self.assertEqual(len(dataset), 1)
        info = dataset.get_data_info(0)
        self.assertEqual(info['token'], 'second-token')
        self.assertEqual(info['frame_idx'], 8)
        self.assertEqual(info['sample_idx'], 0)
        self.assertEqual(info['prev_token'], 'previous-token')
        self.assertEqual(dataset.cat2id, dict(boundary=0, divider=1))

    def test_data_prefix_and_relative_paths(self):
        dataset = self.build(
            data_prefix=dict(img_path='cameras', map_path='assets'))
        info = dataset[0]
        self.assertEqual(
            info['images']['CAM_FRONT']['img_path'],
            str(self.root / 'cameras/samples/CAM_FRONT/frame.jpg'))
        self.assertEqual(
            info['map_path'],
            str(self.root / 'assets/maps/expansion/singapore-onenorth.json'))
        for filename in ('/absolute/frame.jpg', '../frame.jpg'):
            with self.assertRaisesRegex(ValueError, 'relative'):
                dataset._resolve_path(filename, 'img_path')

    def test_real_loading_and_packing_preserves_capture_metadata(self):
        for camera in self.sample['images'].values():
            path = self.root / camera['img_path']
            path.parent.mkdir(parents=True)
            cv2.imwrite(str(path), np.full((2, 3, 3), 100, dtype=np.uint8))
        dataset = self.build(pipeline=[
            dict(type='LoadMultiViewImagesFromFiles'),
            dict(type='PackMapInputs')
        ])
        output = dataset[0]
        sample = output['data_samples']
        self.assertIsInstance(sample, MapDataSample)
        self.assertEqual(len(output['inputs']['imgs']), 6)
        self.assertEqual(tuple(output['inputs']['imgs'][0].shape), (3, 2, 3))
        for key in ('scene_token', 'frame_idx', 'prev_token', 'next_token'):
            self.assertEqual(sample.metainfo[key], self.sample[key])
        self.assertEqual(sample.images['CAM_FRONT']['timestamp'],
                         1532402927612460)
        np.testing.assert_array_equal(
            sample.images['CAM_FRONT']['ego2global'],
            self.sample['images']['CAM_FRONT']['ego2global'])
        self.assertEqual(sample.lidar_points['num_pts_feats'], 5)
        self.assertNotIn('gt_sem_seg', sample)

    def test_invalid_annotations(self):
        dump([self.sample], str(self.ann_file))
        with self.assertRaisesRegex(TypeError, 'should be a dict'):
            self.build()
        self.write_infos([self.sample], version='2.0')
        with self.assertRaisesRegex(ValueError, 'info_version'):
            self.build()
        self.sample['images']['CAM_FRONT']['cam2img'] = np.eye(4)
        self.write_infos([self.sample])
        with self.assertRaisesRegex(ValueError, 'CAM_FRONT/cam2img'):
            self.build()


if __name__ == '__main__':
    unittest.main()
