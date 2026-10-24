import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from pyquaternion import Quaternion

# Load the standalone tool without adding the repository to sys.path.
TOOL_PATH = (
    Path(__file__).resolve().parents[2] / 'tools' / 'dataset_converters' /
    'nuscenes_map_converter.py')
SPEC = importlib.util.spec_from_file_location('nuscenes_map_converter',
                                              TOOL_PATH)
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class FakeNuScenes:
    """SDK table fixture with asynchronous cameras and two complete scenes."""

    def __init__(self, root):
        self.version = 'v1.0-mini'
        self.tables = {
            name: {}
            for name in ('sample', 'sample_data', 'calibrated_sensor',
                         'ego_pose', 'log')
        }
        self.scene = []
        self.tables['log']['log'] = dict(location='test-map')
        map_path = root / 'maps/expansion/test-map.json'
        map_path.parent.mkdir(parents=True)
        map_path.touch()
        for scene_idx in range(2):
            tokens = [f's{scene_idx}-f{i}' for i in range(2)]
            scene = dict(
                name=f'scene-{scene_idx}',
                token=f'scene-token-{scene_idx}',
                log_token='log',
                first_sample_token=tokens[0],
                last_sample_token=tokens[-1],
                nbr_samples=len(tokens))
            self.scene.append(scene)
            for index, token in enumerate(tokens):
                sample = dict(
                    token=token,
                    scene_token=scene['token'],
                    timestamp=1000000 + index * 500000,
                    prev=tokens[index - 1] if index else '',
                    next=tokens[index + 1] if index == 0 else '',
                    data={})
                self.tables['sample'][token] = sample
                for channel in ('LIDAR_TOP', *converter.CAMERA_NAMES):
                    sd_token = f'{token}-{channel}'
                    sample['data'][channel] = sd_token
                    is_lidar = channel == 'LIDAR_TOP'
                    filename = f'samples/{channel}/{token}.bin'
                    path = root / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    self.tables['sample_data'][sd_token] = dict(
                        token=sd_token,
                        filename=filename,
                        timestamp=sample['timestamp'] -
                        (0 if is_lidar else 25000),
                        calibrated_sensor_token=channel,
                        ego_pose_token=sd_token,
                        width=1600,
                        height=900)
                    self.tables['calibrated_sensor'][channel] = dict(
                        rotation=Quaternion(axis=[1, 0, 0],
                                            angle=0.2).elements,
                        translation=[1, 0, 2],
                        camera_intrinsic=np.diag([1000, 1000, 1]).tolist())
                    self.tables['ego_pose'][sd_token] = dict(
                        rotation=Quaternion(
                            axis=[0, 0, 1],
                            angle=0.3 if is_lidar else 0.2).elements,
                        translation=[10 if is_lidar else 9, 20, 0])
        # Iteration must follow scene links, not SDK table storage order.
        self.tables['sample'] = dict(
            reversed(list(self.tables['sample'].items())))

    def get(self, table, token):
        return self.tables[table][token]


class TestNuScenesMapConverter(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.nusc = FakeNuScenes(self.root)
        self.official = dict(
            mini_train=['scene-0'],
            mini_val=['scene-1'],
            train=['scene-0'],
            val=['scene-1'],
            test=['scene-0', 'scene-1'])

    def convert(self, **kwargs):
        with patch.object(converter, 'NuScenes', return_value=self.nusc), \
                patch.object(converter, 'create_splits_scenes',
                             return_value=self.official):
            return converter.create_nuscenes_map_infos(
                self.root,
                self.root / 'infos',
                version=self.nusc.version,
                **kwargs)

    def test_standard_format_paths_order_and_timestamps(self):
        outputs = self.convert()
        self.assertEqual(set(outputs), {'train', 'val'})
        infos = load(str(outputs['train']))
        self.assertEqual(infos['metainfo']['timestamp_unit'], 'us')
        self.assertEqual(infos['metainfo']['scene_names'], ['scene-0'])
        samples = infos['data_list']
        self.assertEqual([s['token'] for s in samples], ['s0-f0', 's0-f1'])
        self.assertEqual([s['frame_idx'] for s in samples], [0, 1])
        self.assertEqual(samples[0]['next_token'], samples[1]['token'])
        self.assertEqual(samples[0]['prev_token'], '')
        self.assertEqual(samples[1]['next_token'], '')
        first = samples[0]
        self.assertNotIn('can_bus', first)
        self.assertNotIn('sweeps', first)
        self.assertNotIn('lidar_sweeps', first)
        self.assertEqual(tuple(first['images']), converter.CAMERA_NAMES)
        self.assertEqual(first['lidar_points']['timestamp'], 1000000)
        self.assertEqual(first['images']['CAM_FRONT']['timestamp'], 975000)
        paths = [c['img_path'] for c in first['images'].values()]
        paths.extend([first['lidar_points']['lidar_path'], first['map_path']])
        for path in paths:
            self.assertFalse(Path(path).is_absolute())
            self.assertTrue((self.root / path).is_file())
        # Standard BaseDataset can consume the new file without a custom loader.
        dataset = BaseDataset(
            ann_file=str(outputs['train']), data_prefix={}, pipeline=[])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['token'], first['token'])

    def test_camera_capture_pose_enables_time_compensation(self):
        first = load(str(self.convert()['train']))['data_list'][0]
        camera = first['images']['CAM_FRONT']
        cam2global = camera['ego2global'] @ np.linalg.inv(camera['ego2cam'])
        lidar2global = first['ego2global'] @ first['lidar2ego']
        global2cam = camera['ego2cam'] @ np.linalg.inv(camera['ego2global'])
        lidar2cam = global2cam @ lidar2global
        lidar_point = np.array([5, 1, 0, 1], dtype=np.float64)
        camera_point = lidar2cam @ lidar_point
        np.testing.assert_allclose(
            cam2global @ camera_point, lidar2global @ lidar_point, atol=1e-10)
        static_camera_point = camera['ego2cam'] @ first[
            'lidar2ego'] @ lidar_point
        self.assertFalse(np.allclose(camera_point, static_camera_point))
        self.assertEqual(camera['ego2cam'].dtype, np.float64)

    def test_test_split_is_written_with_all_test_samples(self):
        self.nusc.version = 'v1.0-test'
        outputs = self.convert()
        self.assertEqual(set(outputs), {'test'})
        self.assertEqual(len(load(str(outputs['test']))['data_list']), 4)

    def test_custom_split_validation(self):
        manifest = dict(
            name='custom-mini',
            version='v1.0-mini',
            splits=dict(train=['scene-1'], val=['scene-0']))
        path = self.root / 'split.json'
        path.write_text(json.dumps(manifest))
        output = load(str(self.convert(split_file=path)['train']))
        self.assertEqual(output['metainfo']['split_scheme'], 'custom-mini')
        self.assertEqual(output['data_list'][0]['scene_name'], 'scene-1')
        for invalid, message in (
            (dict(version='v1.0-test'), 'version'),
            (dict(splits=dict(train=['scene-0'], val=['scene-0'])),
             'duplicate'),
            (dict(splits=dict(train=['missing'], val=['scene-1'])),
             'missing')):
            bad = copy.deepcopy(manifest)
            bad.update(invalid)
            path.write_text(json.dumps(bad))
            with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message):
                self.convert(split_file=path)

    def test_missing_file_does_not_write_partial_splits(self):
        (self.root / 'samples/CAM_FRONT/s1-f1.bin').unlink()
        with self.assertRaises(FileNotFoundError):
            self.convert()
        self.assertFalse((self.root / 'infos').exists())

    def test_broken_scene_chain_is_rejected(self):
        self.nusc.tables['sample']['s0-f1']['prev'] = ''
        with self.assertRaisesRegex(ValueError, 'sample links'):
            self.convert()
        self.assertFalse((self.root / 'infos').exists())

    def test_paths_cannot_escape_root_and_outputs_are_protected(self):
        for path in ('/absolute/file.bin', '../outside.bin'):
            with self.assertRaisesRegex(ValueError, 'relative path'):
                converter.checked_relative_path(path, self.root)
        outputs = self.convert()
        before = outputs['train'].read_bytes()
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertEqual(before, outputs['train'].read_bytes())
        self.convert(overwrite=True)


if __name__ == '__main__':
    unittest.main()
