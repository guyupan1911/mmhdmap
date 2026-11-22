import copy
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

TOOL_PATH = (
    Path(__file__).resolve().parents[2] / 'projects' / 'vma' / 'tools' /
    'sd_converter.py')
SPEC = importlib.util.spec_from_file_location('sd_converter', TOOL_PATH)
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class TestSDConverter(unittest.TestCase):

    def test_arc_length_clamping_and_short_trajectory(self):
        windows = list(
            converter.sample_windows(
                [[], [[1, 2]], [[0, 0], [3000, 0], [3000, 4000]]], 8))
        self.assertEqual([item[0] for item in windows], [2] * 8)
        self.assertEqual([item[1] for item in windows], list(range(8)))
        self.assertEqual([item[2] for item in windows],
                         [[0, 0], [500, 0], [1500, 0], [2500, 0], [2500, 500],
                          [2500, 1500], [2500, 2500], [2500, 3500]])
        self.assertEqual([
            item[2] for item in converter.sample_windows(
                [[[5999, 5999], [6000, 6000]]], 2)
        ], [[5000, 5000]] * 2)
        with self.assertRaises(ValueError):
            list(converter.sample_windows([[[0, 0], [float('nan'), 2]]], 7))

    def make_sources(self, root):
        for kind in converter.SAMPLES_PER_TRAJECTORY:
            origin = root / kind / 'origin_data'
            (origin / 'image_data').mkdir(parents=True)
            (origin / 'trajectory_data').mkdir()
            (origin / 'intensity_image').mkdir()
            annotations = []
            for i in range(5):
                name = f'sample_{i}'
                (origin / 'image_data' / f'{name}.jpg').touch()
                (origin / 'intensity_image' / f'{name}.png').touch()
                (origin / 'trajectory_data' / f'{name}.json').write_text(
                    json.dumps([[[3000, 3000], [3001, 3000]]]))
                ann = dict(image_key=name)
                if kind == 'line':
                    ann['curb'] = [
                        dict(
                            data=[[2000, 3000], [4000, 3000]],
                            attrs=dict(attribution='other'))
                    ]
                elif kind == 'box':
                    ann['Crosswalk'] = [
                        dict(
                            data=[2900, 2900, 3100, 3100],
                            attrs=dict(Attribution='none'))
                    ]
                else:
                    ann['Diversion'] = [
                        dict(data=[[2900, 2900], [3100, 2900], [3100, 3100],
                                   [2900, 3100]])
                    ]
                annotations.append(ann)
            (origin / f'{kind}_data_6k.json').write_text(
                json.dumps(annotations))

    def test_all_types_splits_and_safe_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sources(root)
            source = np.broadcast_to(
                np.array([[[20, 100, 200]]], dtype=np.uint8), (6000, 6000, 3))
            intensity = np.broadcast_to(
                np.arange(6000).astype(np.uint8), (6000, 6000))

            def read_source(path, flags):
                return intensity if str(path).endswith('.png') else source

            with patch.object(
                    converter.cv2, 'imread', side_effect=read_source):
                outputs = converter.convert_sd_dataset(root)
            for kind, paths in outputs.items():
                partitions = []
                for split, path in paths.items():
                    with open(path, 'rb') as stream:
                        infos = pickle.load(stream)
                    self.assertTrue(infos['metainfo']['has_annotations'])
                    self.assertEqual(infos['metainfo']['split'], split)
                    samples = infos['data_list']
                    self.assertEqual(
                        len(samples), converter.SAMPLES_PER_TRAJECTORY[kind] *
                        (4 if split == 'train' else 1))
                    partitions.append({s['source_id'] for s in samples})
                    for sample in samples:
                        self.assertEqual(len(sample['instances']), 1)
                        points = np.asarray(sample['instances'][0]['points'])
                        self.assertTrue(
                            np.all((points >= 0) & (points <= 999)))
                    crop = cv2.imread(
                        str(Path(path).parent / samples[0]['img_path']))
                    self.assertEqual(crop.shape, (1000, 1000, 3))
                    intensity_crop = cv2.imread(
                        str(
                            Path(path).parent /
                            samples[0]['intensity_img_path']),
                        cv2.IMREAD_UNCHANGED)
                    left, top = samples[0]['crop_origin']
                    np.testing.assert_array_equal(
                        intensity_crop, intensity[top:top + 1000,
                                                  left:left + 1000])
                    self.assertTrue(infos['metainfo']['has_intensity'])
                    np.testing.assert_allclose(
                        crop[0, 0], source[0, 0], atol=2)
                self.assertFalse(partitions[0] & partitions[1])
            with self.assertRaises(FileExistsError):
                converter.convert_sd_dataset(root)
            old = root / 'prepared/line/sd_image_infos.pkl'
            old.write_bytes(b'old index')
            original = (root /
                        'prepared/line/sd_map_infos_train.pkl').read_bytes()
            with patch.object(converter.cv2, 'imread', return_value=None):
                with self.assertRaises(ValueError):
                    converter.convert_sd_dataset(root, overwrite=True)
            self.assertTrue(old.exists())
            self.assertEqual(original, (old.parent /
                                        'sd_map_infos_train.pkl').read_bytes())
            with patch.object(
                    converter.cv2, 'imread', side_effect=read_source):
                converter.convert_sd_dataset(root, overwrite=True)
            self.assertFalse(old.exists())
            unmanaged = root / 'prepared/notes.txt'
            unmanaged.write_text('keep')
            with self.assertRaises(ValueError):
                converter.convert_sd_dataset(root, overwrite=True)
            self.assertEqual(unmanaged.read_text(), 'keep')

    def test_line_fragments_translation_and_point_contacts(self):
        instances = converter.read_instances(
            dict(curb=[
                dict(
                    data=[[-10, 100], [100, 100], [100, -10], [200, -10],
                          [200, 100]],
                    attrs=dict(attribution='no'))
            ]), 'line')
        result = converter.crop_instances(instances, 'line', [0, 0])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['points'],
                         [[0., 100.], [100., 100.], [100., 0.]])
        self.assertEqual(result[1]['points'], [[200., 0.], [200., 100.]])
        self.assertEqual(result[0]['attrs'], dict(curb_type='unknown'))
        self.assertEqual(result[0]['source_instance_id'],
                         result[1]['source_instance_id'])
        self.assertEqual(
            converter.crop_instances(instances, 'line', [1000, 1000]), [])
        contact = converter.read_instances(
            dict(stop_line=[dict(data=[[-1, -1], [0, 0]])]), 'line')
        self.assertEqual(converter.crop_instances(contact, 'line', [0, 0]), [])

    def test_box_arrow_filter_and_no_input_mutation(self):
        ann = dict(Arrow=[
            dict(
                data=[[[100, 100], [300, 100], [300, 300], [100, 300]]],
                attrs=dict(Attribution=['Turnleft', 'Straight']))
        ])
        original = copy.deepcopy(ann)
        instances = converter.read_instances(ann, 'box')
        self.assertEqual(ann, original)
        full = converter.crop_instances(instances, 'box', [0, 0])
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0]['attrs'],
                         dict(arrow_type='Straight-Turnleft'))
        self.assertEqual(
            converter.crop_instances(instances, 'box', [200, 0]), [])
        ann['Crosswalk'] = ann.pop('Arrow')
        partial = converter.crop_instances(
            converter.read_instances(ann, 'box'), 'box', [200, 0])
        self.assertEqual(len(partial), 1)
        self.assertEqual(len(partial[0]['points']), 5)

    def test_freespace_polygon_split_and_closure(self):
        # U-shaped polygon: the top crop cuts the connecting lower bar.
        points = [[100, 100], [200, 100], [200, 1200], [300, 1200], [300, 100],
                  [400, 100], [400, 1300], [100, 1300]]
        instances = converter.read_instances(
            dict(Diversion=[dict(data=points)]), 'freespace')
        result = converter.crop_instances(instances, 'freespace', [0, 0])
        self.assertEqual(len(result), 2)
        for item in result:
            self.assertEqual(item['points'][0], item['points'][-1])
            self.assertEqual(item['attrs'], {})

    def test_missing_trajectory_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sources(root)
            (root / 'line/origin_data/trajectory_data/sample_0.json').unlink()
            with self.assertRaises(FileNotFoundError):
                converter.convert_sd_dataset(root)
            self.assertFalse((root / 'prepared').exists())


if __name__ == '__main__':
    unittest.main()
