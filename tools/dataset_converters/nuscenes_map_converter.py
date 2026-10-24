"""Create portable MMEngine infos for nuScenes map perception.

Only sensor metadata and map references are exported. Images, point clouds,
map geometry, CAN bus messages and LiDAR sweeps are not loaded.
"""

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np
from mmengine.fileio import dump
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

CAMERA_NAMES = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
VERSIONS = ('v1.0-mini', 'v1.0-trainval', 'v1.0-test')
INFO_VERSION = '1.0'


def pose_matrix(record: dict) -> np.ndarray:
    """Convert an SDK pose/calibration to a column-vector transform."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    matrix[:3, 3] = record['translation']
    if not np.isfinite(matrix).all():
        raise ValueError('Pose contains non-finite values')
    return matrix


def checked_relative_path(filename: str, data_root: Path) -> str:
    """Validate an SDK filename without embedding the machine's data root."""
    path = PurePosixPath(filename)
    if (not filename or path.is_absolute() or '..' in path.parts
            or '\\' in filename):
        raise ValueError(f'Expected a data-root-relative path: {filename!r}')
    if not (data_root / path).is_file():
        raise FileNotFoundError(data_root / path)
    return path.as_posix()


def resolve_scene_splits(nusc, split_file=None):
    """Return a split name and validated scene-name lists.

    A custom JSON contains ``name``, ``version`` and ``splits``. The latter
    maps train/val (or test for v1.0-test) to scene-name lists. Scenes omitted
    from a custom manifest are deliberately excluded.
    """
    if nusc.version not in VERSIONS:
        raise ValueError(f'Unsupported nuScenes version: {nusc.version}')
    expected = ('test', ) if nusc.version == 'v1.0-test' else ('train', 'val')
    if split_file is None:
        official = create_splits_scenes()
        prefix = 'mini_' if nusc.version == 'v1.0-mini' else ''
        scene_splits = {key: official[prefix + key] for key in expected}
        name = 'official'
    else:
        with open(split_file, encoding='utf-8') as stream:
            manifest = json.load(stream)
        if manifest['version'] != nusc.version:
            raise ValueError(
                'Split manifest version does not match the dataset')
        name = manifest['name']
        scene_splits = manifest['splits']
        if not isinstance(name, str) or not name or name == 'official':
            raise ValueError(
                'Custom split name must be nonempty and not official')
    if set(scene_splits) != set(expected):
        raise ValueError(
            f'Expected splits {expected}, got {tuple(scene_splits)}')

    available = {scene['name'] for scene in nusc.scene}
    selected = set()
    for split, names in scene_splits.items():
        if (not isinstance(names, list) or not names
                or not all(isinstance(name, str) for name in names)):
            raise ValueError(
                f'{split}: expected a nonempty list of scene names')
        if len(names) != len(set(names)) or selected.intersection(names):
            raise ValueError(
                f'{split}: duplicate scenes within or across splits')
        missing = set(names) - available
        if missing:
            raise ValueError(f'{split}: missing scenes: {sorted(missing)}')
        selected.update(names)
    return name, scene_splits


def iter_scene_samples(nusc, scene):
    """Follow SDK sample links, validating scene boundaries and frame order."""
    token = scene['first_sample_token']
    previous = ''
    previous_timestamp = None
    seen = set()
    while token:
        if token in seen:
            raise ValueError(f'{scene["name"]}: cycle in sample links')
        sample = nusc.get('sample', token)
        if (sample['scene_token'] != scene['token']
                or sample['prev'] != previous):
            raise ValueError(f'{scene["name"]}: inconsistent sample links')
        if (previous_timestamp is not None
                and sample['timestamp'] <= previous_timestamp):
            raise ValueError(f'{scene["name"]}: non-increasing timestamps')
        seen.add(token)
        yield sample
        previous = token
        previous_timestamp = sample['timestamp']
        token = sample['next']
    if (len(seen) != scene['nbr_samples']
            or previous != scene['last_sample_token']):
        raise ValueError(f'{scene["name"]}: incomplete sample chain')


def build_sample_info(nusc, sample, scene, frame_idx, data_root):
    """Collect one keyframe and all six cameras at their own capture times.

    Top-level ego2global is the LIDAR_TOP-time ego pose. Each camera's
    ego2global is its own capture-time ego pose; ego2cam remains a static
    extrinsic. All timestamps are integer microseconds and poses use meters.
    """
    lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_calibration = nusc.get('calibrated_sensor',
                                 lidar['calibrated_sensor_token'])
    lidar_pose = nusc.get('ego_pose', lidar['ego_pose_token'])
    location = nusc.get('log', scene['log_token'])['location']
    images = {}
    for name in CAMERA_NAMES:
        camera = nusc.get('sample_data', sample['data'][name])
        calibration = nusc.get('calibrated_sensor',
                               camera['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', camera['ego_pose_token'])
        intrinsic = np.array(calibration['camera_intrinsic'], dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(f'{sample["token"]}/{name}: invalid intrinsics')
        images[name] = dict(
            img_path=checked_relative_path(camera['filename'], data_root),
            sample_data_token=camera['token'],
            timestamp=int(camera['timestamp']),
            width=int(camera['width']),
            height=int(camera['height']),
            cam2img=intrinsic,
            ego2cam=np.linalg.inv(pose_matrix(calibration)),
            ego2global=pose_matrix(pose))
    return dict(
        token=sample['token'],
        scene_token=scene['token'],
        scene_name=scene['name'],
        frame_idx=frame_idx,
        timestamp=int(sample['timestamp']),
        prev_token=sample['prev'],
        next_token=sample['next'],
        location=location,
        map_path=checked_relative_path(f'maps/expansion/{location}.json',
                                       data_root),
        ego2global=pose_matrix(lidar_pose),
        lidar2ego=pose_matrix(lidar_calibration),
        lidar_points=dict(
            lidar_path=checked_relative_path(lidar['filename'], data_root),
            sample_data_token=lidar['token'],
            timestamp=int(lidar['timestamp']),
            num_pts_feats=5),
        images=images)


def create_nuscenes_map_infos(data_root,
                              out_dir,
                              version='v1.0-mini',
                              info_prefix='mmhdmap_nuscenes',
                              split_file=None,
                              overwrite=False):
    """Validate metadata/files and write one standard info PKL per split.

    Missing files or scenes are errors rather than silently changing the
    benchmark split. A custom split manifest can explicitly select a subset.
    No sensor payload, map geometry, CAN bus or sweep data is read.
    """
    if version not in VERSIONS:
        raise ValueError(f'Unsupported nuScenes version: {version}')
    if (not info_prefix or Path(info_prefix).name != info_prefix
            or info_prefix in ('.', '..') or '\\' in info_prefix):
        raise ValueError('info_prefix must be a filename prefix')
    data_root = Path(data_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=True)
    split_name, scene_splits = resolve_scene_splits(nusc, split_file)
    version_tag = version[len('v1.0-'):]
    output_paths = {
        split: out_dir / f'{info_prefix}_{version_tag}_infos_{split}.pkl'
        for split in scene_splits
    }
    for path in output_paths.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f'{path} already exists; use --overwrite')

    scenes = {scene['name']: scene for scene in nusc.scene}
    outputs = {}
    seen_tokens = set()
    for split, names in scene_splits.items():
        data_list = []
        for name in sorted(names):
            scene = scenes[name]
            for frame_idx, sample in enumerate(
                    iter_scene_samples(nusc, scene)):
                if sample['token'] in seen_tokens:
                    raise ValueError(
                        f'Duplicate sample token: {sample["token"]}')
                seen_tokens.add(sample['token'])
                data_list.append(
                    build_sample_info(nusc, sample, scene, frame_idx,
                                      data_root))
        outputs[split] = dict(
            metainfo=dict(
                dataset='nuScenes',
                dataset_version=version,
                info_version=INFO_VERSION,
                split=split,
                split_scheme=split_name,
                scene_names=sorted(names),
                timestamp_unit='us',
                reference_sensor='LIDAR_TOP',
                camera_names=CAMERA_NAMES),
            data_list=data_list)

    # Finish validation of every split before writing any output.
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, output in outputs.items():
        path = output_paths[split]
        dump(output, str(path))
        print(f'{split}: {len(scene_splits[split])} scenes, '
              f'{len(output["data_list"])} samples -> {path}')
    return output_paths


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Create nuScenes map infos without CAN bus or sweeps')
    parser.add_argument(
        '--data-root', required=True, help='nuScenes data root')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--version', choices=VERSIONS, default='v1.0-mini')
    parser.add_argument('--info-prefix', default='mmhdmap_nuscenes')
    parser.add_argument(
        '--split-file', help='Custom scene split JSON manifest')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    """Create the requested infos."""
    args = parse_args()
    create_nuscenes_map_infos(**vars(args))


if __name__ == '__main__':
    main()
