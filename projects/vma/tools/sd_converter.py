"""Convert SD images and vector GT using the original VMA crop rules."""

import json
import pickle
import tempfile
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import (GeometryCollection, LineString, MultiLineString,
                              MultiPolygon, Polygon, box)

SAMPLES_PER_TRAJECTORY = dict(line=7, box=7, freespace=14)
SOURCE_SIZE = 6000
CROP_SIZE = 1000
CLASSES = dict(
    line=['lane', 'curb', 'stopline'],
    box=[
        'Crosswalk', 'SpeedBump', 'NoParkingLine', 'Arrow', 'Diamond_marking',
        'ExclusiveLaneSign', 'InvertedTriangleMarking'
    ],
    freespace=['Diversion'])
LINE_ATTRIBUTES = dict(
    lane_direction=['unidirectional', 'bidirectional', 'unknown'],
    lane_type=[
        'solid', 'dotted', 'solid_fishbone', 'ldotted_fishbone', 'unknown',
        'no'
    ],
    lane_properties=[
        'general', 'stay', 'tide', 'bus', 'three_color', 'unknown'
    ],
    lane_flag=['single', 'double', 'triple', 'unknown'],
    lane_width=['normal', 'wide', 'unknown'],
    curb_type=[
        'groundside', 'roadside', 'cone', 'water_horse', 'guardrail', 'unknown'
    ])
ARROW_ATTRIBUTES = [
    'Mergeright', 'Mergeleft', 'Turnright', 'Turnleft', 'Straight', 'Turnoff',
    'Straight-Turnleft', 'Straight-Turnright', 'Turnright-Turnleft', 'Other',
    'Turnoff-Turnleft', 'Turnright-Straight-Turnleft', 'Turnoff-Straight',
    'Turnright-Turnoff', 'Turnright-Other-Straight-Turnleft', 'Other-Turnleft',
    'No_Turnoff', 'Straight-Mergeleft', 'No_Straight-Straight'
]


def attribute_schema(kind):
    if kind == 'line':
        return {
            key: dict(values=values, applies_to=[key.split('_')[0]])
            for key, values in LINE_ATTRIBUTES.items()
        }
    if kind == 'box':
        return dict(
            arrow_type=dict(values=ARROW_ATTRIBUTES, applies_to=['Arrow']),
            exclusive_lane_type=dict(
                values=['BikeLane', 'BusLane'],
                applies_to=['ExclusiveLaneSign']))
    return {}


def _unknown(value):
    return 'unknown' if value in ('no', 'unknown', 'other') else value


def _line_attrs(attrs, parsing=False):
    direction = (('unknown' if attrs['type'] == 'lane_unknown' else
                  'unidirectional') if parsing else attrs['direction'])
    # Retain legacy's unidirectional value even for parsing/bidirectional_lane.
    result = dict(lane_direction=direction)
    for key in ('type', 'properties', 'color', 'flag', 'width'):
        raw_key = f'lane_{key}' if parsing and key not in ('flag',
                                                           'width') else key
        result[f'lane_{key}'] = _unknown(attrs[raw_key])
    return result


def read_instances(annotation, kind):
    """Normalize raw annotations without resampling or mutating the JSON."""
    records = []
    if kind == 'line' and 'parsing' in annotation:
        for instance in annotation['parsing']:
            attrs = instance['attrs']
            raw_class = attrs['type']
            if raw_class in ('lane_marking', 'bidirectional_lane',
                             'lane_unknown'):
                records.append(
                    ('lane', instance['data'], _line_attrs(attrs, True)))
            elif raw_class == 'Road_teeth':
                records.append(('curb', instance['data'],
                                dict(curb_type=_unknown(attrs['curb']))))
            elif raw_class == 'Stop_Line':
                records.append(('stopline', instance['data'], {}))
    else:
        if kind == 'box' and 'parsing' in annotation:
            raise ValueError('Legacy SD box parsing format is not supported')
        if kind == 'box' and annotation.get('StopLine'):
            raise ValueError(
                'StopLine is absent from the legacy box class vocabulary')
        wanted = ['lane', 'curb', 'stop_line'
                  ] if kind == 'line' else CLASSES[kind]
        for name, instances in annotation.items():
            if name not in wanted:
                continue
            for instance in instances:
                attrs = {}
                label = name
                if kind == 'line':
                    label = 'stopline' if name == 'stop_line' else name
                    if name != 'curb' and 'segments' in instance:
                        points = [
                            p['data']
                            for p in instance['segments'][0]['points']
                        ]
                    else:
                        points = instance['data']
                    if name == 'lane':
                        attrs = _line_attrs(
                            instance.get('attrs', instance.get('attr')))
                    elif name == 'curb':
                        attrs = dict(
                            curb_type=_unknown(instance['attrs']
                                               ['attribution']))
                elif kind == 'box':
                    data = instance['data']
                    if isinstance(data[0], list):
                        points = data[0] + [data[0][0]]
                    else:
                        x1, y1, x2, y2 = data
                        points = [[x1, y1], [x1, y2], [x2, y2], [x2, y1],
                                  [x1, y1]]
                    value = instance['attrs']['Attribution']
                    if name == 'Arrow':
                        candidates = {
                            frozenset(v.split('-')): v
                            for v in ARROW_ATTRIBUTES
                        }
                        key = frozenset(value) if isinstance(
                            value, list) else frozenset([value])
                        if key not in candidates:
                            raise ValueError(
                                f'Unknown Arrow attribute: {value}')
                        attrs = dict(arrow_type=candidates[key])
                    elif name == 'ExclusiveLaneSign':
                        attrs = dict(exclusive_lane_type=value)
                else:
                    points = instance['data']
                records.append((label, points, attrs))
    normalized = []
    for index, (label, points, attrs) in enumerate(records):
        points = np.asarray([p[:2] for p in points], dtype=np.float64)
        if points.size == 0:
            continue
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(
                points).all():
            raise ValueError(f'{label}: expected finite 2D vertices')
        for key, schema in attribute_schema(kind).items():
            if key in attrs and attrs[key] not in schema['values']:
                raise ValueError(f'{label}: invalid {key}={attrs[key]}')
        normalized.append(
            dict(
                label=CLASSES[kind].index(label),
                points=points,
                attrs=attrs,
                source_instance_id=index))
    return normalized


def sample_windows(trajectories, sample_num):
    """Yield trajectory index, sample index and integer [left, top]."""
    for trajectory_index, trajectory in enumerate(trajectories):
        if len(trajectory) < 2:
            continue
        points = np.asarray(trajectory, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2
                or not np.isfinite(points).all()):
            raise ValueError('Trajectory must contain finite 2D points')
        line = LineString(points)
        for sample_index, distance in enumerate(
                np.linspace(0, line.length, sample_num)):
            x, y = line.interpolate(distance).coords[0]
            x, y = min(max(x, 500), 5500), min(max(y, 500), 5500)
            yield trajectory_index, sample_index, [
                int(round(x) - 500), int(round(y) - 500)
            ]


def crop_instances(instances, kind, origin):
    """Intersect GT with pixel-center bounds [left, top, left+999, top+999]."""
    left, top = origin
    window = box(left, top, left + CROP_SIZE - 1, top + CROP_SIZE - 1)
    cropped = []
    for instance in instances:
        points = instance['points']
        if len(points) < 2:
            continue
        closed = np.array_equal(points[0], points[-1])
        if closed and len(points) == 2:
            continue
        geometry = Polygon(
            points) if closed or kind == 'freespace' else LineString(points)
        if kind == 'line' and closed:
            geometry = geometry.buffer(0.001)  # Legacy closed-line behavior.
        intersection = geometry.intersection(window)
        parts = []
        if kind == 'box':
            if isinstance(intersection, Polygon) and not intersection.is_empty:
                coords = np.asarray(intersection.exterior.coords)
                if len(coords) != 5:
                    continue
                if CLASSES[kind][instance['label']] == 'Arrow':
                    if not np.array_equal(coords, points):
                        coords = coords[:-1][::-1]
                        coords = np.concatenate([coords, coords[:1]])
                        if not np.array_equal(coords, points):
                            continue
                parts.append(coords)
        else:
            components = (
                list(intersection.geoms) if isinstance(
                    intersection, (MultiLineString, MultiPolygon,
                                   GeometryCollection)) else [intersection])
            for component in components:
                if component.is_empty:
                    continue
                if kind == 'line' and isinstance(component, LineString):
                    parts.append(np.asarray(component.coords))
                elif isinstance(component, Polygon):
                    # Legacy line GeometryCollection only keeps LineStrings.
                    if kind == 'line' and isinstance(intersection,
                                                     GeometryCollection):
                        continue
                    coords = np.asarray(component.exterior.coords)
                    parts.append(coords[:-1] if kind == 'line' else coords)
                    # For freespace GeometryCollection use component.exterior;
                    # legacy incorrectly accesses collection.exterior.
        for coords in parts:
            local = coords - np.asarray(origin)
            cropped.append(
                dict(
                    label=instance['label'],
                    points=local.tolist(),
                    attrs=dict(instance['attrs']),
                    source_instance_id=instance['source_instance_id']))
    return cropped


def _discover_sources(data_root):
    sources = {}
    for kind in SAMPLES_PER_TRAJECTORY:
        origin = data_root / kind / 'origin_data'
        if not (data_root / kind).exists():
            continue
        with (origin / f'{kind}_data_6k.json').open() as stream:
            annotations = json.load(stream)
        names = [ann['image_key'] for ann in annotations]
        if len(names) != len(set(names)) or any(
                Path(n).name != n for n in names):
            raise ValueError(f'{kind}: duplicate or unsafe image_key')
        images = {p.stem for p in (origin / 'image_data').glob('*.jpg')}
        if not images or images != set(names):
            raise ValueError(
                f'{kind}: image files and annotation image_keys differ')
        for name in names:
            trajectory = origin / 'trajectory_data' / f'{name}.json'
            if not trajectory.is_file():
                raise FileNotFoundError(trajectory)
            intensity = origin / 'intensity_image' / f'{name}.png'
            if not intensity.is_file():
                raise FileNotFoundError(
                    f'{intensity}; run projects/vma/tools/recover_sd_intensity.py first'
                )
        sources[kind] = annotations
    if not sources:
        raise FileNotFoundError(f'No line/box/freespace data in {data_root}')
    return sources


def _check_output(out_dir, overwrite):
    if not out_dir.exists():
        return
    if not out_dir.is_dir() or (any(out_dir.iterdir()) and not overwrite):
        raise FileExistsError(f'Output exists; use --overwrite: {out_dir}')
    # Replacement is restricted to the converter's generated layout.
    for path in out_dir.rglob('*'):
        relative = path.relative_to(out_dir).parts
        allowed = (
            relative[0] in SAMPLES_PER_TRAJECTORY
            and ((len(relative) == 1 and path.is_dir()) or
                 (len(relative) == 2 and relative[1] in
                  ('images', 'intensity_image') and path.is_dir()) or
                 (len(relative) == 2 and path.is_file() and relative[1] in
                  ('sd_image_infos.pkl', 'sd_map_infos_train.pkl',
                   'sd_map_infos_val.pkl')) or
                 (len(relative) == 3 and relative[1] == 'images'
                  and path.suffix == '.jpg' and path.is_file()) or
                 (len(relative) == 3 and relative[1] == 'intensity_image'
                  and path.suffix == '.png' and path.is_file())))
        if path.is_symlink() or not allowed:
            raise ValueError(f'Refusing to replace unmanaged output: {path}')


def convert_sd_dataset(data_root, out_dir=None, overwrite=False):
    """Stage all types, then replace generated output; raw inputs stay intact.

    Returns {kind: {train: pickle_path, val: pickle_path}}. Each type has
    separate classes/attributes and an 80/20 split by source image, as in VMA.
    """
    data_root = Path(data_root).expanduser().resolve()
    out_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None else data_root / 'prepared')
    if out_dir == data_root or out_dir in data_root.parents or any(
            out_dir == data_root / k or data_root / k in out_dir.parents
            for k in SAMPLES_PER_TRAJECTORY):
        raise ValueError('Output must not overlap raw SD type directories')
    sources = _discover_sources(data_root)
    _check_output(out_dir, overwrite)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix='.sd-convert-', dir=out_dir.parent) as temporary:
        stage = Path(temporary) / 'prepared'
        for kind, annotations in sources.items():
            image_dir = stage / kind / 'images'
            image_dir.mkdir(parents=True)
            intensity_dir = stage / kind / 'intensity_image'
            intensity_dir.mkdir()
            groups = {}
            for annotation in annotations:
                name = annotation['image_key']
                source = data_root / kind / 'origin_data'
                image_path = source / 'image_data' / f'{name}.jpg'
                intensity_path = source / 'intensity_image' / f'{name}.png'
                trajectory_path = source / 'trajectory_data' / f'{name}.json'
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None or image.shape != (SOURCE_SIZE, SOURCE_SIZE,
                                                    3):
                    raise ValueError(
                        f'{image_path}: expected 6000x6000 color image')
                intensity = cv2.imread(
                    str(intensity_path), cv2.IMREAD_UNCHANGED)
                if (intensity is None or intensity.dtype != np.uint8
                        or intensity.shape != image.shape[:2]):
                    raise ValueError(
                        f'{intensity_path}: expected aligned uint8 intensity image'
                    )
                with trajectory_path.open() as stream:
                    trajectories = json.load(stream)
                instances = read_instances(annotation, kind)
                samples = []
                for crop_index, (trajectory_index, sample_index,
                                 origin) in enumerate(
                                     sample_windows(
                                         trajectories,
                                         SAMPLES_PER_TRAJECTORY[kind])):
                    gt = crop_instances(instances, kind, origin)
                    if not gt:
                        continue
                    left, top = origin
                    filename = f'{name}_{crop_index}.jpg'
                    if not cv2.imwrite(
                            str(image_dir / filename),
                            image[top:top + CROP_SIZE, left:left + CROP_SIZE]):
                        raise OSError(f'Cannot write crop: {filename}')
                    intensity_name = f'{name}_{crop_index}.png'
                    if not cv2.imwrite(
                            str(intensity_dir / intensity_name),
                            intensity[top:top + CROP_SIZE,
                                      left:left + CROP_SIZE]):
                        raise OSError(
                            f'Cannot write intensity crop: {intensity_name}')
                    samples.append(
                        dict(
                            sample_id=f'sd_{kind}/{Path(filename).stem}',
                            img_path=f'images/{filename}',
                            height=CROP_SIZE,
                            width=CROP_SIZE,
                            intensity_img_path=
                            f'intensity_image/{intensity_name}',
                            source_id=f'sd_{kind}/{name}',
                            crop_origin=origin,
                            source_img_path=image_path.relative_to(
                                data_root).as_posix(),
                            source_intensity_img_path=intensity_path.
                            relative_to(data_root).as_posix(),
                            trajectory_path=trajectory_path.relative_to(
                                data_root).as_posix(),
                            trajectory_index=trajectory_index,
                            trajectory_sample_index=sample_index,
                            instances=gt))
                if samples or kind != 'line':
                    groups[name] = samples
            names = list(groups) if kind == 'line' else sorted(groups)
            split_index = round(0.8 * len(names))
            for split, selected in [('train', names[:split_index]),
                                    ('val', names[split_index:])]:
                samples = [
                    sample for name in selected for sample in groups[name]
                ]
                infos = dict(
                    metainfo=dict(
                        dataset_type='bev_image_map',
                        info_version='1.0',
                        dataset_name=f'sd_{kind}',
                        image_size=[CROP_SIZE, CROP_SIZE],
                        coordinate_system='image_pixel',
                        has_annotations=True,
                        has_intensity=True,
                        intensity_encoding=
                        'estimated_normalized_jet_index_uint8',
                        classes=CLASSES[kind],
                        attribute_schema=attribute_schema(kind),
                        split=split,
                        source_ids=[f'sd_{kind}/{n}' for n in selected],
                        crop_policy=dict(
                            name='vma_legacy_trajectory',
                            source_size=[SOURCE_SIZE, SOURCE_SIZE],
                            samples_per_trajectory=SAMPLES_PER_TRAJECTORY[
                                kind],
                            filter_by_gt=True),
                        source_paths_relative_to='original_sd_data_root'),
                    data_list=samples)
                with (stage / kind /
                      f'sd_map_infos_{split}.pkl').open('wb') as stream:
                    pickle.dump(infos, stream, protocol=4)
                print(
                    f'{kind}/{split}: {len(selected)} source images -> '
                    f'{len(samples)} crops, {sum(len(s["instances"]) for s in samples)} instances',
                    flush=True)
        backup = Path(temporary) / 'previous'
        if out_dir.exists():
            out_dir.rename(backup)
        try:
            stage.rename(out_dir)
        except BaseException:
            if backup.exists():
                backup.rename(out_dir)
            raise
        # TemporaryDirectory removes only the previously validated generated tree.
    return {
        kind: {
            split: str(out_dir / kind / f'sd_map_infos_{split}.pkl')
            for split in ('train', 'val')
        }
        for kind in sources
    }
