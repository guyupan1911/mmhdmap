"""Build a dataset, inspect a MapDataSample and visualize its camera views."""

import argparse
import math
from pathlib import Path
from pprint import pprint

from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmhdmap.registry import DATASETS, VISUALIZERS
from mmhdmap.structures import MapDataSample
from mmhdmap.visualization import crop_nusc_basemap


def parse_args():
    """Parse the config path, dataset split, sample index and output directory."""
    parser = argparse.ArgumentParser(
        description='Initialize NuscMapDataset and print sample information')
    parser.add_argument('config', help='Path to the configuration file')
    parser.add_argument(
        '--split',
        choices=['train', 'val'],
        default='train',
        help='Dataset split (default: train)')
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=0,
        help='Index of the sample to inspect (default: 0)')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/dataset_vis',
        help='Visualization output directory (default: work_dirs/dataset_vis)')
    parser.add_argument(
        '--basemap-dir',
        default='/data/nuscenes/mini/basemap',
        help='Directory containing nuScenes basemap PNGs '
        '(default: /data/nuscenes/mini/basemap)')
    parser.add_argument(
        '--map-roi-size',
        type=float,
        nargs=2,
        metavar=('WIDTH', 'LENGTH'),
        default=(60.0, 80.0),
        help='Lateral width and longitudinal length in meters (default: 60 80)'
    )
    args = parser.parse_args()
    if any(not math.isfinite(size) or size <= 0 for size in args.map_roi_size):
        parser.error('--map-roi-size must contain two positive finite lengths')
    return args


def main():
    # 1. Load the config and resolve registry names in the mmhdmap scope.
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.basemap_dir and 'visualizer' not in cfg:
        raise ValueError('--basemap-dir requires a visualizer in the config')
    init_default_scope(cfg.get('default_scope', 'mmhdmap'))

    # 2. Build the selected dataset and print its basic information.
    dataset = DATASETS.build(cfg[f'{args.split}_dataset'])

    print('Dataset information:')
    pprint(
        dict(
            dataset=type(dataset).__name__,
            split=args.split,
            ann_file=dataset.ann_file,
            data_root=dataset.data_root,
            samples=len(dataset),
            classes=dataset.classes,
            modality=dataset.modality,
            test_mode=dataset.test_mode),
        sort_dicts=False)

    # 3. Validate the index and run the dataset pipeline for one sample.
    if not 0 <= args.sample_idx < len(dataset):
        raise IndexError(f'sample-idx must be in [0, {len(dataset)}), '
                         f'got {args.sample_idx}')

    sample = dataset[args.sample_idx]
    data_sample = sample.get('data_samples')
    if not isinstance(data_sample, MapDataSample):
        raise TypeError('The pipeline must end with PackMapInputs and return '
                        'a MapDataSample')
    image_tensors = sample['inputs']['imgs']

    # 4. Inspect metadata and tensor shapes without printing image pixels.
    # print(f'\nSample {args.sample_idx} after pipeline:')
    # summary = dict(
    #     inputs=dict(imgs=[
    #         dict(shape=tuple(image.shape), dtype=str(image.dtype))
    #         for image in image_tensors
    #     ]),
    #     data_samples=dict(
    #         type=type(data_sample).__name__,
    #         metainfo=data_sample.metainfo,
    #         data_fields=list(data_sample.keys())))
    # pprint(summary, sort_dicts=False)

    # 5. Convert CHW tensors to HWC display images when visualization is enabled.
    if 'visualizer' in cfg:
        # The current pipeline preserves RGB uint8 pixels without normalization.
        images = [
            image.detach().cpu().permute(1, 2, 0).numpy()
            for image in image_tensors
        ]
        # 6. Build the visualizer and pass images separately from MapDataSample.
        visualizer = VISUALIZERS.build(cfg.visualizer)
        visualizer.dataset_meta = dataset.metainfo
        output = Path(args.out_dir) / f'{data_sample.token}_surround.png'
        try:
            # 7. Prepare the optional basemap outside the dataset pipeline.
            bev_image = None
            if args.basemap_dir:
                basemap_path = (
                    Path(args.basemap_dir) / f'{data_sample.location}.png')
                bev_image = crop_nusc_basemap(
                    basemap_path=str(basemap_path),
                    map_path=data_sample.map_path,
                    ego2global=data_sample.ego2global,
                    roi_size=tuple(args.map_roi_size))
                print(f'\nBasemap: {basemap_path}, '
                      f'ROI: {args.map_roi_size[0]:g} m wide x '
                      f'{args.map_roi_size[1]:g} m long (ego-centered), '
                      f'crop: {bev_image.shape[1]} x {bev_image.shape[0]} px '
                      'at native resolution')

            # 8. Draw camera views and the basemap with the vehicle facing up.
            visualizer.add_datasample(
                name=data_sample.token,
                images=images,
                data_sample=data_sample,
                out_file=str(output),
                bev_image=bev_image)
        finally:
            # Release plotting resources even if rendering or saving fails.
            visualizer.close()
        print(f'\nVisualization saved to: {output.resolve()}')


if __name__ == '__main__':
    main()
