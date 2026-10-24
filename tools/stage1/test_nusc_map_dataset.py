"""Build a dataset, inspect a MapDataSample and visualize its camera views."""

import argparse
import math
from pathlib import Path
from pprint import pprint

from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmhdmap.datasets.map_utils import NuScenesMapExtractor
from mmhdmap.registry import DATASETS, VISUALIZERS
from mmhdmap.structures import MapDataSample


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
        help='Directory containing nuScenes basemap PNGs '
        '(default: <dataset.data_root>/basemap; empty string disables basemap)'
    )
    parser.add_argument(
        '--map-roi-size',
        type=float,
        nargs=2,
        metavar=('WIDTH', 'LENGTH'),
        default=(60.0, 80.0),
        help='Lateral width and longitudinal length in meters (default: 60 80)'
    )
    parser.add_argument(
        '--map-layers',
        nargs='*',
        choices=NuScenesMapExtractor.SUPPORTED_LAYERS,
        default=list(NuScenesMapExtractor.DEFAULT_LAYERS),
        help='Raw SDK layers to overlay; pass no values for basemap only')
    args = parser.parse_args()
    if any(not math.isfinite(size) or size <= 0 for size in args.map_roi_size):
        parser.error('--map-roi-size must contain two positive finite lengths')
    return args


def main():
    # 1. Load the config and resolve registry names in the mmhdmap scope.
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if 'visualizer' not in cfg:
        raise ValueError(
            'Map visualization requires a visualizer in the config')
    init_default_scope(cfg.get('default_scope', 'mmhdmap'))

    # 2. Build the selected dataset and print its basic information.
    dataset = DATASETS.build(cfg[f'{args.split}_dataset'])
    basemap_dir = (
        str(Path(dataset.data_root) /
            'basemap') if args.basemap_dir is None else args.basemap_dir)

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
        extractor = None
        try:
            # 7. Extract SDK basemap and raw annotations outside the pipeline.
            bev_image = None
            map_geometries = None
            map_resolution = None
            map_local2global = None
            extractor = NuScenesMapExtractor(
                data_root=dataset.data_root,
                map_name=data_sample.location,
                basemap_dir=basemap_dir)
            roi_size = tuple(args.map_roi_size)
            if basemap_dir:
                basemap = extractor.extract_basemap(
                    data_sample.ego2global, roi_size=roi_size)
                roi_size = basemap['roi_size']
                bev_image = basemap['image']
                map_resolution = basemap['resolution']
                print(f'\nMap: {data_sample.location}, '
                      f'ROI: {roi_size} m (width, length), '
                      f'crop: {bev_image.shape[1]} x {bev_image.shape[0]} px '
                      f'at {map_resolution:g} m/px')
            # Use the pixel-rounded ROI for both raster and vector data.
            annotations = extractor.extract_geometries(
                data_sample.ego2global, roi_size, layers=args.map_layers)
            map_geometries = annotations['geometries']
            map_local2global = annotations['local2global']
            print('Raw map geometry counts:')
            pprint({
                name: len(geoms)
                for name, geoms in map_geometries.items()
            },
                   sort_dicts=False)
            ground_z = data_sample.ego2global[2, 3]
            print(f'Camera projection: horizontal ground at global '
                  f'Z={ground_z:.3f} m; capture-time camera poses; '
                  'no occlusion handling')

            # 8. Draw camera views and the basemap with the vehicle facing up.
            visualizer.add_datasample(
                name=data_sample.token,
                images=images,
                data_sample=data_sample,
                out_file=str(output),
                bev_image=bev_image,
                map_geometries=map_geometries,
                map_resolution=map_resolution,
                map_local2global=map_local2global)
        finally:
            if extractor is not None:
                extractor.clear_cache()
            # Release plotting resources even if rendering or saving fails.
            visualizer.close()
        print(f'\nVisualization saved to: {output.resolve()}')


if __name__ == '__main__':
    main()
