"""Dispatch dataset preparation to the corresponding map converter."""

import argparse

from dataset_converters import nuscenes_map_converter


def nuscenes_data_prep(args):
    """Convert only the explicitly selected nuScenes release."""
    return nuscenes_map_converter.create_nuscenes_map_infos(
        data_root=args.data_root,
        out_dir=args.out_dir,
        version=args.version,
        info_prefix=args.info_prefix,
        split_file=args.split_file,
        overwrite=args.overwrite)


def parse_args(argv=None):
    """Parse dataset-specific arguments for the unified entry point."""
    parser = argparse.ArgumentParser(description='Prepare HD map datasets')
    subparsers = parser.add_subparsers(dest='dataset', required=True)
    nuscenes = subparsers.add_parser(
        'nuscenes', help='Prepare nuScenes camera, LiDAR and map metadata')
    nuscenes.add_argument(
        '--data-root', required=True, help='nuScenes data root')
    nuscenes.add_argument(
        '--out-dir', required=True, help='Output directory for info PKLs')
    nuscenes.add_argument(
        '--version',
        choices=nuscenes_map_converter.VERSIONS,
        required=True,
        help='mini/trainval produce train and val; test produces test only')
    nuscenes.add_argument(
        '--info-prefix', '--extra-tag', default='mmhdmap_nuscenes')
    nuscenes.add_argument(
        '--split-file', help='Custom scene split JSON manifest')
    nuscenes.add_argument('--overwrite', action='store_true')
    return parser.parse_args(argv)


def main():
    """Dispatch to the selected dataset preparation function."""
    args = parse_args()
    if args.dataset == 'nuscenes':
        nuscenes_data_prep(args)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')


if __name__ == '__main__':
    main()
