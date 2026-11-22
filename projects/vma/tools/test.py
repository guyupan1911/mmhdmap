"""Initialize the configured dataset and optionally test its visualizer."""

import argparse
import sys
from pathlib import Path

from mmcv.transforms import LoadImageFromFile
from mmengine.config import Config
from mmengine.registry import init_default_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mmhdmap.registry import DATASETS, VISUALIZERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--data-root', help='Override dataset.data_root')
    parser.add_argument('--split', choices=('train', 'val'))
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Limit visualized samples; default: all')
    parser.add_argument('--out-dir', help='Save visualizations as PNG files')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--wait-time', type=float, default=0)
    parser.add_argument('--no-gt', action='store_true')
    parser.add_argument('--draw-points', action='store_true')
    parser.add_argument('--draw-endpoints', action='store_true')
    parser.add_argument('--draw-instance-ids', action='store_true')
    parser.add_argument('--draw-attributes', action='store_true')
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error('--max-samples must be positive')

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmhdmap'))
    if args.data_root:
        cfg.dataset.data_root = args.data_root
    if args.split:
        cfg.dataset.ann_file = f'sd_map_infos_{args.split}.pkl'
    dataset = DATASETS.build(cfg.dataset)
    print(f'{type(dataset).__name__}: {len(dataset)} samples')
    if not (args.out_dir or args.show):
        return

    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo
    intensity_loader = LoadImageFromFile(
        color_type='unchanged', imdecode_backend='pillow')
    count = len(dataset) if args.max_samples is None else min(
        len(dataset), args.max_samples)
    try:
        for index in range(count):
            output = dataset[index]
            sample = output['data_samples']
            # Model inputs are BGR CHW; the visualizer expects RGB HWC.
            image = output['inputs'].detach().cpu().permute(
                1, 2, 0).numpy()[..., ::-1].copy()
            backgrounds = [('', image)]
            if 'intensity_img_path' in sample:
                # Current pipeline has no geometric augmentation: GT aligns
                # with this original crop. Load intensity only for display.
                intensity = intensity_loader(
                    dict(img_path=sample.intensity_img_path))['img']
                backgrounds.append(('intensity_image', intensity))
            filename = f'{index:06d}_{Path(sample.img_path).stem}.png'
            for kind, background in backgrounds:
                out_file = str(Path(args.out_dir) / kind /
                               filename) if args.out_dir else None
                visualizer.add_datasample(
                    name=f'{sample.sample_id}/{kind}'
                    if kind else sample.sample_id,
                    image=background,
                    data_sample=sample,
                    out_file=out_file,
                    draw_gt=not args.no_gt,
                    draw_points=args.draw_points,
                    draw_endpoints=args.draw_endpoints,
                    draw_instance_ids=args.draw_instance_ids,
                    draw_attributes=args.draw_attributes,
                    show=args.show,
                    wait_time=args.wait_time)
    finally:
        visualizer.close()
    print(f'Visualized {count} samples' +
          (f' -> {args.out_dir}' if args.out_dir else ''))


if __name__ == '__main__':
    main()
