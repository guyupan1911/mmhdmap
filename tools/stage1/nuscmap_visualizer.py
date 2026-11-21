"""Render selected nuScenes map layers around a global XY position."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from nuscenes.map_expansion.map_api import NuScenesMap, locations
from PIL import Image
from shapely import affinity
from shapely.plotting import plot_line, plot_polygon

MAP_LAYERS = ('drivable_area', 'road_segment', 'road_block', 'lane',
              'ped_crossing', 'walkway', 'stop_line', 'carpark_area',
              'road_divider', 'lane_divider')


def parse_args():
    """Parse the map region, global center, patch size and display layers."""
    parser = argparse.ArgumentParser(
        description='Visualize raw nuScenes map layers around a global XY '
        'position in meters. No sample PKL or camera data is required.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data-root',
        default='/home/guyu/data/nuscenes/mini',
        help='nuScenes root containing maps/expansion')
    parser.add_argument(
        '--map-name',
        default='boston-seaport',
        choices=locations,
        help='Map region')
    parser.add_argument(
        '--basemap-dir',
        default=None,
        help='Directory containing <map-name>.png; defaults to '
        '<data-root>/maps/basemap, then <data-root>/basemap')
    parser.add_argument(
        '--center',
        type=float,
        nargs=2,
        default=(1854.52, 867.99),
        metavar=('X', 'Y'),
        help='Center in the selected map global coordinates, in meters')
    parser.add_argument(
        '--size',
        type=float,
        nargs=2,
        default=(80.0, 60.0),
        metavar=('WIDTH', 'HEIGHT'),
        help='Extent along global X and Y in meters')
    parser.add_argument(
        '--layers',
        nargs='*',
        choices=MAP_LAYERS,
        default=[
            'road_segment', 'lane', 'ped_crossing', 'road_divider',
            'lane_divider'
        ],
        help='Raw SDK layers to overlay; pass no values for basemap only')
    parser.add_argument(
        '--out-file',
        default='work_dirs/dataset_vis/nuscmap.png',
        help='Output image path')
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in args.center):
        parser.error('--center must contain two finite coordinates')
    if not all(math.isfinite(value) and value > 0 for value in args.size):
        parser.error('--size must contain two positive finite lengths')
    args.layers = list(dict.fromkeys(args.layers))
    return args


def load_basemap_patch(path, canvas_edge, patch):
    """Crop native pixels and return their exact global coordinate extent."""
    # Only disable Pillow's size guard while opening a trusted dataset map.
    max_pixels = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as source:
            map_width, map_height = canvas_edge
            scale_x = source.width / map_width
            scale_y = source.height / map_height
            xmin, ymin, xmax, ymax = patch
            left = max(0, math.floor(xmin * scale_x))
            right = min(source.width, math.ceil(xmax * scale_x))
            top = max(0, math.floor((map_height - ymax) * scale_y))
            bottom = min(source.height, math.ceil(
                (map_height - ymin) * scale_y))
            if left >= right or top >= bottom:
                raise ValueError('The requested patch is outside the basemap')
            # PNG rows run downwards, while global map Y runs upwards.
            extent = (left / scale_x, right / scale_x,
                      map_height - bottom / scale_y,
                      map_height - top / scale_y)
            with source.crop((left, top, right, bottom)) as crop:
                pixels = np.array(crop.convert('RGB'))
        return pixels, extent
    finally:
        Image.MAX_IMAGE_PIXELS = max_pixels


def main():
    args = parse_args()
    # Save figures without requiring a desktop display on the training server.
    plt.switch_backend('Agg')

    # Load the region map directly through the official SDK.
    nusc_map = NuScenesMap(dataroot=args.data_root, map_name=args.map_name)
    if args.basemap_dir is not None:
        basemap_path = Path(args.basemap_dir) / f'{args.map_name}.png'
    else:
        basemap_path = (
            Path(args.data_root) / 'maps' / 'basemap' / f'{args.map_name}.png')
        if not basemap_path.is_file():
            basemap_path = (
                Path(args.data_root) / 'basemap' / f'{args.map_name}.png')
    if not basemap_path.is_file():
        raise FileNotFoundError(
            f'Basemap not found: {basemap_path}. Set --basemap-dir.')
    x, y = args.center
    width, height = args.size
    patch = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
    basemap, extent = load_basemap_patch(basemap_path, nusc_map.canvas_edge,
                                         patch)

    # The SDK returns clipped geometry in patch-local coordinates.
    geometries = nusc_map.get_map_geom((x, y, height, width),
                                       patch_angle=0,
                                       layer_names=args.layers)
    fig, ax = plt.subplots(figsize=(12, 9))
    try:
        ax.imshow(
            basemap,
            extent=extent,
            origin='upper',
            interpolation='nearest',
            zorder=0)
        ax.grid(False)
        # Shapely plotting supports holes without the SDK's legacy descartes.
        handles = []
        for layer, items in geometries:
            color = nusc_map.explorer.color_map[layer]
            is_polygon = layer in nusc_map.non_geometric_polygon_layers
            for geometry in items:
                geometry = affinity.translate(geometry, xoff=x, yoff=y)
                if is_polygon:
                    plot_polygon(
                        geometry,
                        ax=ax,
                        add_points=False,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.4)
                else:
                    plot_line(
                        geometry,
                        ax=ax,
                        add_points=False,
                        color=color,
                        linewidth=1.5)
            handles.append(
                Patch(
                    facecolor=color, edgecolor=color, alpha=0.4, label=layer
                ) if is_polygon else Line2D([], [], color=color, label=layer))

        # Keep the display in global meters and restrict it to the exact ROI.
        ax.set_xlim(patch[0], patch[2])
        ax.set_ylim(patch[1], patch[3])
        ax.set_aspect('equal', adjustable='box')
        center_marker, = ax.plot(
            x,
            y,
            marker='+',
            color='red',
            markersize=12,
            markeredgewidth=2,
            linestyle='None',
            label='Input position')
        ax.set_xlabel('Global X (m)')
        ax.set_ylabel('Global Y (m)')
        ax.set_title(f'{args.map_name} | center=({x:g}, {y:g}) m | '
                     f'size={width:g} x {height:g} m')
        ax.legend(
            handles=[*handles, center_marker],
            loc='upper left',
            bbox_to_anchor=(1.02, 1))

        output = Path(args.out_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig)

    print(f'Map: {args.map_name}')
    print(f'Basemap: {basemap_path}')
    print(f'Center: ({x:g}, {y:g}) m; size: {width:g} x {height:g} m')
    print(f'Layers: {", ".join(args.layers)}')
    print(f'Saved: {output.resolve()}')


if __name__ == '__main__':
    main()
