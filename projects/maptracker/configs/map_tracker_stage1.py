# Stage 1 currently loads and packs camera images for dataset inspection.
_base_ = ['./_base_/datasets/nusc_map_mini.py']

visualizer = dict(
    type='MapLocalVisualizer',
    name='map_visualizer',
    icon_height=36,
)
