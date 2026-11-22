"""SD image and raw vector GT check; no model or training targets yet."""

default_scope = 'mmhdmap'

visualizer = dict(type='BevMapVisualizer', name='bev_map_visualizer')

dataset = dict(
    type='BevMapDataset',
    data_root='/home/guyu/data/vma/sd_data/prepared/line',
    ann_file='sd_map_infos_train.pkl',
    modality=dict(use_bev_image=True),
    test_mode=True,
    pipeline=[
        dict(
            type='mmcv.LoadImageFromFile',
            to_float32=False,
            imdecode_backend='pillow',
            color_type='color_ignore_orientation'),  # BGR uint8
        dict(type='LoadVectorMapAnnotations'),
        dict(type='PackBEVMapInputs'),
    ])
