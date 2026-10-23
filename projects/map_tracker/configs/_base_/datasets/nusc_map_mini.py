default_scope = 'mmhdmap'

dataset_type = 'NuscMapDataset'
data_root = '/data/nuscenes/mini'

# TODO: Remove after regenerating infos with data-root-relative paths.
ann_path_prefix = 'datasets/nuscenes/mini'

train_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles'),
    dict(type='PackMapInputs'),
]
test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles'),
    dict(type='PackMapInputs'),
]

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='nuscenes_map_infos_train.pkl',
    ann_path_prefix=ann_path_prefix,
    modality=dict(use_camera=True, use_lidar=False),
    pipeline=train_pipeline,
    test_mode=False,
)

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='nuscenes_map_infos_val.pkl',
    ann_path_prefix=ann_path_prefix,
    modality=dict(use_camera=True, use_lidar=False),
    pipeline=test_pipeline,
    test_mode=True,
)
