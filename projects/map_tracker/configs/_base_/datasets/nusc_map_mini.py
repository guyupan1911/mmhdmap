default_scope = 'mmhdmap'

dataset_type = 'NuscMapDataset'
data_root = '/home/guyu/data/nuscenes/mini'

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
    ann_file='mmhdmap_nuscenes_mini_infos_train.pkl',
    modality=dict(use_camera=True, use_lidar=False),
    pipeline=train_pipeline,
    test_mode=False,
)

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='mmhdmap_nuscenes_mini_infos_val.pkl',
    modality=dict(use_camera=True, use_lidar=False),
    pipeline=test_pipeline,
    test_mode=True,
)
