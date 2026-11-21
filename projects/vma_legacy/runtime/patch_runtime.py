"""Isolate VMA's image path; never edit the mounted upstream checkout.

The original MVX base is copied with its image construction unchanged. Unused
3D imports are removed to avoid building unrelated pre-Blackwell point ops.
The MMCV 1.4 Python API uses the base image's 2.1 compiled kernels; the actual
deformable-attention forward/backward is tested by the validation harness.
"""
from pathlib import Path
import json
import os
import mmcv

if os.environ.get('VMA_PATCH_OPS', '1') == '1':
    ops_init = Path(mmcv.__path__[0]) / 'ops/__init__.py'
    ops_text = ops_init.read_text()
    ops_text = ops_text.replace(
        'from .iou3d import boxes_iou_bev, nms_bev, nms_normal_bev', '')
    for name in ('boxes_iou_bev', 'nms_bev', 'nms_normal_bev'):
        ops_text = ops_text.replace(repr(name) + ',', '')
    ops_init.write_text(ops_text)

root = Path(os.environ.get('VMA_RUNTIME_ROOT', '/opt/VMA'))
changes = []


def write(rel, text):
    p = root / rel
    p.write_text(text)
    changes.append(rel)


write(
    'projects/mmdet3d_plugin/__init__.py',
    'from .vma import *\nfrom .bevformer.modules.decoder import CustomMSDeformableAttention\n'
)
write('projects/mmdet3d_plugin/models/utils/__init__.py',
      'from .grid_mask import GridMask\n')
write('projects/mmdet3d_plugin/bevformer/__init__.py', '')
write('projects/mmdet3d_plugin/bevformer/modules/__init__.py', '')
write(
    'projects/mmdet3d_plugin/datasets/__init__.py',
    'from .pipelines import *\nfrom .sd_driving_line_dataset import SD_Driving_Line_Dataset\n'
)
base = (root /
        'mmdetection3d/mmdet3d/models/detectors/mvx_two_stage.py').read_text()
base = base.replace(
    'from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,\n                          merge_aug_bboxes_3d, show_result)',
    '')
base = base.replace('from mmdet3d.ops import Voxelization', '')
base = base.replace('from .. import builder',
                    'from mmdet.models import builder')
base = base.replace(
    'from .base import Base3DDetector',
    'from mmdet.models.detectors.base import BaseDetector as Base3DDetector')
write('projects/mmdet3d_plugin/vma/detectors/mvx_image_base.py', base)
for rel, old, new in [
    ('projects/mmdet3d_plugin/vma/detectors/vma.py',
     'from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector',
     'from .mvx_image_base import MVXTwoStageDetector'),
    ('projects/mmdet3d_plugin/vma/dense_heads/vmahead.py',
     'from mmdet3d.core.bbox.coders import build_bbox_coder',
     'from mmdet.core.bbox import build_bbox_coder'),
    ('projects/mmdet3d_plugin/datasets/sd_driving_line_dataset.py',
     'from mmdet3d.datasets.pipelines import Compose',
     'from mmdet.datasets.pipelines import Compose'),
]:
    text = (root / rel).read_text()
    assert old in text, (rel, old)
    write(rel, text.replace(old, new))
Path(
    os.environ.get('VMA_PATCH_MANIFEST',
                   '/opt/vma-support/patch_manifest.json')).write_text(
                       json.dumps(changes, indent=2))
