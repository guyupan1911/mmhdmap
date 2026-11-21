"""Explicit bootstrap for legacy VMA on torch 2.7 (not an OpenMMLab 2 port)."""
import sys
import types
# MMCV 1.4's collator imports this module removed in recent torch.
six = types.ModuleType('torch._six')
six.string_classes = (str, bytes)
six.int_classes = (int, )
six.PY3 = True
sys.modules.setdefault('torch._six', six)
import projects.mmdet3d_plugin  # noqa: E402,F401
from projects.mmdet3d_plugin.datasets import SD_Driving_Line_Dataset  # noqa: E402,F401
