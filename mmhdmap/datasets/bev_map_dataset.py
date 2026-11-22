from copy import deepcopy
from pathlib import Path
from typing import Optional

from mmhdmap.registry import DATASETS
from .base_map_dataset import BaseMapDataset


@DATASETS.register_module()
class BevMapDataset(BaseMapDataset):
    """Read prepared BEV image infos in ``metainfo + data_list`` format.

    Images are already in BEV; this class does not project sensor data or
    crop images. BaseDataset loads the pickle and runs the pipeline. Image
    decoding and MapDataSample packing belong to transforms.

    ``img_path`` is relative to ``data_root`` (or ``data_prefix.img_path``).
    Optional intensity_img_path uses its own data_prefix (default data_root).
    Only the selected model image is decoded by the pipeline. Intensity paths
    are retained for on-demand visualization; initialization decodes neither.
    Source image/trajectory paths remain provenance relative to the original
    data root; they are not loaded. Missing ``instances`` means no GT, not
    an annotated negative sample. This class does not generate GT targets.

    Args:
        ann_file: Path to BEV image infos, relative to data_root or absolute.
        data_root: Prepared dataset root containing the cropped images.
        data_prefix: Optional img_path prefix, relative to data_root unless
            absolute. Defaults to data_root itself.
        modality: Defaults to use_bev_image=True.
        **kwargs: BaseMapDataset options, including pipeline and test_mode.
    """

    def __init__(self,
                 ann_file: str,
                 data_root: str = '',
                 data_prefix: Optional[dict] = None,
                 modality: Optional[dict] = None,
                 **kwargs) -> None:
        prefixes = dict(img_path='', intensity_img_path='')
        if data_prefix is not None:
            prefixes.update(data_prefix)
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=prefixes,
            modality=(dict(
                use_bev_image=True) if modality is None else modality),
            **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Validate declared dimensions and resolve the prepared image path."""
        meta = self.metainfo
        if (meta.get('dataset_type') != 'bev_image_map'
                or meta.get('info_version') != '1.0'):
            raise ValueError('BevMapDataset requires bev_image_map infos 1.0')
        if meta.get('coordinate_system') != 'image_pixel':
            raise ValueError('BEV infos must use image_pixel coordinates')
        size = meta.get('image_size')
        if (not isinstance(size, (list, tuple)) or len(size) != 2
                or any(type(v) is not int or v <= 0 for v in size)):
            raise ValueError('image_size must be positive integer [H, W]')
        info = deepcopy(raw_data_info)
        if not isinstance(info.get('sample_id'), str) or not info['sample_id']:
            raise ValueError('sample_id must be a nonempty string')
        dimensions = [info.get('height'), info.get('width')]
        if (any(type(v) is not int for v in dimensions)
                or dimensions != list(size)):
            raise ValueError(f'{info["sample_id"]}: height/width must match '
                             f'image_size {size}')
        keys = ['img_path']
        if meta.get('has_intensity') or 'intensity_img_path' in info:
            keys.append('intensity_img_path')
        for key in keys:
            filename = info.get(key)
            if not isinstance(filename, str) or not filename:
                raise ValueError(f'{key} must be a nonempty relative path')
            path = Path(filename)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'{key} must be relative: {filename!r}')
            info[key] = str(Path(self.data_prefix[key]) / path)
        return info
