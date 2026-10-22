from typing import Dict, Optional, Tuple

from mmengine.dataset import BaseDataset

from mmhdmap.registry import DATASETS


@DATASETS.register_module()
class BaseMapDataset(BaseDataset):
    """Base dataset for HD map perception.

    Subclasses define ``METAINFO`` and adapt source annotations to the input
    format expected by their pipeline. Map extraction, target generation and
    data sample packing belong to transforms.

    Args:
        ann_file (str): Annotation file in MMEngine's standard format unless
            the subclass overrides ``load_data_list``.
        metainfo (dict, optional): Dataset metadata, including ordered map
            class names in ``classes``. Source labels must be translated to
            this order by the dataset adapter or transforms.
        data_root (str): Root directory for annotations and data prefixes.
        data_prefix (dict, optional): Path prefixes. Defaults to an empty
            dict because map samples need not have a top-level ``img_path``.
        modality (dict, optional): Boolean flags describing model inputs.
            Defaults to camera-only. Missing flags in an explicit config
            are disabled. Additional modalities may be declared, but must
            be supported by the pipeline and model. These flags do not load
            sensor data or remove calibration metadata by themselves.
        **kwargs: Arguments forwarded to ``BaseDataset``, including pipeline,
            test_mode, lazy_init, serialize_data, indices and max_refetch.
    """

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: Optional[dict] = None,
                 modality: Optional[Dict[str, bool]] = None,
                 **kwargs) -> None:
        if modality is None:
            modality = dict(use_camera=True, use_lidar=False)
        self.modality = dict(use_camera=False, use_lidar=False)
        self.modality.update(modality)
        if not all(
                isinstance(value, bool) for value in self.modality.values()):
            raise TypeError('modality flags must be bool values')
        if not any(self.modality.values()):
            raise ValueError('At least one input modality must be enabled')

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix={} if data_prefix is None else data_prefix,
            **kwargs)

    @property
    def classes(self) -> Tuple[str, ...]:
        """Map class names in training label order."""
        return tuple(self.metainfo.get('classes', ()))

    @property
    def cat2id(self) -> Dict[str, int]:
        """Map class names to contiguous training labels."""
        return {name: label for label, name in enumerate(self.classes)}

    @property
    def num_classes(self) -> int:
        """Number of map classes."""
        return len(self.classes)
