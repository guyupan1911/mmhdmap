from mmengine.structures import BaseDataElement, InstanceData, PixelData


class MapDataSample(BaseDataElement):
    """Data shared by map perception models, metrics and visualizers.

    Camera names, calibration, poses and sample identifiers belong to
    ``metainfo``. Input images are stored separately in the pipeline's
    ``inputs`` (a single-image tensor or multi-view dict), not in this
    container. For prepared BEV images, metadata includes sample_id,
    ori_shape, img_shape and optional source_id/crop_origin. Unannotated
    images leave gt_instances absent.

    For nuScenes, top-level ego2global is the reference LiDAR-time ego pose;
    images[camera_name].ego2global is the camera capture-time ego pose.
    Frame indices and prev/next tokens are metadata, not GT instance fields.

    Optional data fields:
        gt_instances (InstanceData): Vector map ground truth. Per-instance
            labels (int64 tensor), polylines (variable-length float32 arrays),
            and attributes (raw dictionaries) share the same instance count.
            Model transforms may additionally attach encoded training targets.
        pred_instances (InstanceData): Predicted map instances and scores.
        gt_sem_seg (PixelData): BEV semantic ground truth, e.g. a ``sem_seg``
            tensor with shape (C, H, W) for per-class binary masks.
        pred_sem_seg (PixelData): Predicted BEV semantic maps.

    Fields remain absent until explicitly assigned. Geometry coordinates and
    BEV grid conventions must be specified by the transforms producing them.
    """

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData) -> None:
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self) -> None:
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData) -> None:
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self) -> None:
        del self._pred_instances

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self) -> None:
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self) -> None:
        del self._pred_sem_seg
