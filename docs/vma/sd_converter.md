# SD 图像与 GT 转换

入口为 `tools/create_data.py sd`，实现位于 `projects/vma/tools/sd_converter.py`。一次处理 SD 根目录下存在的 line、box、freespace，裁剪图像和标注，生成统一的 train/val infos。

```bash
python projects/vma/tools/recover_sd_intensity.py --data-root /home/guyu/data/vma/sd_data
python tools/create_data.py sd --data-root /home/guyu/data/vma/sd_data
# 替换已有的 converter 产物：
python tools/create_data.py sd --data-root /home/guyu/data/vma/sd_data --overwrite
```

依赖 NumPy、OpenCV、Shapely；无需加载模型。默认输出 `DATA_ROOT/prepared`，可用 `--out-dir` 指定。`--overwrite` 只接受 converter 管理的目录结构；全部转换成功后替换旧产物，同时移除旧 `sd_image_infos.pkl` 和不再保留的小图。失败时保留旧结果，原始数据不变。

## 输入与输出

先运行恢复脚本生成 `{type}/origin_data/intensity_image/*.png`，converter 要求这些单通道 uint8 图与原图尺寸相同。强度为逆 JET 的近似 0–255 索引，不做 log 映射。

每类输入为 `{type}/origin_data/` 下的 `image_data/*.jpg`、`trajectory_data/*.json` 和 `{type}_data_6k.json`。图像与 GT 中的 `image_key` 必须一一对应；轨迹文件按图像同名查找，多余轨迹文件忽略。

```text
sd_data/prepared/
├── line/
│   ├── images/*.jpg
│   ├── intensity_image/*.png
│   ├── sd_map_infos_train.pkl
│   └── sd_map_infos_val.pkl
├── box/          # 相同结构
└── freespace/    # 相同结构
```

三类数据分别输出两份 pickle，类别与属性词表互相独立，不混为一个训练任务。

## 裁剪与划分

- 原图固定为 6000×6000，裁块为 1000×1000；不缩放、旋转或补充网格采样。
- 沿轨迹弧长均匀采样：line/box 每条 7 点，freespace 每条 14 点；少于两个点的轨迹跳过。
- 中心限制到 `[500, 5500]`，先 round 再减 500。GT 使用像素中心边界 `[left, top, left+999, top+999]`。
- line：折线与窗口求交，分裂时输出多个实例。闭合线沿用原版 `Polygon.buffer(0.001)`，裁剪后的外轮廓去掉重复闭合点。
- box：只保留裁剪后仍有四个顶点的 Polygon。Arrow 还必须满足原版完整顶点与点序检查；其他类别允许满足条件的部分裁剪框。
- freespace：Polygon 求交，分裂为多个区域时分别输出外轮廓，保留重复闭合点；不保存孔洞。修正了原版 GeometryCollection 分支错误访问集合 exterior 的问题。
- 强度底图复用完全相同的窗口、空 GT 筛选和 train/val 划分，以无损 PNG 保存；不会对 JPEG 裁块再次反查。
- 坐标减去裁块左上角，保留浮点精度及原版裁剪产生的点序，不重采样、不归一化。
- 跳过没有保留实例的窗口；保留重复和重叠窗口，编号在过滤前递增，因此文件名可能不连续。
- 按来源大图划分：前 `round(0.8*N)` 张为 train，其余为 val。line 按 JSON 顺序，且先移除无有效裁块的大图；box/freespace 按图名排序，计入无有效裁块的大图。与原版一致，同一大图不会跨集合。

## pickle 协议

使用 `metainfo + data_list`，不存 Shapely 或 Tensor：

- `metainfo`：`dataset_type='bev_image_map'`、`info_version='1.0'`、尺寸、`classes`、`attribute_schema`、`has_annotations=True`、`has_intensity=True`、`intensity_encoding='estimated_normalized_jet_index_uint8'`、`split`、来源大图列表及裁剪规则。
- 样本：`sample_id`、`img_path`、`height/width`、`source_id`、`crop_origin`、原图/轨迹路径、轨迹编号、采样编号及 `instances`。
- 实例：`label` 为类别表下标，`points` 为裁块像素坐标系的有序二维点，`attrs` 为属性字典，`source_instance_id` 为该大图中解析后的实例编号。同一实例裁出的多个片段保留相同编号，与 `source_id` 一起追溯。
- `img_path` 和 `intensity_img_path` 相对于该类型的输出根目录；`source_intensity_img_path`、原图和轨迹路径相对于原始 SD 根目录。
- line 保留六个监督属性及原始有用的 `lane_color`；box 的 Arrow/ExclusiveLaneSign 属性分别存为 `arrow_type` / `exclusive_lane_type`，其他类别为 `{}`；freespace 属性为 `{}`。词表顺序与原配置一致。

GT 的固定点数采样、匹配点序、属性编码属于后续训练 transform。当前 pipeline 已加载并打包原始 GT 到 `gt_instances`；`test.py` 用于 Dataset 初始化和 GT 叠加可视化；GT 打包一致性由单元测试检查。训练目标尚未生成。

## 验证

本机输出位于 `/home/guyu/data/vma/sd_data/prepared`：

| 类型 | train 图块 / 实例 | val 图块 / 实例 |
| --- | --- | --- |
| line | 35 / 422 | 7 / 40 |
| box | 22 / 70 | 5 / 29 |
| freespace | 18 / 37 | 3 / 3 |

共 90 张图块、601 个实例。在 `vma_legacy:blackwell-cu128` 中与原版三套脚本逐项比较：所有裁块的划分、文件名、偏移、JPEG 字节、GT 坐标、类别和规范化后的属性完全一致。历史比较记录曾保存为 `work_dirs/vma/sd_conversion/legacy_gt_comparison.json`；运行产物可清理，不作为 converter 的输入依赖。

转换器相关测试已通过，覆盖采样、三类 GT 裁剪、分裂与闭合、Arrow 过滤、来源划分、属性转换、缺失文件、失败回滚和旧索引清理。以下第一条执行单元测试，后两条仅初始化 Dataset：

```bash
python -m unittest discover -s tests/test_datasets -p test_sd_converter.py -v
python projects/vma/tools/test.py projects/vma/configs/sd_images.py --split train
python projects/vma/tools/test.py projects/vma/configs/sd_images.py --split val
```
