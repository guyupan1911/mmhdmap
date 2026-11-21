"""GPU validation of SD-line VMA using original model/loss/dataset code."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import vma_runtime  # noqa: F401
import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet.models import build_detector
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from projects.mmdet3d_plugin.datasets.sd_driving_line_dataset import SD_Driving_Line_Dataset


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def model_load(args, cfg):
    cfg.model.pretrained = None
    model = build_detector(cfg.model)
    model.init_weights()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = ckpt['state_dict']
    own = model.state_dict()
    missing = sorted(set(own) - set(state))
    extra = sorted(set(state) - set(own))
    mismatch = {
        k: [list(state[k].shape), list(own[k].shape)]
        for k in own.keys() & state.keys() if state[k].shape != own[k].shape
    }
    report = dict(
        missing=missing,
        extra=extra,
        shape_mismatch=mismatch,
        model_keys=len(own),
        checkpoint_keys=len(state))
    dump(args.output / 'checkpoint_audit.json', report)
    print('CHECKPOINT_AUDIT', json.dumps(report), flush=True)
    if args.mode == 'audit':
        return model
    if missing or mismatch or extra:
        raise RuntimeError(
            'Checkpoint mismatch: inspect checkpoint_audit.json before running'
        )
    model.load_state_dict(state, strict=True)
    if args.reset_head:
        # Independent initialization establishes that the training path learns,
        # rather than merely starting with memorized predictions.
        fresh = build_detector(cfg.model)
        fresh.init_weights()
        model.pts_bbox_head.load_state_dict(fresh.pts_bbox_head.state_dict())
        del fresh
    del ckpt, state
    return model.cuda()


def dataset_load(args, cfg):
    ds_cfg = dict(cfg.data.train)
    ds_cfg.update(data_root=str(args.crops), mode='train', test_mode=False)
    ds_cfg.pop('type')
    ds = SD_Driving_Line_Dataset(**ds_cfg)
    if args.mode == 'infer':
        valid = SD_Driving_Line_Dataset(**dict(ds_cfg, mode='valid'))
        ds.annotation_dict.update(valid.annotation_dict)
        ds.seq_len = len(ds.annotation_dict)
    return ds


def batch(ds, idx):
    ex = ds[idx]
    img = torch.as_tensor(ex['img']).unsqueeze(0).cuda()
    meta = ex['img_metas'].data
    return dict(
        img=img,
        img_metas=[meta],
        gt_labels=[ex['gt_labels'].data.cuda()],
        gt_bboxes=[ex['gt_bboxes'].data],
        gt_attrs=[ex['gt_attrs'].data.cuda()])


def predict(model, data):
    model.eval()
    with torch.no_grad():
        result = model(
            return_loss=False, img=data['img'], img_metas=data['img_metas'])
    return result[0]['pts_bbox']


def measure(pred, data):
    keep = pred['scores_3d'].numpy() >= .3
    pts = pred['pts_3d'].numpy()[keep]
    labels = pred['labels_3d'].numpy()[keep]
    gt = data['gt_bboxes'][0]
    gt_labels = data['gt_labels'][0].cpu().numpy()
    gpts = []
    for geom in gt.instance_list:
        gpts.append(
            np.array([
                geom.interpolate(d).coords[0]
                for d in np.linspace(0, geom.length, 100)
            ]))
    cost = np.full((len(gpts), len(pts)), 1e6)
    # Dense interpolation makes the metric independent of the output's point spacing.
    from shapely.geometry import LineString
    for j, pp in enumerate(pts):
        geom = LineString(pp)
        pp = np.array([
            geom.interpolate(d).coords[0]
            for d in np.linspace(0, geom.length, 100)
        ])
        for i, gp in enumerate(gpts):
            if gt_labels[i] == labels[j]:
                dist = cdist(gp, pp)
                cost[i, j] = (dist.min(0).mean() + dist.min(1).mean()) / 2
    rows, cols = linear_sum_assignment(cost)
    distances = cost[rows, cols]
    matched = distances[distances < 1e5]
    metrics = dict(
        gt=len(gpts),
        predictions=int(keep.sum()),
        matched_chamfer_px=float(matched.mean()) if len(matched) else None)
    for threshold in (5, 10, 20):
        tp = int((distances <= threshold).sum())
        metrics[f'precision_{threshold}px'] = tp / max(len(pts), 1)
        metrics[f'recall_{threshold}px'] = tp / max(len(gpts), 1)
    return metrics


def visualize(path, pred, data):
    background = cv2.imread(data['img_metas'][0]['filename'])
    gt_image, pred_image = background.copy(), background.copy()
    colors = [(0, 255, 0), (255, 128, 0), (0, 128, 255)]
    for geom, label in zip(data['gt_bboxes'][0].instance_list,
                           data['gt_labels'][0].cpu().tolist()):
        pts = np.round(np.array(geom.coords)).astype(np.int32)
        cv2.polylines(gt_image, [pts], False, colors[label], 3)
    for pts, label, score in zip(pred['pts_3d'], pred['labels_3d'],
                                 pred['scores_3d']):
        if score < .3:
            continue
        cv2.polylines(pred_image, [np.round(pts.numpy()).astype(np.int32)],
                      False, colors[int(label)], 3)
    for image, label in ((gt_image, 'GT'), (pred_image,
                                            'Prediction (score >= 0.3)')):
        cv2.rectangle(image, (0, 0), (600, 42), (0, 0, 0), -1)
        cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .8,
                    (255, 255, 255), 2)
    cv2.imwrite(str(path), np.concatenate([gt_image, pred_image], axis=1))


def export_prediction(path, pred, data):
    dump(
        path,
        dict(
            image=data['img_metas'][0]['filename'],
            scores=pred['scores_3d'].tolist(),
            labels=pred['labels_3d'].tolist(),
            points=pred['pts_3d'].tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['audit', 'infer', 'overfit'])
    parser.add_argument('--checkpoint', default='/data/sd_ckpts/sd_line.pth')
    parser.add_argument('--crops', type=Path, default=Path('/work_dirs/crops'))
    parser.add_argument('--output', type=Path, default=Path('/work_dirs'))
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--index', type=int, default=-1)
    parser.add_argument('--reset-head', action='store_true')
    parser.add_argument(
        '--disable-dropout',
        action='store_true',
        help='Disable stochastic regularization for the memorization diagnostic'
    )
    parser.add_argument('--cosine-lr', action='store_true')
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(
        str(
            Path(os.environ.get('VMA_RUNTIME_ROOT', '/opt/VMA')) /
            'projects/configs/vma_res152_e80_line.py'))
    model = model_load(args, cfg)
    if args.mode == 'audit':
        return
    dump(
        args.output / 'environment.json',
        dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            mmcv=mmcv.__version__,
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability()))
    ds = dataset_load(args, cfg)
    if args.mode == 'infer':
        results = []
        indices = [args.index] if args.index >= 0 else range(len(ds))
        for idx in indices:
            data = batch(ds, idx)
            pred = predict(model, data)
            metrics = measure(pred, data)
            metrics.update(
                index=idx, image=Path(data['img_metas'][0]['filename']).name)
            results.append(metrics)
            visualize(args.output / f'infer_{idx:03d}.jpg', pred, data)
            export_prediction(args.output / f'infer_{idx:03d}.json', pred,
                              data)
            print('INFERENCE', json.dumps(metrics), flush=True)
            dump(args.output / 'inference_metrics.json', results)
        return
    # Pick a modest, nonempty crop with both lane/curb where available.
    annotations = list(ds.annotation_dict.values())
    idx = args.index if args.index >= 0 else min(
        range(len(ds)),
        key=lambda i: abs(len(annotations[i]['instances']) - 8) +
        (100 if not annotations[i]['instances'] else 0))
    data = batch(ds, idx)
    model.use_grid_mask = False
    if args.disable_dropout:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
            elif isinstance(module, torch.nn.MultiheadAttention):
                module.dropout = 0.
    initial = measure(predict(model, data), data)
    visualize(args.output / 'overfit_before.jpg', predict(model, data), data)
    groups = [
        dict(
            params=[
                p for n, p in model.named_parameters()
                if p.requires_grad and n.startswith('img_backbone.')
            ],
            lr=args.lr * .1),
        dict(
            params=[
                p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith('img_backbone.')
            ],
            lr=args.lr)
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    history = []
    start = time.time()
    for step in range(args.steps):
        if args.cosine_lr:
            factor = .05 + .95 * (1 + np.cos(np.pi * step / args.steps)) / 2
            for group, multiplier in zip(optimizer.param_groups, (.1, 1.)):
                group['lr'] = args.lr * multiplier * factor
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = model(return_loss=True, **data)
        total, logs = model._parse_losses(losses)
        if not torch.isfinite(total):
            raise RuntimeError(f'Nonfinite loss at step {step}: {logs}')
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 35.)
        if not torch.isfinite(grad):
            raise RuntimeError(f'Nonfinite gradient at step {step}')
        optimizer.step()
        if step % 10 == 0 or step == args.steps - 1:
            row = dict(
                step=step + 1,
                seconds=time.time() - start,
                grad_norm=float(grad),
                **logs)
            history.append(row)
            print('TRAIN', json.dumps(row), flush=True)
            dump(args.output / 'overfit_history.json', history)
        if (step + 1) % 50 == 0 or step == args.steps - 1:
            pred = predict(model, data)
            print(
                'EVAL', step + 1, json.dumps(measure(pred, data)), flush=True)
            visualize(args.output / f'overfit_{step+1:04d}.jpg', pred, data)
    final_pred = predict(model, data)
    export_prediction(args.output / 'overfit_final.json', final_pred, data)
    final = measure(final_pred, data)
    torch.save(
        dict(
            state_dict={
                k: v.cpu()
                for k, v in model.state_dict().items()
            },
            meta=dict(steps=args.steps, index=idx,
                      reset_head=args.reset_head)),
        args.output / 'overfit_line.pth')
    dump(
        args.output / 'overfit_summary.json',
        dict(
            index=idx,
            image=data['img_metas'][0]['filename'],
            steps=args.steps,
            reset_head=args.reset_head,
            disable_dropout=args.disable_dropout,
            cosine_lr=args.cosine_lr,
            lr=args.lr,
            initial=initial,
            final=final,
            seconds=time.time() - start,
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3))


if __name__ == '__main__':
    main()
