"""Recover approximate uint8 intensity indices from SD's JET-colored JPGs.

Original physical LiDAR intensity cannot be recovered without its scaling
parameters. JPEG artifacts and deviations from OpenCV JET also cause error.
No log mapping, contrast stretching or background masking is applied.
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

KINDS = ('line', 'box', 'freespace')
METHOD = 'nearest_opencv_jet_rgb_squared_distance'


class InverseJet:
    """Exact nearest LUT entry, caching the mapping for each 24-bit color."""

    def __init__(self):
        self.palette = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8)[:, None],
            cv2.COLORMAP_JET).reshape(256, 3).astype(np.int32)
        self.cache = np.full(1 << 24, -1, dtype=np.int16)

    def __call__(self, bgr):
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError('Expected uint8 BGR HWC image')
        codes = bgr[..., 0].astype(np.int32)
        codes |= bgr[..., 1].astype(np.int32) << 8
        codes |= bgr[..., 2].astype(np.int32) << 16
        unique = np.unique(codes)
        missing = unique[self.cache[unique] < 0]
        for start in range(0, len(missing), 4096):
            block = missing[start:start + 4096]
            distances = np.zeros((len(block), 256), dtype=np.int32)
            for channel in range(3):
                values = (block >> (8 * channel)) & 255
                delta = values[:, None] - self.palette[None, :, channel]
                distances += delta * delta
            # argmin resolves exact ties to the lowest index, deterministically.
            self.cache[block] = distances.argmin(axis=1)
        return self.cache[codes].astype(np.uint8)


def recover_sd_intensity(data_root, overwrite=False):
    root = Path(data_root).expanduser().resolve()
    sources = [
        p for kind in KINDS
        for p in sorted((root / kind / 'origin_data/image_data').glob('*.jpg'))
    ]
    if not sources:
        raise FileNotFoundError(f'No SD source images under {root}')
    for source in sources:
        output = source.parent.parent / 'intensity_image' / f'{source.stem}.png'
        if output.exists() and not overwrite:
            raise FileExistsError(f'{output} exists; use --overwrite')
    inverse = InverseJet()
    reports = {kind: [] for kind in KINDS}
    for source in sources:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.shape != (6000, 6000, 3):
            raise ValueError(f'{source}: expected 6000x6000 BGR image')
        intensity = inverse(image)
        output = source.parent.parent / 'intensity_image' / f'{source.stem}.png'
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            suffix='.png', prefix='.intensity-', dir=output.parent)
        os.close(fd)
        try:
            if not cv2.imwrite(temporary, intensity):
                raise OSError(output)
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        kind = source.relative_to(root).parts[0]
        reports[kind].append(
            dict(
                source=source.relative_to(root).as_posix(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                output=output.relative_to(root).as_posix(),
                output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                shape=list(intensity.shape)))
        print(f'{kind}/{source.name} -> {output.name}', flush=True)
    for kind, images in reports.items():
        if images:
            manifest = dict(
                method=METHOD,
                opencv_version=cv2.__version__,
                interpretation='estimated_normalized_jet_index',
                dtype='uint8',
                log_mapping=False,
                images=images)
            path = root / kind / 'origin_data/intensity_image/recovery_manifest.json'
            path.write_text(json.dumps(manifest, indent=2) + '\n')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    recover_sd_intensity(args.data_root, args.overwrite)
