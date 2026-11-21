"""Patch a disposable copy of the read-only upstream checkout, then run."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source = Path('/workspace/VMA')
if not (source / 'projects/configs/vma_res152_e80_line.py').is_file():
    raise SystemExit(
        'Mount the original VMA repo at /workspace/VMA (see docker-compose.yaml).'
    )
root = Path(tempfile.mkdtemp(prefix='vma-legacy-')) / 'VMA'
shutil.copytree(
    source,
    root,
    ignore=shutil.ignore_patterns('.git', '__pycache__', 'work_dirs', '.venv'))
os.environ.update(
    VMA_RUNTIME_ROOT=str(root),
    VMA_PATCH_OPS='0',
    VMA_PATCH_MANIFEST=str(root / 'legacy_patch_manifest.json'),
    PYTHONPATH=f'{root}:/opt/vma-support',
    PYTHONDONTWRITEBYTECODE='1')
subprocess.run([sys.executable, '/opt/vma-support/patch_runtime.py'],
               check=True)
os.chdir(root)
args = sys.argv[1:] or ['bash']
os.execvp(args[0], args)
