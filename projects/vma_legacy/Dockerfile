# Original OpenMMLab 1.x VMA API on this host's Blackwell GPU.
# The base provides torch 2.7.0/cu128 and sm_120-compiled MMCV CUDA kernels.
ARG BASE_IMAGE=mmhdmap-dev:latest
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root
RUN python - <<'PY'
import glob, shutil
import mmcv, torch
assert mmcv.__version__ == '2.1.0', mmcv.__version__
assert torch.__version__.startswith('2.7.0') and torch.version.cuda == '12.8'
for p in glob.glob(mmcv.__path__[0] + '/_ext*.so'):
    shutil.copy2(p, '/tmp/' + p.rsplit('/', 1)[-1])
shutil.copy2(mmcv.__path__[0] + '/ops/corner_pool.py', '/tmp/corner_pool.py')
PY
RUN pip uninstall -y mmdet mmdet3d mmcv \
    && pip install --no-cache-dir 'setuptools<70' 'numpy==1.26.4' 'yapf==0.40.1' \
       'shapely==1.8.5.post1' 'opencv-python==4.8.0.76' 'similaritymeasures==0.6.0' \
    && pip install --no-cache-dir --no-build-isolation 'mmcv==1.4.0' \
    && pip install --no-cache-dir --no-deps 'mmdet==2.14.0' 'mmsegmentation==0.14.1' \
    && python - <<'PY'
import glob, shutil, mmcv
for p in glob.glob('/tmp/_ext*.so'):
    shutil.copy2(p, mmcv.__path__[0])
shutil.copy2('/tmp/corner_pool.py', mmcv.__path__[0] + '/ops/corner_pool.py')
PY
COPY --from=vma_source /projects /opt/VMA/projects
COPY --from=vma_source /tools /opt/VMA/tools
COPY --from=vma_source /mmdetection3d /opt/VMA/mmdetection3d
COPY runtime /opt/vma-support
RUN python /opt/vma-support/patch_runtime.py
ENV PYTHONPATH=/opt/VMA:/opt/vma-support
ENV MPLBACKEND=Agg OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
WORKDIR /opt/VMA
RUN python -c "import vma_runtime; import torch, mmcv, mmdet; print(torch.__version__, mmcv.__version__, mmdet.__version__)"
ENTRYPOINT ["python", "/opt/vma-support/entrypoint.py"]
CMD ["bash"]
