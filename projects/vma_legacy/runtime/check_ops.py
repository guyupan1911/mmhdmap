"""Check the reused CUDA kernel against MMCV's PyTorch reference."""
import json
import torch
import vma_runtime  # noqa: F401
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)

torch.manual_seed(0)
shapes = torch.tensor([[4, 4], [2, 2]], device='cuda', dtype=torch.long)
starts = torch.tensor([0, 16], device='cuda', dtype=torch.long)
value = torch.randn(1, 20, 2, 4, device='cuda', requires_grad=True)
locations = torch.rand(1, 5, 2, 2, 3, 2, device='cuda', requires_grad=True)
weights = torch.rand(1, 5, 2, 2, 3, device='cuda', requires_grad=True)
cuda_out = MultiScaleDeformableAttnFunction.apply(value, shapes, starts,
                                                  locations, weights, 64)
ref_out = multi_scale_deformable_attn_pytorch(value, shapes, locations,
                                              weights)
cuda_grads = torch.autograd.grad(
    cuda_out.square().sum(), (value, locations, weights), retain_graph=True)
ref_grads = torch.autograd.grad(ref_out.square().sum(),
                                (value, locations, weights))
torch.testing.assert_close(cuda_out, ref_out, atol=1e-5, rtol=1e-4)
for a, b in zip(cuda_grads, ref_grads):
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
print(
    json.dumps(
        dict(
            forward_max_abs=float((cuda_out - ref_out).abs().max()),
            backward_max_abs=[
                float((a - b).abs().max())
                for a, b in zip(cuda_grads, ref_grads)
            ],
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name())))
