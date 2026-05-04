This environment was tested on NVIDIA Tesla V100 GPUs.

The causal-conv1d and mamba-ssm extensions are compiled for compute capability sm_70; if you use a different GPU, rebuild the image with TORCH_CUDA_ARCH_LIST set to your architecture.

If training fails with a CUDNN_STATUS_NOT_INITIALIZED error, comment out the `total_flops = flops.total()` line in `trainer.py` to disable FLOPs estimation.

This does not affect training results.