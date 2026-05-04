FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git wget curl unzip build-essential

ENV VIRTUAL_ENV=/opt/venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    packaging ninja wheel setuptools \
    einops "numpy<2" tqdm timm PyWavelets xarray netCDF4 python-dateutil thop torchinfo fvcore \
    transformers==4.38.2

WORKDIR /workspace

RUN git clone https://github.com/Dao-AILab/causal-conv1d.git && \
    cd causal-conv1d && git checkout v1.4.0 && \
    TORCH_CUDA_ARCH_LIST="7.0" MAX_JOBS=1 pip install --no-cache-dir . --no-build-isolation --no-deps

RUN git clone https://github.com/state-spaces/mamba.git && \
    cd mamba && git checkout v1.2.0.post1 && \
    TORCH_CUDA_ARCH_LIST="7.0" MAX_JOBS=1 pip install --no-cache-dir . --no-build-isolation --no-deps

RUN git clone https://github.com/oucailab/FH-Mamba.git /workspace/FH-Mamba

WORKDIR /workspace/FH-Mamba