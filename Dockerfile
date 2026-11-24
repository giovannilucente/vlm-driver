FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade \
    pip==23.3.2 \
    setuptools==68.2.2 \
    wheel==0.41.3

# Keep PyTorch fixed
RUN python -m pip install \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install HF stack without touching torch
RUN python -m pip install --no-deps \
    transformers==4.45.2 \
    accelerate==0.27.2 

# Install required dependencies manually
RUN python -m pip install \
    sentencepiece==0.1.99 \
    protobuf==4.25.3 \
    einops==0.7.0 \
    pillow==10.2.0 \
    huggingface_hub==0.24.5 \
    jinja2==3.1.6 \
    psutil==5.9.8 \
    regex==2023.12.25 \
    tokenizers==0.20.1 \
    safetensors==0.4.5 \
    pyyaml==6.0.1 \
    packaging==23.2 \
    filelock==3.13.1 \
    tqdm==4.66.1

WORKDIR /workspace

CMD ["/bin/bash"]
