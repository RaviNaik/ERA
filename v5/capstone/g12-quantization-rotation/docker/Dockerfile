# Build tools only. The code, the models, the venvs and the results stay OUTSIDE the image, in mounted folders:
#   /app   = this folder (config.sh, scripts, results/)          -> results are written straight to the host
#   /work  = WORK_DIR (llama.cpp build, venvs, HF cache, GGUFs)  -> model files stay on the host and survive rebuilds
# CUDA 12.4 needs a host driver >= 550. For an older driver pass --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WORK_DIR=/work \
    HOME=/work/home \
    XDG_CACHE_HOME=/work/home/.cache \
    HF_HOME=/work/home/.cache/huggingface \
    UV_CACHE_DIR=/work/home/.cache/uv \
    UV_PYTHON_INSTALL_DIR=/work/home/.uv-python \
    PATH=/work/home/.local/bin:/usr/local/cuda/bin:${PATH}

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates build-essential cmake pkg-config \
        python3 python3-pip python3-venv less nano \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m pip install --no-cache-dir "uv==0.13.6"

# HOME lives in the /work volume, so the container can run as any user (the host user, see run_docker.sh)
WORKDIR /app
CMD ["./run_all.sh"]
