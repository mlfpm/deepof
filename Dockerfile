# syntax=docker/dockerfile:1
#
# Self-contained DeepOF image for long-term reproducibility. Contains the exact dependency versions of
# poetry.lock (plus the CUDA build of PyTorch, which also runs on machines without GPU) and the pretrained
# models, so that running it needs no package index or download server.
#
# Build:   docker build -t deepof .
# Run:     docker run -it --rm -v /path/to/data:/data deepof
# GPU:     docker run -it --rm --gpus all -v /path/to/data:/data deepof
# Jupyter: docker run -it --rm -p 8888:8888 -v /path/to/data:/data deepof jupyter lab --ip=0.0.0.0 --no-browser --allow-root
#
# Layers are ordered from rarely to frequently changing (PyTorch -> models -> dependencies -> DeepOF), so that
# consecutive releases share the large layers in the container registry.

ARG PYTHON_IMAGE=python:3.10.22-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370


# ---- Stage 1: turn the locked dependencies into wheels -------------------------------------------------------
FROM ${PYTHON_IMAGE} AS wheels

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir poetry==2.3.3 poetry-plugin-export==1.9.0

WORKDIR /src
COPY pyproject.toml poetry.lock README.md ./

# The lock file pins the CPU build of torch on Linux. torch and torchvision are therefore installed separately
# from the CUDA index (same version), together with the NCCL version that this torch build requires.
# All other packages are installed exactly as locked (main + dev group, the latter contains jupyter and pytest).
RUN poetry export --with dev --format requirements.txt --output requirements_all.txt \
 && sed -nE 's/^(torch|torchvision)==([^ +;]+).*/\1==\2/p' requirements_all.txt | sort -u > torch_requirements.txt \
 && awk '/^[^ ]/ {skip = ($0 ~ /^(torch|torchvision|nvidia-nccl-cu12)==/)} !skip' requirements_all.txt > requirements.txt \
 && test "$(wc -l < torch_requirements.txt)" -eq 2 \
 && pip wheel --no-cache-dir --no-deps --require-hashes -r requirements.txt -w /wheels/deps

COPY deepof ./deepof
RUN poetry build --format wheel \
 && mkdir -p /wheels/deepof \
 && cp dist/*.whl /wheels/deepof/


# ---- Stage 2: pretrained models (otherwise downloaded at runtime) -------------------------------------------
FROM ${PYTHON_IMAGE} AS models

COPY docker/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py /models/deepof


# ---- Final image ---------------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG SITE_PACKAGES=/usr/local/lib/python3.10/site-packages

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=wheels /src/torch_requirements.txt /opt/deepof/torch_requirements.txt
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} -r /opt/deepof/torch_requirements.txt

COPY --from=models /models/ ${SITE_PACKAGES}/

COPY --from=wheels /src/requirements.txt /opt/deepof/requirements.txt
RUN --mount=type=bind,from=wheels,source=/wheels/deps,target=/tmp/wheels \
    pip install --no-cache-dir --no-deps --no-index /tmp/wheels/*.whl

COPY --from=wheels /wheels/deepof/ /opt/deepof/
RUN pip install --no-cache-dir --no-deps --no-index /opt/deepof/*.whl \
 && pip check

# Writable locations for caches, so the image also works with "docker run --user ..." and under Apptainer
ENV HOME=/home/deepof \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTHONUNBUFFERED=1
RUN mkdir -p /home/deepof /data && chmod 1777 /home/deepof /data

ARG DEEPOF_VERSION=unknown
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="DeepOF" \
      org.opencontainers.image.version="${DEEPOF_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://gitlab.mpcdf.mpg.de/lucasmir/deepof" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /data
EXPOSE 8888
CMD ["bash"]
