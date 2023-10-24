# Base image
FROM runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel

ARG HUGGING_FACE_HUB_WRITE_TOKEN
ENV HUGGING_FACE_HUB_WRITE_TOKEN=$HUGGING_FACE_HUB_WRITE_TOKEN

ENV HF_HOME = "/cache/huggingface/"
ENV HF_DATASETS_CACHE = "/cache/huggingface/datasets/"
ENV DEFAULT_HF_METRICS_CACHE = "/cache/huggingface/metrics/"
ENV DEFAULT_HF_MODULES_CACHE = "/cache/huggingface/modules/"

ENV HUGGINFACE_HUB_CACHE = "/cache/huggingface/hub/"
ENV HUGGINGFACE_ASSETS_CACHE = "/cache/huggingface/assets/"

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /workspace

# Install Python Dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    rm /requirements.txt

# SDXL Training Script
COPY src/train_dreambooth_lora_sdxl.py /workspace/src/train_dreambooth_lora_sdxl.py

# Cache Models
COPY builder/cache_model.py /cache_model.py
RUN python /cache_model.py && \
    rm /cache_model.py

# Copy Source Code
ADD src .

# Basic validation
# Verify that the cache folder is not empty
RUN test -n "$(ls -A /cache/huggingface/modules/)" && \
    test -n "$(ls -A /cache/huggingface/datasets/)" && \
    test -n "$(ls -A /cache/huggingface/metrics/)" && \
    test -n "$(ls -A /cache/huggingface/hub/)" && \
    test -n "$(ls -A /cache/huggingface/assets/)" && \
    test -n "$(ls -A /cache/huggingface/)" && \
    test -n "$(ls -A /cache/)" && \
    test -n "$(ls -A /workspace/src/)" && \
    test -n "$(ls -A /workspace/)" && \
    test -n "$(ls -A /)"


CMD ["python", "-u", "handler.py"]
