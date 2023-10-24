# Base image
FROM runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel

ARG HUGGING_FACE_HUB_WRITE_TOKEN
ENV HUGGING_FACE_HUB_WRITE_TOKEN=$HUGGING_FACE_HUB_WRITE_TOKEN

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


ADD src .

CMD ["bash", "-c", "accelerate config default && python -u handler.py"]
