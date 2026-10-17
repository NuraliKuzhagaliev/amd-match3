FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4

COPY app/requirements.txt /app/requirements.txt
COPY runtime-pins.txt /tmp/runtime-pins.txt
COPY scripts/install_dependencies.py /tmp/install_dependencies.py
RUN python3 /tmp/install_dependencies.py /tmp/runtime-pins.txt

COPY models.lock.json /app/models.lock.json
COPY scripts/download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py --lock /app/models.lock.json --destination /models

COPY app/ /app/
COPY licenses/ /app/licenses/
COPY THIRD_PARTY_NOTICES.md /app/THIRD_PARTY_NOTICES.md
RUN mkdir -p /app/corpus /app/output /app/index
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    MC3_MODEL_DIR=/models/vision MC3_EMBED_DIR=/models/embedding

# Retain the mandated base layers. No exposed network port or runtime downloads.
CMD ["python3", "-m", "mc3.worker"]
