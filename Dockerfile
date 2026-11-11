# Preparation image; build/run verification is pending Docker daemon access.
FROM python:3.11-slim-bookworm@sha256:8a00ecb83ed9d0864dfd102a1bdb2067fda907ceb30bb86d09da9ee558b19590
ENV PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="" MPLCONFIGDIR=/tmp/qaml-matplotlib
WORKDIR /app
COPY kit/requirements-cpu.lock ./requirements-cpu.lock
RUN python -m pip install --no-cache-dir -r requirements-cpu.lock && python -m pip check
COPY kit ./kit
COPY client ./client
COPY examples ./examples
RUN chmod -R a+rX /app
USER 10001:10001
CMD ["python", "-c", "from kit.sdk import verify_assets; print(verify_assets()['status'])"]
