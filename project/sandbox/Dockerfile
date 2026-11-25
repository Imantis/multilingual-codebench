FROM python:3.12.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY worker.py runner.py /opt/
WORKDIR /work
USER 65534:65534
ENTRYPOINT ["python", "-I", "/opt/runner.py"]
