# hnswlib on PyPI is distributed as source only: there is no wheel for this interpreter,
# and in a slim image without a compiler the build fails with "Unsupported compiler".
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir hnswlib==0.8.0 numpy
COPY example.py /example.py
CMD ["python", "/example.py"]
