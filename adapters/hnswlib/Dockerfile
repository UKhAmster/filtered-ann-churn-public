# hnswlib на PyPI распространяется только исходниками: wheel под этот интерпретатор
# отсутствует, и в slim-образе без компилятора сборка падает на "Unsupported compiler".
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir hnswlib==0.8.0 numpy
COPY example.py /example.py
CMD ["python", "/example.py"]
