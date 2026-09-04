FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY data/tokenizer_final/tokenizer.json ./data/tokenizer_final/tokenizer.json

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[test]"

CMD ["python", "-m", "pytest", "-q"]
