FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY train.py generate.py evaluate.py ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY constraints ./constraints
COPY data/tokenizer_final/tokenizer.json ./data/tokenizer_final/tokenizer.json

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu -c constraints/verified-py311-windows.txt \
    && python -m pip install ".[test]" ruff==0.11.13 -c constraints/verified-py311-windows.txt

COPY docs ./docs

# Subprocess entrypoints must resolve the same checkout/configs as pytest.
ENV PYTHONPATH=/workspace/src

CMD ["python", "-m", "pytest", "-q"]
