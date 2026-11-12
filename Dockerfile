FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.11.6
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 HF_HOME=/app/.cache/huggingface

# Dependencies first, so code changes do not invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY app ./app
COPY evals ./evals
COPY data/sample ./data/sample
COPY results ./results
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501
CMD ["uvicorn", "govqa.api:app", "--host", "0.0.0.0", "--port", "8000"]
