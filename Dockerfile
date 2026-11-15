FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/

ENV PATH="/code/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /code
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

# model.bin is not in git, so the model is trained while building the image
COPY features.py train.py predict.py ./
COPY data/snapshots.csv data/
RUN python train.py

EXPOSE 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
