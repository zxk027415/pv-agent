FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

WORKDIR /app
COPY requirements/docker-universal.txt requirements/
RUN uv pip install --system --no-cache -r requirements/docker-universal.txt

# Code, prompts, playbooks and the licensed kb snapshot; observations are mounted read-only.
# Run from the source layout (not site-packages): modules locate harness/, kb/ and
# playbooks/ relative to src/.
COPY src src
COPY harness harness
COPY playbooks playbooks
COPY kb kb
COPY config config
ENV PYTHONPATH=/app/src
RUN useradd --create-home --uid 10001 pvdiag
USER pvdiag

EXPOSE 8000
CMD ["python", "-m", "pvdiag.api", "serve", "--host", "0.0.0.0", "--profiles", "config/api.example.json"]
