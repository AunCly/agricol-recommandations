FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# On copie directement depuis la racine du contexte
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# On copie tout le reste du code
COPY . .

EXPOSE 7860 8501

# FastAPI par défaut ; le service streamlit du docker-compose surcharge cette commande.
# Forme « shell » pour lire $PORT (fourni par Render), 7860 par défaut (docker compose).
CMD uv run fastapi run api/main.py --host 0.0.0.0 --port ${PORT:-7860}