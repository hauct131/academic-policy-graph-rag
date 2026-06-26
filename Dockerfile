# ── Stage: production image ───────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# ── Dependency layer (cached separately from application source) ───────────────
# Copy only the requirements file first so Docker can cache this layer
# independently of application code changes.
COPY requirements-api.txt ./

RUN pip install --no-cache-dir -r requirements-api.txt

# ── Application source ────────────────────────────────────────────────────────
# Copy application packages and supporting project files.
# data/ and domains/ are intentionally excluded here; they are mounted at
# runtime via docker-compose volumes so that host-generated artifacts (chunks,
# graphs, eval results) are visible inside the container without rebuilding.
COPY app/       ./app/
COPY core/      ./core/
COPY scripts/   ./scripts/
COPY domains/   ./domains/

# ── Runtime configuration ─────────────────────────────────────────────────────
# Expose the uvicorn port
EXPOSE 8000

# Default command — overridable in docker-compose
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
