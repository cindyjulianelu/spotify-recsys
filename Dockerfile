FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/       ./src/
COPY app/       ./app/
COPY setup.py   .

# data/ and models/ are mounted as volumes at runtime (see docker-compose.yml)
# so they are not baked into the image — models stay out of the image layer,
# and data can be refreshed without a rebuild.
RUN mkdir -p data models

EXPOSE 8501

# Health check — Streamlit exposes /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
