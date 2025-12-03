# Multi-stage Dockerfile to speed up builds by prebuilding wheels
# Stage 1: builder - install build deps and build wheels for requirements
# Stage 2: runtime - install only from wheels and run app

FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /wheels

# Install build deps needed to compile some packages/wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       gfortran \
       libopenblas-dev \
       liblapack-dev \
       git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements and build wheels into /wheels
COPY requirements.txt /wheels/requirements.txt
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels -r /wheels/requirements.txt || true

# Stage 2: runtime image
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal runtime deps (for some wheels that need system libs)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libopenblas-dev \
       liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy built wheels from builder
COPY --from=builder /wheels /wheels

# Install Python packages from wheels if available, otherwise pip will fall back to PyPI
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt

# Copy app code
COPY . /app

# Expose port and run
EXPOSE 5000
CMD ["python", "src/app.py"]