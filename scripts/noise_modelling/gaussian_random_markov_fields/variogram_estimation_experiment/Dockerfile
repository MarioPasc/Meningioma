# Use a base Python 3.9 (or 3.10) image
FROM python:3.9-slim

# Create a working directory
WORKDIR /app

# Install any system dependencies you might need (e.g., for OpenCV, building wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your Meningioma project (including pyproject.toml, src/, scripts/, etc.) into /app
COPY . /app

# Upgrade pip, then install your package using pyproject.toml
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .

# Run the "variogram_fitting_experiment.py" script
ENTRYPOINT ["python", "scripts/noise_modelling/gaussian_random_markov_fields/variogram_estimation_experiment/variogram_fitting_experiment.py"]

# Build with:
# docker build --no-cache -t mpascualg/variogram_fitting:latest -f ./scripts/noise_modelling/gaussian_random_markov_fields/variogram_estimation_experiment/Dockerfile .