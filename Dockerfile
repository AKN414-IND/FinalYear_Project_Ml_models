# Use Python 3.8 as base image (stable version for ML applications)
FROM python:3.8-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    python3-dev \
    gcc \
    g++ \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir flask numpy pillow scikit-learn
RUN pip install --no-cache-dir --timeout=1000 tensorflow==2.13.0

# Copy the application code and models
COPY app/ ./app/
COPY models/ ./models/

# Set environment variables
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app/app.py"]
