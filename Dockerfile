# Use CUDA 12.1 base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    wget \
    ninja-build \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install custom wheels
COPY wheels/ ./wheels/
RUN if [ -d "wheels" ]; then \
    pip3 install wheels/*.whl; \
    fi

# Copy application files
COPY app.py ./
COPY trellis/ ./trellis/

# Create necessary directories
RUN mkdir -p /tmp/Trellis-demo

# Set permissions
RUN chmod -R 755 /app
RUN chmod -R 777 /tmp/Trellis-demo

# Expose the FastAPI port
EXPOSE 8000

# Create a non-root user
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app /tmp/Trellis-demo

# Switch to non-root user
USER appuser

# Command to run the FastAPI app with multiple workers
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]