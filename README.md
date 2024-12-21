
# TRELLIS Microsoft - Dockerized FastAPI by [UNES97](https://github.com/UNES97)

This repository contains a Dockerized implementation of the Microsoft TRELLIS API using FastAPI.

## Prerequisites

- Docker installed on your system
- NVIDIA GPU with compatible drivers

## System Requirements

- NVIDIA GPU with CUDA support
- Minimum 16GB GPU memory recommended
- Docker version 19.03 or later
- NVIDIA driver updated

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/UNES97/TRELLIS-3D-API
cd TRELLIS-3D-API
```


2. Build the Docker image:
```bash
docker build -t trellis-fastapi-app .
```

3. Run the container:
```bash
docker run -it -p 8000:8000 trellis-fastapi-app
```

The API will be available at http://localhost:8000

