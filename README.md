# Security Live Booth App

- Reference App: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/deepstream-nvdsanalytics/README


# Demo Booth App v1.1

## Quick Start

1. Build image:
```bash
docker build -t demo-booth:v1.1 .
```

2. Run container:
```bash
./run_container.sh
```

3. Inside container:
```bash
cd /app
python3 main.py
```

## Requirements
- Docker
- NVIDIA GPU
- NVIDIA Container Toolkit

