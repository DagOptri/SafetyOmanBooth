# Use DeepStream 7.0 base image
FROM nvcr.io/nvidia/deepstream:7.0-gc-triton-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-gi \
    python3-dev \
    python3-gst-1.0 \
    python3-opencv \
    python3-numpy \
    libgstrtspserver-1.0-0 \
    gstreamer1.0-rtsp \
    libgirepository1.0-dev \
    gobject-introspection \
    gir1.2-gst-rtsp-server-1.0 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy application files
COPY app/* ./
COPY app/utils ./utils/
COPY data ./data/

# Set up Python environment
RUN pip3 install --upgrade pip
RUN pip3 install numpy pyyaml

# Environment variables for DeepStream
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream-7.0/lib/gst-plugins/
ENV GST_DEBUG=3

# Make the export script executable
RUN chmod +x export_int8.sh

# Set the working directory
WORKDIR /app

# Set the default command to run the Python script
CMD ["python3", "main.py"]

# Labels
LABEL maintainer="ddagaev222"
LABEL version="v1.1"
LABEL description="DeepStream Demo Booth Application"
