#!/bin/bash

docker run --gpus all -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/app:/app \
  demo-booth:v1.1 /bin/bash
