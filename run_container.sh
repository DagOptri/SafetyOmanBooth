docker run --gpus all -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  demo-booth:v1.1 /bin/bash
