#!/bin/bash

docker run -p 9967:8080 -p 9968:8081 -p 9969:8082 \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=/app/openretro/mars \
  --models USPTO_50k_retrocomposer=USPTO_50k_retrocomposer.mar \
  --ts-config ./config.properties
