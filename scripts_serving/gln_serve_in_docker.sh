#!/bin/bash

docker run -p 9917:8080 -p 9918:8081 -p 9919:8082 \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=/app/openretro/mars \
  --models USPTO_50k_gln=USPTO_50k_gln.mar \
  --ts-config ./config.properties
