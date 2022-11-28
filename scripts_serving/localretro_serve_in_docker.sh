#!/bin/bash

docker run -p 9957:8080 -p 9958:8081 -p 9959:8082 \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=/app/openretro/mars \
  --models USPTO_50k_localretro=USPTO_50k_localretro.mar \
  --ts-config ./config.properties
