#!/bin/bash

docker run -p 9947:8080 -p 9948:8081 -p 9949:8082 \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=/app/openretro/mars \
  --models \
  USPTO_50k_retroxpert=USPTO_50k_retroxpert.mar \
  USPTO_full_retroxpert=USPTO_full_retroxpert.mar \
  --ts-config ./config.properties
