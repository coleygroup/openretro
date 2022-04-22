#!/bin/bash

docker run -p 9927:8080 -p 9928:8081 -p 9929:8082 \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=/app/openretro/mars \
  --models \
  USPTO_50k_neuralsym=USPTO_50k_neuralsym.mar \
  USPTO_full_neuralsym=USPTO_full_neuralsym.mar \
  pistachio_21Q1_neuralsym=pistachio_21Q1_neuralsym.mar \
  --ts-config ./config.properties
