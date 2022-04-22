#!/bin/bash

docker run -p 9937:8080 -p 9938:8081 -p 9939:8082 \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=/app/openretro/mars \
  --models \
  USPTO_50k_transformer=USPTO_50k_transformer.mar \
  USPTO_full_transformer=USPTO_full_transformer.mar \
  pistachio_21Q1_transformer=pistachio_21Q1_transformer.mar \
  --ts-config ./config.properties
