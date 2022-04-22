#!/bin/bash

export EXTRA_FILES="\
/app/openretro/utils,\
/app/openretro/checkpoints/USPTO_50k_transformer/model_step_250000.pt\
"

docker run \
  -v "$PWD/checkpoints/USPTO_50k_transformer/model_step_250000.pt":/app/openretro/checkpoints/USPTO_50k_transformer/model_step_250000.pt \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:gpu \
  torch-model-archiver \
  --model-name=USPTO_50k_transformer \
  --version=1.0 \
  --handler=/app/openretro/models/transformer_model/transformer_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/openretro/mars \
  --force

export EXTRA_FILES="\
/app/openretro/utils,\
/app/openretro/checkpoints/USPTO_full_transformer/model_step_250000.pt\
"

docker run \
  -v "$PWD/checkpoints/USPTO_full_transformer/model_step_250000.pt":/app/openretro/checkpoints/USPTO_full_transformer/model_step_250000.pt \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:gpu \
  torch-model-archiver \
  --model-name=USPTO_full_transformer \
  --version=1.0 \
  --handler=/app/openretro/models/transformer_model/transformer_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/openretro/mars \
  --force

export EXTRA_FILES="\
/app/openretro/utils,\
/app/openretro/checkpoints/pistachio_21Q1_transformer/model_step_250000.pt\
"

docker run \
  -v "$PWD/checkpoints/pistachio_21Q1_transformer/model_step_250000.pt":/app/openretro/checkpoints/pistachio_21Q1_transformer/model_step_250000.pt \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:gpu \
  torch-model-archiver \
  --model-name=pistachio_21Q1_transformer \
  --version=1.0 \
  --handler=/app/openretro/models/transformer_model/transformer_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/openretro/mars \
  --force
