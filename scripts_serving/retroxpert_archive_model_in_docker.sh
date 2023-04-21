#!/bin/bash

export EXTRA_FILES="\
/app/openretro/utils,\
/app/openretro/models,\
/app/openretro/checkpoints/USPTO_50k_retroxpert/USPTO_50k_untyped_checkpoint.pt,\
/app/openretro/data/USPTO_50k/processed_retroxpert/product_patterns.txt,\
/app/openretro/checkpoints/USPTO_50k_retroxpert/model_step_300000.pt\
"

docker run \
  -v "$PWD/checkpoints/USPTO_50k_retroxpert/USPTO_50k_untyped_checkpoint.pt":/app/openretro/checkpoints/USPTO_50k_retroxpert/USPTO_50k_untyped_checkpoint.pt \
  -v "$PWD/data/USPTO_50k/processed_retroxpert/product_patterns.txt":/app/openretro/data/USPTO_50k/processed_retroxpert/product_patterns.txt \
  -v "$PWD/checkpoints/USPTO_50k_retroxpert/model_step_300000.pt":/app/openretro/checkpoints/USPTO_50k_retroxpert/model_step_300000.pt \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torch-model-archiver \
  --model-name=USPTO_50k_retroxpert \
  --version=1.0 \
  --handler=/app/openretro/models/retroxpert_model/retroxpert_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/openretro/mars \
  --force

export EXTRA_FILES="\
/app/openretro/utils,\
/app/openretro/models,\
/app/openretro/checkpoints/USPTO_full_retroxpert/USPTO_full_untyped_checkpoint.pt,\
/app/openretro/data/USPTO_full/processed_retroxpert/product_patterns.txt,\
/app/openretro/checkpoints/USPTO_full_retroxpert/model_step_300000.pt\
"

docker run \
  -v "$PWD/checkpoints/USPTO_full_retroxpert/USPTO_full_untyped_checkpoint.pt":/app/openretro/checkpoints/USPTO_full_retroxpert/USPTO_full_untyped_checkpoint.pt \
  -v "$PWD/data/USPTO_full/processed_retroxpert/product_patterns.txt":/app/openretro/data/USPTO_full/processed_retroxpert/product_patterns.txt \
  -v "$PWD/checkpoints/USPTO_full_retroxpert/model_step_300000.pt":/app/openretro/checkpoints/USPTO_full_retroxpert/model_step_300000.pt \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torch-model-archiver \
  --model-name=USPTO_full_retroxpert \
  --version=1.0 \
  --handler=/app/openretro/models/retroxpert_model/retroxpert_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/openretro/mars \
  --force
