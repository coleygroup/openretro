#!/bin/bash

export EXTRA_FILES="\
/app/openretro/models,\
/app/openretro/utils,\
/app/openretro/checkpoints/USPTO_50k_neuralsym/USPTO_50k.pth.tar,\
/app/openretro/data/USPTO_50k/processed_neuralsym/training_templates.txt,\
/app/openretro/data/USPTO_50k/processed_neuralsym/variance_indices.txt\
"

docker run \
  -v "$PWD/checkpoints/USPTO_50k_neuralsym/USPTO_50k.pth.tar":/app/openretro/checkpoints/USPTO_50k_neuralsym/USPTO_50k.pth.tar \
  -v "$PWD/data/USPTO_50k/processed_neuralsym/training_templates.txt":/app/openretro/data/USPTO_50k/processed_neuralsym/training_templates.txt \
  -v "$PWD/data/USPTO_50k/processed_neuralsym/variance_indices.txt":/app/openretro/data/USPTO_50k/processed_neuralsym/variance_indices.txt \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torch-model-archiver \
  --model-name=USPTO_50k_neuralsym \
  --version=1.0 \
  --handler=/app/openretro/models/neuralsym_model/neuralsym_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/openretro/mars \
  --force

export EXTRA_FILES="\
/app/openretro/models,\
/app/openretro/utils,\
/app/openretro/checkpoints/USPTO_full_neuralsym/USPTO_full.pth.tar,\
/app/openretro/data/USPTO_full/processed_neuralsym/training_templates.txt,\
/app/openretro/data/USPTO_full/processed_neuralsym/variance_indices.txt\
"

docker run \
  -v "$PWD/checkpoints/USPTO_full_neuralsym/USPTO_full.pth.tar":/app/openretro/checkpoints/USPTO_full_neuralsym/USPTO_full.pth.tar \
  -v "$PWD/data/USPTO_full/processed_neuralsym/training_templates.txt":/app/openretro/data/USPTO_full/processed_neuralsym/training_templates.txt \
  -v "$PWD/data/USPTO_full/processed_neuralsym/variance_indices.txt":/app/openretro/data/USPTO_full/processed_neuralsym/variance_indices.txt \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torch-model-archiver \
  --model-name=USPTO_full_neuralsym \
  --version=1.0 \
  --handler=/app/openretro/models/neuralsym_model/neuralsym_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/openretro/mars \
  --force

export EXTRA_FILES="\
/app/openretro/models,\
/app/openretro/utils,\
/app/openretro/checkpoints/pistachio_21Q1_neuralsym/pistachio_21Q1.pth.tar,\
/app/openretro/data/pistachio_21Q1/processed_neuralsym/training_templates.txt,\
/app/openretro/data/pistachio_21Q1/processed_neuralsym/variance_indices.txt\
"

docker run \
  -v "$PWD/checkpoints/pistachio_21Q1_neuralsym/pistachio_21Q1.pth.tar":/app/openretro/checkpoints/pistachio_21Q1_neuralsym/pistachio_21Q1.pth.tar \
  -v "$PWD/data/pistachio_21Q1/processed_neuralsym/training_templates.txt":/app/openretro/data/pistachio_21Q1/processed_neuralsym/training_templates.txt \
  -v "$PWD/data/pistachio_21Q1/processed_neuralsym/variance_indices.txt":/app/openretro/data/pistachio_21Q1/processed_neuralsym/variance_indices.txt \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torch-model-archiver \
  --model-name=pistachio_21Q1_neuralsym \
  --version=1.0 \
  --handler=/app/openretro/models/neuralsym_model/neuralsym_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/openretro/mars \
  --force
