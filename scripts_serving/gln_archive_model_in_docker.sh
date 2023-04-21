#!/bin/bash

export EXTRA_FILES="\
/app/openretro/checkpoints/USPTO_50k_gln/model-6.dump/args.pkl,\
/app/openretro/checkpoints/USPTO_50k_gln/model-6.dump/model.dump,\
/app/openretro/data/USPTO_50k/processed_gln\
"

docker run \
  -v "$PWD/checkpoints/USPTO_50k_gln/model-6.dump":/app/openretro/checkpoints/USPTO_50k_gln/model-6.dump \
  -v "$PWD/data/USPTO_50k/processed_gln":/app/openretro/data/USPTO_50k/processed_gln \
  -v "$PWD/mars":/app/openretro/mars \
  -t openretro:serving-cpu \
  torch-model-archiver \
  --model-name=USPTO_50k_gln \
  --version=1.0 \
  --handler=/app/openretro/models/gln_model/gln_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/openretro/mars \
  --force
