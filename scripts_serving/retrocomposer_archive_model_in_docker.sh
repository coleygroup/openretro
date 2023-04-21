#!/bin/bash

export EXTRA_FILES="\
models.zip,\
utils.zip,\
/app/openretro/checkpoints/USPTO_50k_retrocomposer/model.pt,\
/app/openretro/data/USPTO_50k/processed_retrocomposer/seq_to_templates.data,\
/app/openretro/data/USPTO_50k/processed_retrocomposer/templates_cano_train.json\
"

zip -r models.zip models/
zip -r utils.zip utils/

docker run \
  -v "$PWD/checkpoints/USPTO_50k_retrocomposer/model.pt":/app/openretro/checkpoints/USPTO_50k_retrocomposer/model.pt \
  -v "$PWD/data/USPTO_50k/processed_retrocomposer/seq_to_templates.data":/app/openretro/data/USPTO_50k/processed_retrocomposer/seq_to_templates.data \
  -v "$PWD/data/USPTO_50k/processed_retrocomposer/templates_cano_train.json":/app/openretro/data/USPTO_50k/processed_retrocomposer/templates_cano_train.json \
  -v "$PWD/mars":/app/openretro/mars \
  -v "$PWD/models.zip":/app/openretro/models.zip \
  -v "$PWD/utils.zip":/app/openretro/utils.zip \
  -t openretro:serving-cpu \
  torch-model-archiver \
  --model-name=USPTO_50k_retrocomposer \
  --version=1.0 \
  --handler=/app/openretro/models/retrocomposer_model/retrocomposer_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/openretro/mars \
  --force
