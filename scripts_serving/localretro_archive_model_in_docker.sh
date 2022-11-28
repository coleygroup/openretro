#!/bin/bash

export EXTRA_FILES="\
models.zip,\
utils.zip,\
/app/openretro/checkpoints/USPTO_50k_localretro/LocalRetro.pth,\
/app/openretro/data/USPTO_50k/processed_localretro/atom_templates.csv,\
/app/openretro/data/USPTO_50k/processed_localretro/bond_templates.csv,\
/app/openretro/data/USPTO_50k/processed_localretro/template_infos.csv\
"

zip models.zip models/*
zip utils.zip utils/*

docker run \
  -v "$PWD/checkpoints/USPTO_50k_localretro/LocalRetro.pth":/app/openretro/checkpoints/USPTO_50k_localretro/LocalRetro.pth \
  -v "$PWD/data/USPTO_50k/processed_localretro/atom_templates.csv":/app/openretro/data/USPTO_50k/processed_localretro/atom_templates.csv \
  -v "$PWD/data/USPTO_50k/processed_localretro/bond_templates.csv":/app/openretro/data/USPTO_50k/processed_localretro/bond_templates.csv \
  -v "$PWD/data/USPTO_50k/processed_localretro/template_infos.csv":/app/openretro/data/USPTO_50k/processed_localretro/template_infos.csv \
  -v "$PWD/mars":/app/openretro/mars \
  -v "$PWD/models.zip":/app/openretro/models.zip \
  -v "$PWD/utils.zip":/app/openretro/utils.zip \
  -t openretro:gpu \
  /bin/bash -c \
  torch-model-archiver \
  --model-name=USPTO_50k_localretro \
  --version=1.0 \
  --handler=/app/openretro/models/localretro_model/localretro_handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/openretro/mars \
  --force
