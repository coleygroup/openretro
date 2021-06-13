#!/bin/bash

docker run --gpus 1 \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$PROCESSED_DATA_PATH_RETROXPERT":/app/openretro/data/tmp_for_docker/processed \
  -v "$MODEL_PATH_RETROXPERT":/app/openretro/checkpoints/tmp_for_docker \
  -t openretro:gpu \
  python train.py \
  --do_train \
  --model_name="retroxpert" \
  --stage=1 \
  --data_name="$DATA_NAME" \
  --log_file="retroxpert_train_s1_$DATA_NAME" \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path=/app/openretro/checkpoints/tmp_for_docker \
  --batch_size=32 \
  --epochs=80
