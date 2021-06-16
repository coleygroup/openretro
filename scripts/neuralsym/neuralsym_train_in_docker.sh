#!/bin/bash

docker run --gpus 1 \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$TRAIN_FILE":/app/openretro/data/tmp_for_docker/raw_train.csv \
  -v "$PROCESSED_DATA_PATH_NEURALSYM":/app/openretro/data/tmp_for_docker/processed \
  -v "$MODEL_PATH_NEURALSYM":/app/openretro/checkpoints/tmp_for_docker \
  -t openretro:gpu \
  python train.py \
  --do_train \
  --model_name="neuralsym" \
  --data_name="$DATA_NAME" \
  --log_file="neuralsym_train_$DATA_NAME" \
  --train_file=/app/openretro/data/tmp_for_docker/raw_train.csv \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path=/app/openretro/checkpoints/tmp_for_docker \
  --model 'Highway' \
  --bs 300 \
  --bs_eval 300 \
  --seed 77777777 \
  --learning_rate 1e-3 \
  --epochs 30 \
  --early_stop \
  --early_stop_patience 2 \
  --depth 0 \
  --hidden_size 300 \
  --lr_scheduler_factor 0.3 \
  --lr_scheduler_patience 1
