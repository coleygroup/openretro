#!/bin/bash

docker run --gpus 1 \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$TRAIN_FILE":/app/openretro/data/tmp_for_docker/raw_train.csv \
  -v "$VAL_FILE":/app/openretro/data/tmp_for_docker/raw_val.csv \
  -v "$TEST_FILE":/app/openretro/data/tmp_for_docker/raw_test.csv \
  -v "$PROCESSED_DATA_PATH_GLN":/app/openretro/data/tmp_for_docker/processed \
  -v "$MODEL_PATH_GLN":/app/openretro/checkpoints/tmp_for_docker \
  -v "$TEST_OUTPUT_PATH_GLN":/app/openretro/results/tmp_for_docker \
  -t openretro:gpu \
  python predict.py \
  --test_all_ckpts \
  --model_name="gln" \
  --data_name="$DATA_NAME" \
  --log_file="gln_predict_$DATA_NAME" \
  --train_file=/app/openretro/data/tmp_for_docker/raw_train.csv \
  --val_file=/app/openretro/data/tmp_for_docker/raw_val.csv \
  --test_file=/app/openretro/data/tmp_for_docker/raw_test.csv \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path=/app/openretro/checkpoints/tmp_for_docker \
  --test_output_path=/app/openretro/results/tmp_for_docker
