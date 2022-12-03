#!/bin/bash

docker run --gpus 1 \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$PROCESSED_DATA_PATH_RETROCOMPOSER":/app/openretro/data/tmp_for_docker/processed \
  -v "$MODEL_PATH_RETROCOMPOSER":/app/openretro/checkpoints/tmp_for_docker \
  -v "$TEST_OUTPUT_PATH_RETROCOMPOSER":/app/openretro/results/tmp_for_docker \
  -t openretro:gpu \
  python predict.py \
  --model_name="retrocomposer" \
  --data_name="$DATA_NAME" \
  --log_file="retrocomposer_predict_$DATA_NAME" \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path=/app/openretro/checkpoints/tmp_for_docker \
  --test_output_path=/app/openretro/results/tmp_for_docker
