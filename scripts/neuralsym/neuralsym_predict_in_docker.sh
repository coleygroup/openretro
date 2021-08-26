#!/bin/bash

docker run --gpus 1 \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$TEST_FILE":/app/openretro/data/tmp_for_docker/raw_test.csv \
  -v "$PROCESSED_DATA_PATH_NEURALSYM":/app/openretro/data/tmp_for_docker/processed \
  -v "$MODEL_PATH_NEURALSYM":/app/openretro/checkpoints/tmp_for_docker \
  -v "$TEST_OUTPUT_PATH_NEURALSYM":/app/openretro/results/tmp_for_docker \
  -t openretro:gpu \
  python predict.py \
  --model_name="neuralsym" \
  --data_name="$DATA_NAME" \
  --log_file="neuralsym_predict_$DATA_NAME" \
  --test_file=/app/openretro/data/tmp_for_docker/raw_test.csv \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path=/app/openretro/checkpoints/tmp_for_docker \
  --test_output_path=/app/openretro/results/tmp_for_docker \
  --num_cores="$NUM_CORES" \
  --model_arch 'Highway' \
  --bs 500 \
  --depth 0 \
  --hidden_size 300 \
  --topk=100
