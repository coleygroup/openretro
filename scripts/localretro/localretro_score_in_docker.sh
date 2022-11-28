#!/bin/bash

docker run \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$TEST_FILE":/app/openretro/data/tmp_for_docker/raw_test.csv \
  -v "$TEST_OUTPUT_PATH_LOCALRETRO":/app/openretro/results/tmp_for_docker \
  -t openretro:gpu \
  python score.py \
  --model_name="localretro" \
  --log_file="localretro_score_$DATA_NAME" \
  --test_file=/app/openretro/data/tmp_for_docker/raw_test.csv \
  --prediction_file=/app/openretro/results/tmp_for_docker/predictions.csv
