#!/bin/bash

docker run --gpus 1 \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$PROCESSED_DATA_PATH_TRANSFORMER":/app/openretro/data/tmp_for_docker/processed \
  -v "$MODEL_PATH_TRANSFORMER":/app/openretro/checkpoints/tmp_for_docker \
  -v "$TEST_OUTPUT_PATH_TRANSFORMER":/app/openretro/results/tmp_for_docker \
  -t openretro:gpu \
  python predict.py \
  --model_name="transformer" \
  --data_name="$DATA_NAME" \
  --log_file="transformer_predict_$DATA_NAME" \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path=/app/openretro/checkpoints/tmp_for_docker \
  --test_output_path=/app/openretro/results/tmp_for_docker \
  -batch_size 64 \
  -replace_unk \
  -max_length 200 \
  -beam_size 5 \
  -n_best 20 \
  -gpu 0 \
  -model "do_not_change_this" \
  --src="do_not_change_this"
