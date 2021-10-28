#!/bin/bash

docker run --gpus 1 \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$PROCESSED_DATA_PATH_RETROXPERT":/app/openretro/data/tmp_for_docker/processed \
  -v "$MODEL_PATH_RETROXPERT":/app/openretro/checkpoints/tmp_for_docker \
  -v "$TEST_OUTPUT_PATH_RETROXPERT":/app/openretro/results/tmp_for_docker \
  -t openretro:gpu \
  python predict.py \
  --model_name="retroxpert" \
  --data_name="$DATA_NAME" \
  --log_file="retroxpert_predict_$DATA_NAME" \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path=/app/openretro/checkpoints/tmp_for_docker \
  --test_output_path=/app/openretro/results/tmp_for_docker \
  --shard_size 1000 \
  --beam_size 20 \
  --n_best 20 \
  --batch_size 8 \
  --replace_unk \
  --max_length 300 \
  -gpu 0 \
  -model "do_not_change_this" \
  --src="do_not_change_this"
