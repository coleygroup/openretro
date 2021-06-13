#!/bin/bash

docker run --gpus 1 \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$PROCESSED_DATA_PATH_RETROXPERT":/app/openretro/data/tmp_for_docker/processed \
  -v "$MODEL_PATH_RETROXPERT":/app/openretro/checkpoints/tmp_for_docker \
  -t openretro:gpu \
  python preprocess.py \
  --model_name="retroxpert" \
  --stage=2 \
  --data_name="$DATA_NAME" \
  --log_file="retroxpert_preprocess_s2_$DATA_NAME" \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  --model_path_s1=/app/openretro/checkpoints/tmp_for_docker \
  --load_checkpoint_s1 \
  --num_cores="$NUM_CORES" \
  --save_data="do_not_change_this" \
  --train_src="do_not_change_this" \
  --train_tgt="do_not_change_this" \
  --valid_src="do_not_change_this" \
  --valid_tgt="do_not_change_this"
