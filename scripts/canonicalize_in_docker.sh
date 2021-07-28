#!/bin/bash

docker run \
  -v "$PWD/logs":/app/openretro/logs \
  -v "$PWD/checkpoints":/app/openretro/checkpoints \
  -v "$PWD/results":/app/openretro/results \
  -v "$TRAIN_FILE":/app/openretro/data/tmp_for_docker/raw_train.csv \
  -v "$VAL_FILE":/app/openretro/data/tmp_for_docker/raw_val.csv \
  -v "$TEST_FILE":/app/openretro/data/tmp_for_docker/raw_test.csv \
  -v "$PROCESSED_DATA_PATH_TRANSFORMER":/app/openretro/data/tmp_for_docker/processed \
  -t openretro:gpu \
  python preprocess.py \
  --model_name="transformer" \
  --data_name="$DATA_NAME" \
  --log_file="transformer_preprocess_$DATA_NAME" \
  --train_file=/app/openretro/data/tmp_for_docker/raw_train.csv \
  --val_file=/app/openretro/data/tmp_for_docker/raw_val.csv \
  --test_file=/app/openretro/data/tmp_for_docker/raw_test.csv \
  --processed_data_path=/app/openretro/data/tmp_for_docker/processed \
  -src_seq_length 1000 \
  -tgt_seq_length 1000 \
  --num_cores="$NUM_CORES" \
  --save_data="do_not_change_this" \
  --train_src="do_not_change_this" \
  --train_tgt="do_not_change_this" \
  --valid_src="do_not_change_this" \
  --valid_tgt="do_not_change_this"
