#!/bin/bash

# Loading the required module
module load anaconda/2020b
source activate openretro

python preprocess.py \
  --model_name="gln" \
  --data_name="uspto_full" \
  --log_file="gln_preprocess_uspto_full" \
  --train_file="./data/gln_uspto_full/raw/raw_train.csv" \
  --val_file="./data/gln_uspto_full/raw/raw_val.csv" \
  --test_file="./data/gln_uspto_full/raw/raw_test.csv" \
  --processed_data_path="./data/gln_uspto_full/processed" \
  --num_cores=20
