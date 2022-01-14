#!/bin/bash

# global
#DATA_NAME="uspto_full"
DATA_NAME="USPTO_50k"
#TRAIN_FILE=/home/ztu/pistachio/data_2021Q1/raw_train.csv
#VAL_FILE=/home/ztu/pistachio/data_2021Q1/raw_val.csv
#TEST_FILE=/home/ztu/pistachio/data_2021Q1/raw_test.csv
#TRAIN_FILE=$PWD/data/uspto_full/raw/raw_train.csv
#VAL_FILE=$PWD/data/uspto_full/raw/raw_val.csv
#TEST_FILE=$PWD/data/uspto_full/raw/raw_test.csv
TRAIN_FILE=$PWD/data/USPTO_50k/raw/raw_train.csv
VAL_FILE=$PWD/data/USPTO_50k/raw/raw_val.csv
TEST_FILE=$PWD/data/USPTO_50k/raw/raw_test.csv
#TEST_FILE=$PWD/data/schneider50k/raw/raw_test.csv
NUM_CORES=32

# paths for gln
PROCESSED_DATA_PATH_GLN=$PWD/data/USPTO_50k/processed_gln
MODEL_PATH_GLN=$PWD/checkpoints/USPTO_50k_gln
TEST_OUTPUT_PATH_GLN=$PWD/results/USPTO_50k_gln

# paths for retroxpert
#PROCESSED_DATA_PATH_RETROXPERT=$PWD/data/pistachio/processed_retroxpert_2021Q1
#MODEL_PATH_RETROXPERT=$PWD/checkpoints/pistachio_retrained_retroxpert
#TEST_OUTPUT_PATH_RETROXPERT=$PWD/results/pistachio_retroxpert
PROCESSED_DATA_PATH_RETROXPERT=$PWD/data/uspto_full/processed_retroxpert_uspto_full
MODEL_PATH_RETROXPERT=$PWD/checkpoints/uspto_full_retrained_retroxpert
TEST_OUTPUT_PATH_RETROXPERT=$PWD/results/uspto_full_retroxpert

# paths for transformer
#PROCESSED_DATA_PATH_TRANSFORMER=$PWD/data/pistachio/processed_transformer
#MODEL_PATH_TRANSFORMER=$PWD/checkpoints/pistachio_transformer
#TEST_OUTPUT_PATH_TRANSFORMER=$PWD/results/pistachio_transformer
PROCESSED_DATA_PATH_TRANSFORMER=$PWD/data/USPTO_50k/processed_transformer
MODEL_PATH_TRANSFORMER=$PWD/checkpoints/USPTO_50k_transformer
TEST_OUTPUT_PATH_TRANSFORMER=$PWD/results/USPTO_50k_transformer

# paths for neuralsym
#PROCESSED_DATA_PATH_NEURALSYM=$PWD/data/pistachio/processed_neuralsym_2021Q1
#MODEL_PATH_NEURALSYM=$PWD/checkpoints/pistachio_retrained_neuralsym
#TEST_OUTPUT_PATH_NEURALSYM=$PWD/results/pistachio_neuralsym
PROCESSED_DATA_PATH_NEURALSYM=$PWD/data/USPTO_50k/processed_neuralsym
MODEL_PATH_NEURALSYM=$PWD/checkpoints/USPTO_50k_neuralsym
TEST_OUTPUT_PATH_NEURALSYM=$PWD/results/USPTO_50k_neuralsym

export DATA_NAME TRAIN_FILE VAL_FILE TEST_FILE NUM_CORES
export PROCESSED_DATA_PATH_GLN MODEL_PATH_GLN TEST_OUTPUT_PATH_GLN
export PROCESSED_DATA_PATH_RETROXPERT MODEL_PATH_RETROXPERT TEST_OUTPUT_PATH_RETROXPERT
export PROCESSED_DATA_PATH_TRANSFORMER MODEL_PATH_TRANSFORMER TEST_OUTPUT_PATH_TRANSFORMER
export PROCESSED_DATA_PATH_NEURALSYM MODEL_PATH_NEURALSYM TEST_OUTPUT_PATH_NEURALSYM
