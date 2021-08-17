#!/bin/bash

# global
DATA_NAME="pistachio"
TRAIN_FILE=/home/ztu/pistachio/data_2021Q1/raw_train.csv
VAL_FILE=/home/ztu/pistachio/data_2021Q1/raw_val.csv
TEST_FILE=/home/ztu/pistachio/data_2021Q1/raw_test.csv
#TEST_FILE=$PWD/data/schneider50k/raw/raw_test.csv
NUM_CORES=40

# paths for gln
PROCESSED_DATA_PATH_GLN=$PWD/data/schneider50k/processed_gln
MODEL_PATH_GLN=$PWD/checkpoints/schneider50k_retrained_gln
TEST_OUTPUT_PATH_GLN=$PWD/results/schneider50k_gln

# paths for retroxpert
PROCESSED_DATA_PATH_RETROXPERT=$PWD/data/schneider50k/processed_retroxpert
MODEL_PATH_RETROXPERT=$PWD/checkpoints/schneider50k_retrained_retroxpert
TEST_OUTPUT_PATH_RETROXPERT=$PWD/results/schneider50k_retroxpert

# paths for transformer
PROCESSED_DATA_PATH_TRANSFORMER=$PWD/data/pistachio/processed_transformer
MODEL_PATH_TRANSFORMER=$PWD/checkpoints/pistachio_transformer
TEST_OUTPUT_PATH_TRANSFORMER=$PWD/results/pistachio_transformer

# paths for neuralsym
PROCESSED_DATA_PATH_NEURALSYM=$PWD/data/pistachio/processed_neuralsym_2021Q1
MODEL_PATH_NEURALSYM=$PWD/checkpoints/pistachio_retrained_neuralsym
TEST_OUTPUT_PATH_NEURALSYM=$PWD/results/pistachio_neuralsym

export DATA_NAME TRAIN_FILE VAL_FILE TEST_FILE NUM_CORES
export PROCESSED_DATA_PATH_GLN MODEL_PATH_GLN TEST_OUTPUT_PATH_GLN
export PROCESSED_DATA_PATH_RETROXPERT MODEL_PATH_RETROXPERT TEST_OUTPUT_PATH_RETROXPERT
export PROCESSED_DATA_PATH_TRANSFORMER MODEL_PATH_TRANSFORMER TEST_OUTPUT_PATH_TRANSFORMER
export PROCESSED_DATA_PATH_NEURALSYM MODEL_PATH_NEURALSYM TEST_OUTPUT_PATH_NEURALSYM
