#!/bin/bash

# global
DATA_NAME="schneider50k"
TRAIN_FILE=$PWD/data/schneider50k/raw/raw_train.csv
VAL_FILE=$PWD/data/schneider50k/raw/raw_val.csv
TEST_FILE=$PWD/data/schneider50k/raw/raw_test.csv
NUM_CORES=20

# paths for gln
PROCESSED_DATA_PATH_GLN=$PWD/data/schneider50k/processed_gln
MODEL_PATH_GLN=$PWD/checkpoints/schneider50k_retrained_gln
TEST_OUTPUT_PATH_GLN=$PWD/results/schneider50k_gln

# paths for retroxpert
PROCESSED_DATA_PATH_RETROXPERT=$PWD/data/schneider50k/processed_retroxpert
MODEL_PATH_RETROXPERT=$PWD/checkpoints/schneider50k_retrained_retroxpert
TEST_OUTPUT_PATH_RETROXPERT=$PWD/results/schneider50k_retroxpert

export DATA_NAME TRAIN_FILE VAL_FILE TEST_FILE NUM_CORES
export PROCESSED_DATA_PATH_GLN MODEL_PATH_GLN TEST_OUTPUT_PATH_GLN
export PROCESSED_DATA_PATH_RETROXPERT MODEL_PATH_RETROXPERT TEST_OUTPUT_PATH_RETROXPERT
