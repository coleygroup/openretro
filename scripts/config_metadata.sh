#!/bin/bash

# global
DATA_NAME="USPTO_50k"
TRAIN_FILE=$PWD/data/USPTO_50k/raw/raw_train.csv
VAL_FILE=$PWD/data/USPTO_50k/raw/raw_val.csv
TEST_FILE=$PWD/data/USPTO_50k/raw/raw_test.csv
#DATA_NAME="USPTO_full"
#TRAIN_FILE=$PWD/data/USPTO_full/raw/raw_train.csv
#VAL_FILE=$PWD/data/USPTO_full/raw/raw_val.csv
#TEST_FILE=$PWD/data/USPTO_full/raw/raw_test.csv
#DATA_NAME="pistachio_21Q1"
#TRAIN_FILE=/home/ztu/pistachio/data_2021Q1/raw_train.csv
#VAL_FILE=/home/ztu/pistachio/data_2021Q1/raw_val.csv
#TEST_FILE=/home/ztu/pistachio/data_2021Q1/raw_test.csv
NUM_CORES=32

# paths for gln
PROCESSED_DATA_PATH_GLN=$PWD/data/USPTO_50k/processed_gln
MODEL_PATH_GLN=$PWD/checkpoints/USPTO_50k_gln
TEST_OUTPUT_PATH_GLN=$PWD/results/USPTO_50k_gln

# paths for retroxpert
PROCESSED_DATA_PATH_RETROXPERT=$PWD/data/USPTO_50k/processed_retroxpert
MODEL_PATH_RETROXPERT=$PWD/checkpoints/USPTO_50k_retroxpert
TEST_OUTPUT_PATH_RETROXPERT=$PWD/results/USPTO_50k_retroxpert
#PROCESSED_DATA_PATH_RETROXPERT=$PWD/data/USPTO_full/processed_retroxpert
#MODEL_PATH_RETROXPERT=$PWD/checkpoints/USPTO_full_retroxpert
#TEST_OUTPUT_PATH_RETROXPERT=$PWD/results/USPTO_full_retroxpert

# paths for transformer
PROCESSED_DATA_PATH_TRANSFORMER=$PWD/data/USPTO_50k/processed_transformer
MODEL_PATH_TRANSFORMER=$PWD/checkpoints/USPTO_50k_transformer
TEST_OUTPUT_PATH_TRANSFORMER=$PWD/results/USPTO_50k_transformer
#PROCESSED_DATA_PATH_TRANSFORMER=$PWD/data/USPTO_full/processed_transformer
#MODEL_PATH_TRANSFORMER=$PWD/checkpoints/USPTO_full_transformer
#TEST_OUTPUT_PATH_TRANSFORMER=$PWD/results/USPTO_full_transformer
#PROCESSED_DATA_PATH_TRANSFORMER=$PWD/data/pistachio_21Q1/processed_transformer
#MODEL_PATH_TRANSFORMER=$PWD/checkpoints/pistachio_21Q1_transformer
#TEST_OUTPUT_PATH_TRANSFORMER=$PWD/results/pistachio_21Q1_transformer

# paths for neuralsym
PROCESSED_DATA_PATH_NEURALSYM=$PWD/data/USPTO_50k/processed_neuralsym
MODEL_PATH_NEURALSYM=$PWD/checkpoints/USPTO_50k_neuralsym
TEST_OUTPUT_PATH_NEURALSYM=$PWD/results/USPTO_50k_neuralsym
#PROCESSED_DATA_PATH_NEURALSYM=$PWD/data/USPTO_full/processed_neuralsym
#MODEL_PATH_NEURALSYM=$PWD/checkpoints/USPTO_full_neuralsym
#TEST_OUTPUT_PATH_NEURALSYM=$PWD/results/USPTO_full_neuralsym
#PROCESSED_DATA_PATH_NEURALSYM=$PWD/data/pistachio_21Q1/processed_neuralsym
#MODEL_PATH_NEURALSYM=$PWD/checkpoints/pistachio_21Q1_neuralsym
#TEST_OUTPUT_PATH_NEURALSYM=$PWD/results/pistachio_21Q1_neuralsym

export DATA_NAME TRAIN_FILE VAL_FILE TEST_FILE NUM_CORES
export PROCESSED_DATA_PATH_GLN MODEL_PATH_GLN TEST_OUTPUT_PATH_GLN
export PROCESSED_DATA_PATH_RETROXPERT MODEL_PATH_RETROXPERT TEST_OUTPUT_PATH_RETROXPERT
export PROCESSED_DATA_PATH_TRANSFORMER MODEL_PATH_TRANSFORMER TEST_OUTPUT_PATH_TRANSFORMER
export PROCESSED_DATA_PATH_NEURALSYM MODEL_PATH_NEURALSYM TEST_OUTPUT_PATH_NEURALSYM
