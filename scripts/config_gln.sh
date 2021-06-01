DATA_NAME="schneider50k"
TRAIN_FILE=$PWD/data/gln_schneider50k/raw/raw_train.csv
VAL_FILE=$PWD/data/gln_schneider50k/raw/raw_val.csv
TEST_FILE=$PWD/data/gln_schneider50k/raw/raw_test.csv
PROCESSED_DATA_PATH=$PWD/data/gln_schneider50k/processed
MODEL_PATH=$PWD/checkpoints/gln_schneider50k_retrained
TEST_OUTPUT_PATH=$PWD/results/gln_schneider50k
NUM_CORES=20

export DATA_NAME TRAIN_FILE VAL_FILE TEST_FILE PROCESSED_DATA_PATH MODEL_PATH TEST_OUTPUT_PATH NUM_CORES
