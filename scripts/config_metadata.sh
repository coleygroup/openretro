# global
DATA_NAME="schneider50k"
TRAIN_FILE=$PWD/data/gln_schneider50k/raw/raw_train.csv
VAL_FILE=$PWD/data/gln_schneider50k/raw/raw_val.csv
TEST_FILE=$PWD/data/gln_schneider50k/raw/raw_test.csv
NUM_CORES=20
# paths for gln
PROCESSED_DATA_PATH_GLN=$PWD/data/gln_schneider50k/processed
MODEL_PATH_GLN=$PWD/checkpoints/gln_schneider50k_retrained
TEST_OUTPUT_PATH_GLN=$PWD/results/gln_schneider50k

export DATA_NAME TRAIN_FILE VAL_FILE TEST_FILE NUM_CORES
export PROCESSED_DATA_PATH_GLN MODEL_PATH_GLN TEST_OUTPUT_PATH_GLN
