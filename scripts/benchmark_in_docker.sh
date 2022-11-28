#!/bin/bash

if [[ $# -eq 0 ]]; then
  echo "No arguments supplied. Please specify models to be benchmarked [gln|retroxpert|neuralsym|transformer]"
fi

# WIP
CANONICALIZATION_FLAG=""
if [[ "$*" == *"no_canonicalization"* ]]; then
  echo "no_canonicalization flag detected, skipping canonicalization"
  CANONICALIZATION_FLAG="no_canonicalization"
else
  echo "running canonicalization = True"
#  bash scripts/canonicalize_in_docker.sh
#  TRAIN_FILE=$TRAIN_FILE.cano.csv
#  VAL_FILE=$VAL_FILE.cano.csv
#  TEST_FILE=$TEST_FILE.cano.csv
#  export TRAIN_FILE VAL_FILE TEST_FILE
fi

# neuralsym
if [[ "$*" == *"neuralsym"* ]]; then
  bash scripts/neuralsym/neuralsym_preprocess_in_docker.sh $CANONICALIZATION_FLAG
  bash scripts/neuralsym/neuralsym_train_in_docker.sh
  bash scripts/neuralsym/neuralsym_predict_in_docker.sh
  bash scripts/neuralsym/neuralsym_score_in_docker.sh
fi

# gln
if [[ "$*" == *"gln"* ]]; then
  bash scripts/gln/gln_preprocess_in_docker.sh $CANONICALIZATION_FLAG
  bash scripts/gln/gln_train_in_docker.sh
  bash scripts/gln/gln_predict_in_docker.sh
  bash scripts/gln/gln_score_in_docker.sh
fi

# transformer
if [[ "$*" == *"transformer"* ]]; then
  bash scripts/transformer/transformer_preprocess_in_docker.sh $CANONICALIZATION_FLAG
  bash scripts/transformer/transformer_train_in_docker.sh
  bash scripts/transformer/transformer_predict_in_docker.sh
  bash scripts/transformer/transformer_score_in_docker.sh
fi

# retroxpert
if [[ "$*" == *"retroxpert"* ]]; then
  bash scripts/retroxpert/retroxpert_stage_1_preprocess_in_docker.sh $CANONICALIZATION_FLAG
  bash scripts/retroxpert/retroxpert_stage_1_train_in_docker.sh
  bash scripts/retroxpert/retroxpert_stage_2_preprocess_in_docker.sh $CANONICALIZATION_FLAG
  bash scripts/retroxpert/retroxpert_stage_2_train_in_docker.sh
  bash scripts/retroxpert/retroxpert_predict_in_docker.sh
  bash scripts/retroxpert/retroxpert_score_in_docker.sh
fi

# gln
if [[ "$*" == *"gln"* ]]; then
  bash scripts/gln/gln_preprocess_in_docker.sh $CANONICALIZATION_FLAG
  bash scripts/gln/gln_train_in_docker.sh
  bash scripts/gln/gln_predict_in_docker.sh
  bash scripts/gln/gln_score_in_docker.sh
fi

# localretro
if [[ "$*" == *"localretro"* ]]; then
  bash scripts/localretro/localretro_preprocess_in_docker.sh $CANONICALIZATION_FLAG
  bash scripts/localretro/localretro_train_in_docker.sh
  bash scripts/localretro/localretro_predict_in_docker.sh
  bash scripts/localretro/localretro_score_in_docker.sh
fi
