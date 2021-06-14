#!/bin/bash

if [[ $# -eq 0 ]]; then
  echo "No arguments supplied. Please specify models to be benchmarked [gln|retroxpert|neuralsym|transformer]"
fi

CANONICALIZATION_FLAG=""
if [[ "$*" == *"no_canonicalization"* ]]; then
  CANONICALIZATION_FLAG="no_canonicalization"
fi

# gln
if [[ "$*" == *"gln"* ]]; then
  bash scripts/gln/gln_preprocess_in_docker.sh $CANONICALIZATION_FLAG
  bash scripts/gln/gln_train_in_docker.sh
  bash scripts/gln/gln_predict_in_docker.sh
  bash scripts/gln/gln_score_in_docker.sh
fi

# retroxpert
if [[ "$*" == *"retroxpert"* ]]; then
#  bash scripts/retroxpert/retroxpert_stage_1_preprocess_in_docker.sh $CANONICALIZATION_FLAG
#  bash scripts/retroxpert/retroxpert_stage_1_train_in_docker.sh
#  bash scripts/retroxpert/retroxpert_stage_2_preprocess_in_docker.sh $CANONICALIZATION_FLAG
#  bash scripts/retroxpert/retroxpert_stage_2_train_in_docker.sh
  bash scripts/retroxpert/retroxpert_predict_in_docker.sh
  bash scripts/retroxpert/retroxpert_score_in_docker.sh
fi

# neuralsym
if [[ "$*" == *"neuralsym"* ]]; then
  bash scripts/neuralsym/neuralsym_preprocess_in_docker.sh $CANONICALIZATION_FLAG
  bash scripts/neuralsym/neuralsym_train_in_docker.sh
  bash scripts/neuralsym/neuralsym_predict_in_docker.sh
  bash scripts/neuralsym/neuralsym_score_in_docker.sh
fi

# transformer
if [[ "$*" == *"transformer"* ]]; then
  bash scripts/transformer/transformer_preprocess_in_docker.sh $CANONICALIZATION_FLAG
  bash scripts/transformer/transformer_train_in_docker.sh
  bash scripts/transformer/transformer_predict_in_docker.sh
  bash scripts/transformer/transformer_score_in_docker.sh
fi
