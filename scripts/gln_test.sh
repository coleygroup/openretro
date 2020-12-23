python train.py \
  --model_name="gln" \
  --data_name="schneider50k" \
  --log_file="gln_test" \
  --train_file="./data/gln_schneider50k/raw/raw_train.csv" \
  --processed_data_path="./data/gln_schneider50k/processed" \
  --model_path="./checkpoints/gln_schneider50k" \
  --test_output_path="./results/gln_schneider50k"
