python train.py \
  --do_train \
  --model_name="gln" \
  --data_name="schneider50k" \
  --log_file="gln_train" \
  --processed_data_path="./data/gln_schneider50k/processed" \
  --model_path="./checkpoints/gln_schneider50k"
