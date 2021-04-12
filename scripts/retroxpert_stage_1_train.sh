python train.py \
  --do_train \
  --model_name="retroxpert" \
  --stage=1 \
  --data_name="retroxpert_uspto_full" \
  --log_file="retroxpert_uspto_full_train_s1" \
  --processed_data_path="./data/retroxpert_uspto_full/processed/" \
  --model_path="./checkpoints/retroxpert_uspto_full_untyped"
