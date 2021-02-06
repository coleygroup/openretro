python train.py \
  --do_train \
  --model_name="retroxpert" \
  --stage=1 \
  --data_name="retroxpert_uspto50k" \
  --log_file="retroxpert_uspto50k_train_s1" \
  --processed_data_path="./data/retroxpert_uspto50k/processed/" \
  --model_path="./checkpoints/retroxpert_uspto50k_untyped"
