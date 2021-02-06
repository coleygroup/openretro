python preprocess.py \
  --model_name="retroxpert" \
  --stage=1 \
  --data_name="retroxpert_uspto50k" \
  --log_file="retroxpert_uspto50k_preprocess_s1" \
  --train_file="./data/retroxpert_uspto50k/raw/train.csv" \
  --val_file="./data/retroxpert_uspto50k/raw/valid.csv" \
  --test_file="./data/retroxpert_uspto50k/raw/test.csv" \
  --processed_data_path="./data/retroxpert_uspto50k/processed/" \
  --num_cores=20