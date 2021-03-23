python preprocess.py \
  --model_name="retroxpert" \
  --stage=2 \
  --data_name="retroxpert_uspto50k" \
  --log_file="retroxpert_uspto50k_preprocess_s2" \
  --processed_data_path="./data/retroxpert_uspto50k/processed/" \
  --model_path_s1="./checkpoints/retroxpert_uspto50k_untyped" \
  --load_checkpoint_s1 \
  --num_cores=20
