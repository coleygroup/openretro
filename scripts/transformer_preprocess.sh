conda activate openretro

python3 preprocess.py \
  --model_name="transformer" \
  --data_name="transformer-karpov" \
  --log_file="transformer_preprocess" \
  --train_file="./data/transformer-karpov/raw/retrosynthesis-train.smi" \
  --val_file="./data/transformer-karpov/raw/retrosynthesis-valid.smi" \
  --test_file="./data/transformer-karpov/raw/retrosynthesis-test.smi" \
  --processed_data_path="./data/transformer-karpov/processed/" \
  -src_seq_length 1000 \
  -tgt_seq_length 1000 \
  --num_cores=24 \
  --save_data="do_not_change_this" \
  --train_src="do_not_change_this" \
  --train_tgt="do_not_change_this" \
  --valid_src="do_not_change_this" \
  --valid_tgt="do_not_change_this"
