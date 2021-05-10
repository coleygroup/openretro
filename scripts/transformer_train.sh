conda activate openretro
python3 train.py \
  --do_train \
  --data="do_not_change_this" \
  --model_name="transformer" \
  --data_name="transformer-karpov" \
  --log_file="transformer_train" \
  --processed_data_path="./data/transformer-karpov/processed/" \
  --model_path="./checkpoints/transformer-karpov" \
  -seed 42 \
  -gpu_ranks 0 \
  -save_checkpoint_steps 20000 \
  -keep_checkpoint 10 \
  -train_steps 1000000 \
  -param_init 0 \
  -param_init_glorot \
  -max_generator_batches 32 \
  -batch_size 3000 \
  -batch_type tokens \
  -normalization tokens \
  -max_grad_norm 0 \
  -optim adam \
  -adam_beta1 0.9 \
  -adam_beta2 0.998 \
  -decay_method noam \
  -warmup_steps 16000 \
  -learning_rate 2 \
  -label_smoothing 0.0 \
  -report_every 1000 \
  -layers 3 \
  -rnn_size 64 \
  -word_vec_size 64 \
  -encoder_type transformer \
  -decoder_type transformer \
  -dropout 0.1 \
  -position_encoding \
  -share_embeddings \
  -global_attention general \
  -global_attention_function softmax \
  -self_attn_type scaled-dot \
  --heads 8 \
  -transformer_ff 512

# batch_size 30000
# train_steps 500000
# added world_size
# changed gpu_ranks from 0 to 1 (didn't work, prolly need to configure the torch.distributed properly)
# -world_size 2