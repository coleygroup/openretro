python3 test.py \
  --model_name="transformer" \
  --data_name="transformer-karpov" \
  --log_file="transformer_test" \
  --processed_data_path="./data/transformer-karpov/processed/" \
  --model_path="./checkpoints/transformer-karpov/model_step_1000000.pt" \
  --test_output_path="./results/transformer-karpov" \
  -batch_size 64 \
  -replace_unk \
  -max_length 200 \
  -beam_size 50 \
  -n_best 50 \
  -gpu 0 \
  -model "do_not_change_this" \
  --src="do_not_change_this"

# beamsize 100 = CUDA OOM, but can use more than 1 GPU?
# n_best 100 = CUDA OOM, but can use more than 1 GPU?

# python3 test.py \
#   --model_name="transformer" \
#   --data_name="transformer-karpov" \
#   --log_file="transformer_test" \
#   --processed_data_path="./data/transformer-karpov/processed/" \
#   --model_path="./checkpoints/transformer-karpov/model_step_1000000.pt" \
#   --test_output_path="./results/transformer-karpov" \
#   -batch_size 64 \
#   -replace_unk \
#   -max_length 200 \
#   -beam_size 10 \
#   -n_best 20 \
#   -gpu 0 \
#   -model "do_not_change_this" \
#   --src="do_not_change_this"