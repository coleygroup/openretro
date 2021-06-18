torch-model-archiver \
  --model-name=transformer_50k_untyped \
  --version=1.0 \
  --handler=./models/transformer_model/transformer_handler.py \
  --extra-files=./utils,./checkpoints/transformer_50k_untyped/model_step_125000.pt \
  --export-path=./checkpoints/transformer_50k_untyped/ \
  --force
