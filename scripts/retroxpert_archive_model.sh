torch-model-archiver \
  --model-name=retroxpert_uspto50k_untyped \
  --version=1.0 \
  --handler=./models/retroxpert_model/retroxpert_handler.py \
  --extra-files=./models,./checkpoints/retroxpert_uspto50k_untyped/retroxpert_uspto50k_untyped_checkpoint.pt,./data/retroxpert_uspto50k/processed/product_patterns.txt,./checkpoints/retroxpert_uspto50k_untyped/model_step_300000.pt \
  --export-path=./checkpoints/retroxpert_uspto50k_untyped/ \
  --force
