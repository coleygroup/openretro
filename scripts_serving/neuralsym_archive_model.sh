torch-model-archiver \
  --model-name=neuralsym_50k \
  --version=1.0 \
  --handler=./models/neuralsym_model/neuralsym_handler.py \
  --extra-files=./models,./utils,./checkpoints/neuralsym_50k/schneider50k.pth.tar,./data/schneider50k/processed_neuralsym/training_templates.txt,./data/schneider50k/processed_neuralsym/variance_indices.txt \
  --export-path=./checkpoints/neuralsym_50k/ \
  --force
