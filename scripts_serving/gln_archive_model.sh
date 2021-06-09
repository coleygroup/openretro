torch-model-archiver \
  --model-name=gln_50k_untyped \
  --version=1.0 \
  --handler=./models/gln_model/gln_handler.py \
  --extra-files=./checkpoints/gln_schneider50k/model-6.dump/args.pkl,./checkpoints/gln_schneider50k/model-6.dump/model.dump,./data/gln_schneider50k/processed \
  --export-path=./checkpoints/gln_schneider50k/model-6.dump \
  --force
