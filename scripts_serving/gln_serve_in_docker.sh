docker run -p 9018:8080 -p 9019:8081 -p 9020:8082 -t openretro:serving-cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=./checkpoints/gln_schneider50k/model-6.dump \
  --models gln_50k_untyped=gln_50k_untyped.mar \
  --ts-config ./config.properties
