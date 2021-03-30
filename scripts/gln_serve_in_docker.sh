docker run -p 9918:8080 -p 9919:8081 -p 9920:8082 -t openretro-serving:dev-gln \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=./checkpoints/gln_schneider50k/model-6.dump \
  --models gln_50k_untyped=gln_50k_untyped.mar \
  --ts-config ./config.properties