docker run -p 9218:8080 -p 9219:8081 -p 9220:8082 -t openretro:cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=./checkpoints/transformer_50k_untyped \
  --models transformer_50k_untyped=transformer_50k_untyped.mar \
  --ts-config ./config.properties
