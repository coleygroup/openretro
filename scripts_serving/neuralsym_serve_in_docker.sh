docker run -p 9318:8080 -p 9319:8081 -p 9320:8082 -t openretro:serving-cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=./checkpoints/neuralsym_50k \
  --models neuralsym_50k=neuralsym_50k.mar \
  --ts-config ./config.properties
