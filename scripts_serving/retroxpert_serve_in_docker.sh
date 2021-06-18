docker run -p 9118:8080 -p 9119:8081 -p 9120:8082 -t openretro:cpu \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=./checkpoints/retroxpert_uspto50k_untyped \
  --models retroxpert_uspto50k_untyped=retroxpert_uspto50k_untyped.mar \
  --ts-config ./config.properties
