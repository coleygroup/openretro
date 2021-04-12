docker run -p 9818:8080 -p 9819:8081 -p 9820:8082 -t openretro-serving:dev \
  torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=./checkpoints/retroxpert_uspto50k_untyped \
  --models retroxpert_uspto50k_untyped=retroxpert_uspto50k_untyped.mar \
  --ts-config ./config.properties
