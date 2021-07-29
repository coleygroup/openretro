docker run -p 9018:8080 -p 9019:8081 -p 9020:8082 \
  -t openretro:serving-cpu \
  --models \
  gln_50k_untyped=gln_50k_untyped.mar \
  neuralsym_50k=neuralsym_50k.mar \
  retroxpert_uspto50k_untyped=retroxpert_uspto50k_untyped.mar \
  transformer_50k_untyped=transformer_50k_untyped.mar
