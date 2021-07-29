docker run -p 8080:8080 -p 8081:8081 -p 8082:8082 \
  -t openretro:serving-cpu \
  --models \
  gln_50k_untyped=gln_50k_untyped.mar \
  neuralsym_50k=neuralsym_50k.mar \
  retroxpert_uspto50k_untyped=retroxpert_uspto50k_untyped.mar \
  transformer_50k_untyped=transformer_50k_untyped.mar
