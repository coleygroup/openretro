docker run -t openretro:serving-cpu \
  --models \
  gln_50k_untyped=gln_50k_untyped.mar \
  neuralsym_50k=neuralsym_50k.mar \
  retroxpert_uspto50k_untyped=retroxpert_uspto50k_untyped.mar \
  transformer_50k_untyped=transformer_50k_untyped.mar
