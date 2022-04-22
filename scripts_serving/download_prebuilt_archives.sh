#!/bin/bash

docker run \
  -v "$PWD/mars":/app/openretro/mars \
  -v "$PWD/utils/download_prebuilt_archives.py":/app/openretro/download_prebuilt_archives.py \
  -t openretro:serving-cpu \
  python download_prebuilt_archives.py
