# OpenRetro
An open source library for benchmarking one-step retrosynthesis.
This README is mainly for one-click benchmarking on existing/new reaction datasets.
For serving retrosynthesis models, please refer to README_serving.md.

# Environment Setup
## Option 1. Using Docker (recommended)
Building the Docker for benchmarking requires GPU support,
which will speed up training with any models in general.
First follow the instruction on https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
to install the NVIDIA Container Toolkit (a.k.a. nvidia-docker2). Then run
```    
docker build -f Dockerfile_gpu -t openretro:gpu .
```

## Option 2. Using Conda (not recommended; WIP)
Please contact Zhengkai (ztu@mit.edu) if you have an urgent need for non-Docker environments.
We recommend using Docker in general because some models have platform-specific dependencies
(e.g. requiring reasonably new G++ for compilation) which we can help you handle inside Docker.

# Benchmarking in Docker
Note: please rebuild the Docker before running if there is any change in code.

## Step 1/3
Prepare the raw atom-mapped .csv files for train, validation and test.
See https://www.dropbox.com/sh/6ideflxcakrak10/AADN-TNZnuGjvwZYiLk7zvwra/schneider50k?dl=0&subfolder_nav_tracking=1
for sample data format.
Atom mapping is *required* for GLN, RetroXpert and NeuralSym;
behaviour of these models without atom mapping is undefined.

It is possible to run benchmarking with non atom-mapped reactions, with *template-free models only*.
Currently, the only template-free model supported is Transformer.

## Step 2/3
Configure variables in ./scripts/config_metadata.sh, especially the paths to point to the <b>absolute</b> path of 
raw files and desired output paths. Only the variables related to benchmarked models need to be configured. 
Once the changes have been made, execute the configuration script with
```
source scripts/config_metadata.sh
```

## Step 3/3
Run benchmarking on a machine with GPU using
```
bash scripts/benchmark_in_docker.sh MODEL_NAME1 MODEL_NAME2 ...
```
Currently we support 4 models as MODEL_NAME, namely,
* <b>gln</b>, adapted from original GLN (https://github.com/Hanjun-Dai/GLN)
* <b>retroxpert</b>, adapted from original RetroXpert (https://github.com/uta-smile/RetroXpert)
* <b>neuralsym</b>, adapted from Min Htoo's re-implementation (https://github.com/linminhtoo/neuralsym)
* <b>transformer</b>, adapted from Augmented Transformer (https://github.com/bigchem/synthesis)
  and Molecular Transformer (https://github.com/pschwllr/MolecularTransformer).
  We (trivially) re-implemented the Transformer using models from OpenNMT, which gave cleaner and more modularized codes. 
  
For example, to benchmark with all for models, run
```
bash scripts/benchmark_in_docker.sh gln retroxpert neuralsym transformer
```

To benchmark with only Transformer (e.g. in case there is no atom mapping information), run
```
bash scripts/benchmark_in_docker.sh transformer
```

By default, the benchmarking engine will canonicalize the reactions and subsequently re-number the atoms.
This is done to avoid the information leak that RetroXpert (https://github.com/uta-smile/RetroXpert/blob/canonical_product/readme.md)
and GraphRetro (https://arxiv.org/pdf/2006.07038.pdf) have suffered from.
Without re-calibrating the atom mapping, the test accuracies would be higher than expected
since the original numbering might hint at where the reaction center is.
Nevertheless, in some cases it would be necessary to turn off canonicalization,
e.g. when benchmarking Transformer with augmented non-canonical data.
To benchmark without canonicalization (for Transformer only), run
```
bash scripts/benchmark_in_docker.sh no_canonicalization transformer
```

# Development (TODO)
