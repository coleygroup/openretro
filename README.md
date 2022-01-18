# OpenRetro
An open source library for benchmarking one-step retrosynthesis.
This README is mainly for one-click benchmarking on existing/new reaction datasets.
For serving retrosynthesis models, please refer to README_serving.md.

# Selected Results
We used the RetroXpert version of USPTO datasets
(https://github.com/uta-smile/RetroXpert/tree/main/data).
We mostly followed the recommended or default hyperparameters without tuning,
which we found to be sufficiently robust.
These are probably not universally optimal for any dataset, and we leave the turning to the users.

### USPTO_50k without reaction type
| Accuracy (%) | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Top-20 | Top-50 |
|--------------|-------|-------|-------|-------|--------|--------|--------|
| NeuralSym    | 45.5  | 59.7  | 67.1  | 74.6  | 81.6   | 84.9   | 85.7   |
| GLN          | 51.8  | 62.5  | 68.8  | 75.9  | 83.4   | 89.3   | 92.3   |
| Transformer  | 43.4  | 53.9  | 58.5  | 63.0  | 67.1   | 69.4   | -      |
| RetroXpert   | 45.4  | 55.5  | 59.8  | 64.1  | 68.8   | 72.0   | -      |

### USPTO_full without reaction type
| Accuracy (%) | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Top-20 | Top-50 |
|--------------|-------|-------|-------|-------|--------|--------|--------|
| NeuralSym    | 43.6  | 54.8  | 59.8  | 64.6  | 68.9   | 71.4   | 72.2   |
| Transformer  | 44.5  | 55.6  | 60.3  | 65.1  | 69.6   | 72.1   | -      |
| RetroXpert   | 39.7  | 47.2  | 49.9  | 52.3  | 54.9   | 56.9   | -      |

# Environment Setup
Building the Docker for benchmarking requires GPU support,
which will speed up training with any models in general.
First follow the instruction on https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
to install the NVIDIA Container Toolkit (a.k.a. nvidia-docker2). Then run
```    
docker build -f Dockerfile_gpu -t openretro:gpu .
```

# Benchmarking in Docker
Note: please rebuild the Docker before running if there is any change in code.

## Step 1/3
Prepare the raw atom-mapped .csv files for train, validation and test.
The required columns are "class", "id" and "rxn_smiles".
See data/USPTO_50k/{train,val,test}.csv for sample data format.
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

This will run the preprocessing, training, predicting and scoring with specified models,
with Top-n accuracies up to n=50 as the final outputs.
Progress and result logs will be saved under ./logs 

Currently, we support 4 models as MODEL_NAME, namely,
* <b>gln</b>, adapted from original GLN (https://github.com/Hanjun-Dai/GLN)
* <b>retroxpert</b>, adapted from original RetroXpert (https://github.com/uta-smile/RetroXpert)
* <b>neuralsym</b>, adapted from Min Htoo's re-implementation (https://github.com/linminhtoo/neuralsym)
* <b>transformer</b>, adapted from Augmented Transformer (https://github.com/bigchem/synthesis)
  and Molecular Transformer (https://github.com/pschwllr/MolecularTransformer).
  We (trivially) re-implemented the Transformer using models from OpenNMT, which gave cleaner and more modularized codes. 
  
For example, to benchmark with all 4 models, run
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

(W.I.P.) Nevertheless, in some cases it would be necessary to turn off canonicalization,
e.g. when benchmarking Transformer with augmented non-canonical data.
To benchmark without canonicalization (for Transformer only), run
```
bash scripts/benchmark_in_docker.sh no_canonicalization transformer
```

The estimated running time for benchmarking the USPTO_50k dataset on a 20-core machine with 1 RTX3090 GPU is
* GLN:
  ~1 hr preprocessing, ~2 hrs training, ~3 hrs testing
* RetroXpert:
  ~5 mins stage 1 preprocessing, ~2 hrs stage 1 training,
  ~10 mins stage 2 preprocessing, ~12 hrs stage 2 training, ~20 mins testing
* Transformer:
  ~1 min preprocessing, ~16 hrs training, ~10 mins testing
* NeuralSym:
  ~15 mins preprocessing, ~5 mins training, ~2 mins testing
