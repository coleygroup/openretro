# OpenRetro
An open source library for benchmarking one-step retrosynthesis.

# Environment Setup
## Option 1. Using Docker (recommended)
### <u>Build Docker with GPU support (<b>only works with Linux-based OS</b>) </u>
Building the Docker image with GPU support will speed up training with any models in general. First follow the instruction on https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
to install the NVIDIA Container Toolkit (a.k.a. nvidia-docker2). Then run
```    
docker build -f Dockerfile_gpu -t openretro:gpu .
```

### <u>Build Docker with CPU only </u>
If you do not need GPU support (e.g. for testing/deployment only), or if you are not using Linux-based OS,
you can choose to build the Docker image with CPU only. First follow the instruction on https://docs.docker.com/engine/install
to install the Docker engine. Then run
```
docker build -f Dockerfile_cpu -t openretro:cpu .
```

## Option 2. Using Conda (not recommended; WIP)
Please contact Zhengkai (ztu@mit.edu) if you have an urgent need for non-Docker environments.
We recommend using Docker in general because some models have dependencies that are platform specific
(e.g. requiring reasonably new G++ for compilation) which we can help you handle inside Docker.

# Sample Usage with Trained Models
Here we demonstrate sample usage with baseline models (re-)trained on USPTO 50k dataset without reaction type.
We have added in four models, GLN, RetroXpert (revised), Transformer and NeuralSym.
Note that the scripts for running Dockers are for the <b> CPU-only </b> images;
extending it to GPU images is as simple as adding the "--gpus all" flag,
though there might be platform specific settings to be configured.  

## (1/4) GLN -- untyped USPTO 50k baseline model 
* Run Docker for serving
``` 
sh scripts/gln_serve_in_docker.sh
```

* Sample query (the "data" field is a single json dict with "smiles" as the key, and list of (optionally atom-mapped) SMILES as the value)
```
curl http://you.got.the.ips:9018/predictions/gln_50k_untyped \
    --header "Content-Type: application/json" \
    --request POST \
    --data '{"smiles": ["[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]", "CC(C)(C)OC(=O)N1CCC(OCCO)CC1"]}'
```

* Sample return
```
List[{
    "template": List[str], list of top k templates,
    "reactants": List[str], list of top k proposed reactants based on the templates,
    "scores": List[float], list of top k corresponding scores
}]
```

Note that the reactants may be duplicated since different templates can give the same reactants.

## (2/4) RetroXpert -- untyped USPTO 50k baseline model

* Run Docker for serving
```
sh scripts/retroxpert_serve_in_docker.sh
```

* Sample query (the "data" field is a single json dict with "smiles" as the key, and list of (optionally atom-mapped) SMILES as the value)
```
curl http://you.got.the.ips:9118/predictions/retroxpert_uspto50k_untyped \
    --header "Content-Type: application/json" \
    --request POST \
    --data '{"smiles": ["[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]", "CC(C)(C)OC(=O)N1CCC(OCCO)CC1"]}'
```

* Sample return
```
List[{
    "reactants": List[str], list of top k proposed reactants,
    "scores": List[float], list of top k corresponding scores
}]
```

## (3/4) Transformer -- untyped USPTO 50k baseline model (TODO)

## (4/4) NeuralSym -- untyped USPTO 50k baseline model (TODO)


# Development with Docker (recommended)
Please rebuild the Docker before running any scripts if there is any change in code.
As a reminder, the docker commands in the scripts are for the CPU-only image.
Please make changes accordingly for GPU image (especially for training and testing).
Training and testing with CPU are likely to be <b>very slow</b>.

## (1/4) GLN
Adapted from original GLN (https://github.com/Hanjun-Dai/GLN)

Step 1. Prepare the raw atom-mapped .csv files for train, validation and test.
See https://www.dropbox.com/sh/6ideflxcakrak10/AADN-TNZnuGjvwZYiLk7zvwra/schneider50k?dl=0&subfolder_nav_tracking=1
for sample data format.
Configure variables in ./scripts/config_gln.sh to point to the <b>absolute</b> path of 
raw files and desired output paths. Then execute the configuration script with

    source scripts/config_gln.sh

Step 2. Preprocessing

    sh scripts/gln_preprocess_in_docker.sh

Step 3. Training
    
    sh scripts/gln_train.sh

Step 4. Testing
    
    sh scripts/gln_test.sh

Once trained, a sample usage of the GLN proposer API is 

    python sample_gln_proposer.py
Refer to sample_gln_proposer.py and modify accordingly for your own use case.

## Transformer
Based on:  
OpenNMT (https://opennmt.net/OpenNMT-py/)  
Molecular Transformer (https://github.com/pschwllr/MolecularTransformer)  
Bigchem/Karpov (https://github.com/bigchem/retrosynthesis)

Step 1. Prepare the raw SMILES .txt/.smi files for train, validation and test.
See https://github.com/bigchem/retrosynthesis/tree/master/data
for sample data format.

Step 2. Preprocessing: modify the args in scripts/transformer_preprocess.sh, then

    sh scripts/transformer_preprocess.sh

Step 3. Training: modify the args in scripts/transformer_train.sh, then
    
    sh scripts/transformer_train.sh

Step 4 (optional). Testing: modify the args in scripts/transformer_test.sh, then
    
    sh scripts/transformer_test.sh
    
NOTE: DO NOT change flags marked as "do_not_change_this"

Once trained, a sample usage of the Transformer proposer API is 

    python sample_transformer_proposer.py
Refer to sample_transformer_proposer.py and modify accordingly for your own use case.

## Development with Conda (not recommended; WIP)
Assuming conda is installed and initiated (i.e. conda activate is a warning-free command).
Then run the following command on a machine with CUDA

    bash -i scripts/setup.sh
    conda activate openretro