# openretro
An open source library for retrosynthesis benchmarking and planning.

# Status for ASKCOS Integration with GLN (Tentative)
- [x] Add in GLN handler/archiver
- [x] Add in RetroXpert handler/archiver
- [x] Containarize GLN and RetroXpert deployment in single Docker with USPTO 50k baseline(s)
- [ ] Finalize inputs/outputs/configs with Max
- [ ] RetroXpert checkpoint trained on larger dataset
- [ ] Clean up doc for offline training
- [ ] (Nah, too long) GLN checkpoint trained on larger dataset
- [ ] (Nah, lower priority) Support online training (might want to use GPU)

# Deployment
### Build docker
```    
docker build -t openretro-serving:dev .
```

### For RetroXpert -- untyped USPTO 50k baseline model

* Run docker for serving
```
sh scripts/retroxpert_serve_in_docker.sh
```

* Sample query (the "data" field is a single json dict with "smiles" as the key, and list of (optionally atom-mapped) SMILES as the value)
```
curl http://you.got.the.ips:9818/predictions/retroxpert_uspto50k_untyped \
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

### For GLN -- untyped USPTO 50k baseline model 

* Run docker for serving
``` 
sh scripts/gln_serve_in_docker.sh
```

* Sample query (the "data" field is a single json dict with "smiles" as the key, and list of (optionally atom-mapped) SMILES as the value)
```
curl http://you.got.the.ips:9918/predictions/gln_50k_untyped \
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

Note that the reactants may be duplicated (from different templates)

------------------------------------UP TO HERE for ASKCOS INTEGRATION------------------------------------

# Development
### Using conda
Assuming conda is installed and initiated (i.e. conda activate is a warning-free command).
Then run the following command on a machine with CUDA

    bash -i scripts/setup.sh
    conda activate openretro

This will ensure GLN uses the CUDA ops (vs. CPU ops) in GPU training.

# Models
## GLN
Adapted from original GLN (https://github.com/Hanjun-Dai/GLN)

Step 1. Prepare the raw atom-mapped .csv files for train, validation and test.
See https://www.dropbox.com/sh/6ideflxcakrak10/AADN-TNZnuGjvwZYiLk7zvwra/schneider50k?dl=0&subfolder_nav_tracking=1
for sample data format.

Step 2. Preprocessing: modify the args in scripts/gln_preprocess.sh, then

    sh scripts/gln_preprocess.sh

Step 3. Training: modify the args in scripts/gln_train.sh, then
    
    sh scripts/gln_train.sh

Step 4 (optional). Testing: modify the args in scripts/gln_test.sh, then
    
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