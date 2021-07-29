# OpenRetro-serving
Serving modules for models in OpenRetro

# Environment Setup
## Using Docker (CPU only)
First follow the instruction on https://docs.docker.com/engine/install
to install the Docker engine. Then either
* pull pre-built image from ASKCOS docker registry and tag
```
docker pull registry.gitlab.com/mlpds_mit/askcos/askcos-data/openretro:serving-cpu
docker tag registry.gitlab.com/mlpds_mit/askcos/askcos-data/openretro:serving-cpu openretro:serving-cpu
```
or
* build from local

first download the model checkpoints (USPTO_50k w/o reaction types).
Note that ASKCOS deployment authentication is needed (https://mlpds.mit.edu/member-resources-releases-versions/)
```
bash scripts_serving/download_checkpoints_50k_untyped.sh
```
then run
```
docker build -f Dockerfile_cpu -t openretro:serving-cpu .
```

# Run Docker for Serving Trained Models (CPU only)
```
sh scripts_serving/serve_all_in_docker.sh
```

# Sample Usage
Here we demonstrate sample usage with baseline models (re-)trained on USPTO 50k dataset without reaction type.
We have added in four models, GLN, RetroXpert (revised), Transformer and NeuralSym.

## (1/4) GLN -- untyped USPTO 50k baseline model 
* Sample query (the "data" field is a single json dict with "smiles" as the key, and list of (optionally atom-mapped) SMILES as the value)
```
curl http://you.got.the.ips:8080/predictions/gln_50k_untyped \
    --header "Content-Type: application/json" \
    --request POST \
    --data '{"smiles": ["[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]", "CC(C)(C)OC(=O)N1CCC(OCCO)CC1"]}'
```

* Sample return
```
List[{
    "templates": List[str], list of top k templates,
    "reactants": List[str], list of top k proposed reactants based on the templates,
    "scores": List[float], list of top k corresponding scores
}]
```

Note that the reactants may be duplicated since different templates can give the same reactants.

## (2/4) RetroXpert -- untyped USPTO 50k baseline model
* Sample query (the "data" field is a single json dict with "smiles" as the key, and list of (optionally atom-mapped) SMILES as the value)
```
curl http://you.got.the.ips:8080/predictions/retroxpert_uspto50k_untyped \
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

## (3/4) Transformer -- untyped USPTO 50k baseline model
* Sample query (the "data" field is a single json dict with "smiles" as the key, and list of (optionally atom-mapped) SMILES as the value)
```
curl http://you.got.the.ips:8080/predictions/transformer_50k_untyped \
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

## (4/4) NeuralSym -- untyped USPTO 50k baseline model
* Sample query (the "data" field is a single json dict with "smiles" as the key, and list of (optionally atom-mapped) SMILES as the value)
```
curl http://you.got.the.ips:8080/predictions/neuralsym_50k \
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
