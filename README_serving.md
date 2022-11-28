# OpenRetro-serving
Serving modules for models in OpenRetro

# Environment Setup (CPU only)
First follow the instruction on https://docs.docker.com/engine/install
to install the Docker engine. Then either
* pull pre-built image from ASKCOS docker registry and tag (MLPDS permission required)
```
docker pull registry.gitlab.com/mlpds_mit/askcos/askcos-data/openretro:serving-cpu-public
docker tag registry.gitlab.com/mlpds_mit/askcos/askcos-data/openretro:serving-cpu-public openretro:serving-cpu
```
or
* build from local
```
docker build -f Dockerfile_serving -t openretro:serving-cpu .
```

# Download Pre-compiled Model Archives
```
sh scripts_serving/download_prebuilt_archives.sh
```

# Run Docker for Serving Trained Models (CPU only)
```
sh scripts_serving/gln_serve_in_docker.sh
sh scripts_serving/neuralsym_serve_in_docker.sh
sh scripts_serving/transformer_serve_in_docker.sh
sh scripts_serving/retroxpert_serve_in_docker.sh
sh scripts_serving/localretro_serve_in_docker.sh
```
By default, any one of the serving scripts would serve all available model archives.
Change the arguments after the --models flag (in the script) before running to specify only the desired archives otherwise.

# Sample query
```
curl http://0.0.0.0:port/predictions/MODEL_NAME \
    --header "Content-Type: application/json" \
    --request POST \
    --data '{"smiles": ["[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]", "CC(C)(C)OC(=O)N1CCC(OCCO)CC1"]}'
```

This query format should work for any served model. Specifically,  
  **0.0.0.0**: the server ip or server name  
  **port**: the port of served model.
By default, it is 9917, 9927, 9937, 9947, 9957 for GLN, NeuralSym, Transformer, RetroXpert and LocalRetro respectively.  
  **MODEL_NAME**: any of USPTO_50k_gln, {USPTO_50k,USPTO_full,pistachio_21Q1}_neuralsym,
{USPTO_50k,USPTO_full,pistachio_21Q1}_transformer, {USPTO_50k,USPTO_full}_retroxpert, USPTO_50k_localretro.  
  **--data**： a single json dict with "smiles" as the key, and list of SMILES as the value.
It is okay to pass in SMILES with atom mapping numbers.
All models would remove atom mapping numbers and canonicalize the target SMILES before inference.

# Sample return
* For GLN and NeuralSym
```
List[{
    "templates": List[str], list of top k templates,
    "reactants": List[str], list of top k proposed reactants based on the templates,
    "scores": List[float], list of top k corresponding scores
}]
```
Note that the reactants may be duplicated since different templates can give the same reactants.

* For Transformer, RetroXpert and LocalRetro
```
List[{
    "reactants": List[str], list of top k proposed reactants,
    "scores": List[float], list of top k corresponding scores
}]
```

# [Optional] Create own model archives
If you want to create servable model archives from own checkpoints (e.g. trained on different datasets),
please refer to the archiving scripts (scripts_serving/{gln,neuralsym,transformer,retroxpert,localretro}_archive_model_in_docker.sh).
Change the arguments accordingly in the script before running.
It's mostly bookkeeping by replacing the data name and/or checkpoint paths; the script should be self-explanatory.

