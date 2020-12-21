# openretro
An open source library for retrosynthesis benchmarking.

# Environment setup
### Using conda
    bash -i scripts/setup.sh
    conda activate openretro
    pip install -e .

### Using docker (TODO)

# Preprocessing
Modify the variables at the top of scripts/preprocess.sh, then

    sh scripts/preprocess.sh

# Training
Modify the variables at the top of scripts/train.sh, then
    
    sh scripts/train.sh

# Testing (partially supported)
Modify the variables at the top of scripts/test.sh, then
    
    sh scripts/test.sh

# Using the prediction API
With the package installed, only the trained checkpoint is needed
to instantiate the predictor API. See ./examples for sample usage.