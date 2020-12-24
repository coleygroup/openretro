# openretro
An open source library for retrosynthesis benchmarking.

# Environment setup
### Using conda
    bash -i scripts/setup.sh
    conda activate openretro

### Using docker (TODO)

# Models
## GLN
Adapted from https://github.com/Hanjun-Dai/GLN

Step 1. Prepare the raw atom-mapped .csv files for train, validation and test.
See https://www.dropbox.com/sh/6ideflxcakrak10/AADN-TNZnuGjvwZYiLk7zvwra/schneider50k?dl=0&subfolder_nav_tracking=1
for how sample data look like

Step 2. Preprocessing: modify the args in scripts/gln_preprocess.sh, then

    sh scripts/gln_preprocess.sh

Step 3. Training: modify the args in scripts/gln_train.sh, then
    
    sh scripts/gln_train.sh

Step 4 (optional). Testing: modify the args in scripts/gln_test.sh, then
    
    sh scripts/gln_test.sh

Once trained, a sample usage of the GLN proposer API is 

    python sample_gln_proposer.py
Refer to gln_proposer_sample.py and modify accordingly for your own use case.