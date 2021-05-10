#!/bin/bash
#SBATCH -n 1                                # number of comands issued in batch script at any times
#SBATCH --cpus-per-task=64                  # number of cores to allocate each task
#SBATCH -N 1                                # Request 1 node
#SBATCH --time=2:00:00

##SBATCH --gres=gpu:4                        # specify GPU count
#SBATCH -p sched_mit_ccoley                 # sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=1000                  # Request 1G of memory per CPU

#SBATCH -o logs/output_%x_%j.txt               # redirect output to output_JOBNAME_JOBID.txt
#SBATCH -e logs/error_%x_%j.txt                # redirect errors to error_JOBNAME_JOBID.txt
#SBATCH -J gln_preprocess                        # name of job
#SBATCH --mail-type=BEGIN,END               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

##SBATCH -C centos7                          # Request only Centos7 nodes, maybe not necessary
##SBATCH --mem-=128G                         # memory pool for each task
##SBATCH --nodelist=node1236                 # specify GPU node on our group

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate openretro

python3 preprocess.py \
  --model_name="gln" \
  --data_name="schneider50k" \
  --log_file="gln_preprocess" \
  --train_file="./data/gln_schneider50k/clean_train.csv" \
  --val_file="./data/gln_schneider50k/clean_valid.csv" \
  --test_file="./data/gln_schneider50k/clean_test.csv" \
  --processed_data_path="./data/gln_schneider50k/processed" \
  --num_cores=64