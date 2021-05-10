#!/bin/bash
#SBATCH -n 1                                # number of comands issued in batch script at any times
#SBATCH --cpus-per-task=16                  # number of cores to allocate each task
#SBATCH -N 1                                # Request 1 node
#SBATCH --time=15:00:00

#SBATCH --gres=gpu:1                        # specify GPU count
#SBATCH -p sched_mit_ccoley                 # sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=4000                  # Request 1G of memory per CPU

#SBATCH -o logs/output_%x_%j.txt               # redirect output to output_JOBNAME_JOBID.txt
#SBATCH -e logs/error_%x_%j.txt                # redirect errors to error_JOBNAME_JOBID.txt
#SBATCH -J gln_test_77777777
#SBATCH --mail-type=BEGIN,END               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

#SBATCH --nodelist node1237

# conda activate openretro
module load gcc/8.3.0
module load cuda/10.1
source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate openretro_cu101

python3 test.py \
  --test_all_ckpts \
  --model_name="gln" \
  --data_name="schneider50k" \
  --log_file="gln_test_77777777" \
  --train_file="./data/gln_schneider50k/clean_train.csv" \
  --val_file="./data/gln_schneider50k/clean_valid.csv" \
  --test_file="./data/gln_schneider50k/clean_test.csv" \
  --processed_data_path="./data/gln_schneider50k/processed" \
  --model_path="./checkpoints/gln_schneider50k_77777777" \
  --test_output_path="./results/gln_schneider50k_77777777"
