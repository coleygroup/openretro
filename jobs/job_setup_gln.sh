#!/bin/bash
#SBATCH -n 1                                # number of comands issued in batch script at any times
#SBATCH --cpus-per-task=4                  # number of cores to allocate each task
#SBATCH -N 1                                # Request 1 node
#SBATCH --time=0:10:00

##SBATCH --gres=gpu:1                        # specify GPU count
#SBATCH -p sched_mit_ccoley                 # sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=2000                  # Request 1G of memory per CPU

#SBATCH -o logs/output_%x_%j.txt               # redirect output to output_JOBNAME_JOBID.txt
#SBATCH -e logs/error_%x_%j.txt                # redirect errors to error_JOBNAME_JOBID.txt
#SBATCH -J setup_gln                         # name of job
#SBATCH --mail-type=BEGIN,END               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

##SBATCH -C centos7                          # Request only Centos7 nodes, maybe not necessary
##SBATCH --mem-=128G                         # memory pool for each task
##SBATCH --nodelist=node1236                 # specify GPU node on our group

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh

conda activate openretro # _cu102

# module load gcc/8.3.0
# module unload gcc/8.3.0
# module load gcc/6.2.0
# module load cuda/10.0

## https://askubuntu.com/questions/420981/how-do-i-save-terminal-output-to-a-file
bash setup_gln.sh &> setup_gln.txt