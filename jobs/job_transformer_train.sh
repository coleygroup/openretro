#!/bin/bash
#SBATCH -n 1                                # number of comands issued in batch script at any times
#SBATCH --cpus-per-task=10                  # number of cores to allocate each task
#SBATCH -N 1                                # Request 1 node
#SBATCH --time=24:00:00

##SBATCH --gres=gpu:1                        # specify GPU count
#SBATCH -p sched_mit_ccoley                 # sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=2000                  # Request 1G of memory per CPU

#SBATCH -o logs/output_%x_%j.txt               # redirect output to output_JOBNAME_JOBID.txt
#SBATCH -e logs/error_%x_%j.txt                # redirect errors to error_JOBNAME_JOBID.txt
#SBATCH -J transformer_train                         # name of job
#SBATCH --mail-type=BEGIN,END               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

##SBATCH -C centos7                          # Request only Centos7 nodes, maybe not necessary
##SBATCH --mem-=128G                         # memory pool for each task
##SBATCH --nodelist=node1236                 # specify GPU node on our group

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh

conda activate openretro

python3 train.py \
  --do_train \
  --data="do_not_change_this" \
  --model_name="transformer" \
  --data_name="transformer-karpov" \
  --log_file="transformer_train_1milsteps" \
  --processed_data_path="./data/transformer-karpov/processed/" \
  --model_path="./checkpoints/transformer-karpov" \
  -seed 42 \
  -gpu_ranks 0 \
  -save_checkpoint_steps 20000 \
  -keep_checkpoint 10 \
  -train_steps 1000000 \
  -param_init 0 \
  -param_init_glorot \
  -max_generator_batches 32 \
  -batch_size 3000 \
  -batch_type tokens \
  -normalization tokens \
  -max_grad_norm 0 \
  -optim adam \
  -adam_beta1 0.9 \
  -adam_beta2 0.998 \
  -decay_method noam \
  -warmup_steps 16000 \
  -learning_rate 2 \
  -label_smoothing 0.0 \
  -report_every 1000 \
  -layers 3 \
  -rnn_size 64 \
  -word_vec_size 64 \
  -encoder_type transformer \
  -decoder_type transformer \
  -dropout 0.1 \
  -position_encoding \
  -share_embeddings \
  -global_attention general \
  -global_attention_function softmax \
  -self_attn_type scaled-dot \
  --heads 8 \
  -transformer_ff 512