#!/bin/bash
#SBATCH --job-name=ace_train_m2
#SBATCH --output=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/slurm/model2_train_%A.out
#SBATCH --error=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/slurm/model2_train_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --account=gvecchi
##SBATCH --reservation=hackathon
##SBATCH --constraint=a100

module purge
module load anaconda3/2025.6
conda activate geoclim
module load cudatoolkit/13.1

export WANDB_JOB_TYPE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
echo "MASTER_PORT=$MASTER_PORT"

YAML_FILE=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/code/yaml/model2_train_initial.yaml

torchrun --master_port=$MASTER_PORT \
    -m fme.ace.train "$YAML_FILE"
