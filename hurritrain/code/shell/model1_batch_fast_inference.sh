#!/bin/bash
#SBATCH --job-name=ace2_inf
#SBATCH --output=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/slurm/model1_inf_%A_%a.out
#SBATCH --error=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/slurm/model1_inf_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --reservation=hackathon
#SBATCH --constraint=a100


module purge
module load anaconda3/2025.6
conda activate geoclim
module load cudatoolkit/13.1

export WANDB_JOB_TYPE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
echo "MASTER_PORT=$MASTER_PORT"

YAML_FILE=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/code/yaml/model1_fast_inference.yaml

srun /usr/local/bin/nsys profile -o timeline_output_v1_june3_145pm_m1_inf --force-overwrite true --trace cuda,nvtx,osrt torchrun  --master_port=$MASTER_PORT \
     -m fme.ace.batched_evaluator "$YAML_FILE"


