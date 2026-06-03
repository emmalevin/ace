#!/bin/bash
#SBATCH --job-name=ace_train_m1
#SBATCH --output=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/slurm/model1_train_%A.out
#SBATCH --error=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/slurm/model1_train_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --account=gvecchi
#SBATCH --reservation=hackathon
#SBATCH --constraint=a100
#SBATCH --exclusive

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

YAML_FILE=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/code/yaml/model1_train_initial.yaml

srun /usr/local/bin/nsys profile -o timeline_output_training_m1 --force-overwrite true --trace cuda,nvtx,osrt --target-processes=all torchrun  --master_port=$MASTER_PORT --nproc_per_node=2 \
    -m fme.ace.train "$YAML_FILE"
