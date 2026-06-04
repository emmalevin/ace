#!/bin/bash
#SBATCH --job-name=ace_train_pair
#SBATCH --output=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/slurm/train_%x_%A.out
#SBATCH --error=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/slurm/train_%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --account=gvecchi
#SBATCH --reservation=hackathon
#SBATCH --constraint=a100

set -euo pipefail

: "${YAML_FILE_MODEL1:?Set YAML_FILE_MODEL1 to the model1 training YAML path}"
: "${YAML_FILE_MODEL2:?Set YAML_FILE_MODEL2 to the model2 training YAML path}"

NPROC_PER_MODEL="${NPROC_PER_MODEL:-2}"
PROFILE_MODEL1="${PROFILE_MODEL1:-0}"
PROFILE_MODEL2="${PROFILE_MODEL2:-0}"
PROFILE_OUTPUT_PREFIX="${PROFILE_OUTPUT_PREFIX:-timeline_output_training}"

module purge
module load anaconda3/2025.6
conda activate geoclim
module load cudatoolkit/13.1

export WANDB_JOB_TYPE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "${ALLOCATED_CUDA_VISIBLE_DEVICES}" ]]; then
    IFS=',' read -r -a ALLOCATED_GPUS <<< "${ALLOCATED_CUDA_VISIBLE_DEVICES}"
else
    ALLOCATED_GPUS=(0 1 2 3)
fi
REQUIRED_GPUS=$((NPROC_PER_MODEL * 2))
if (( ${#ALLOCATED_GPUS[@]} < REQUIRED_GPUS )); then
    echo "Need ${REQUIRED_GPUS} GPUs for paired training, but Slurm exposed ${#ALLOCATED_GPUS[@]}: ${ALLOCATED_CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
fi

gpu_slice() {
    local start="$1"
    local count="$2"
    local values=("${ALLOCATED_GPUS[@]:start:count}")
    local joined
    joined=$(IFS=,; echo "${values[*]}")
    echo "${joined}"
}

MODEL1_GPUS="${MODEL1_GPUS:-$(gpu_slice 0 "${NPROC_PER_MODEL}")}"
MODEL2_GPUS="${MODEL2_GPUS:-$(gpu_slice "${NPROC_PER_MODEL}" "${NPROC_PER_MODEL}")}"
MASTER_PORT_MODEL1=$((20000 + SLURM_JOB_ID % 30000))
MASTER_PORT_MODEL2=$((MASTER_PORT_MODEL1 + 1))

echo "YAML_FILE_MODEL1=${YAML_FILE_MODEL1}"
echo "YAML_FILE_MODEL2=${YAML_FILE_MODEL2}"
echo "NPROC_PER_MODEL=${NPROC_PER_MODEL}"
echo "ALLOCATED_CUDA_VISIBLE_DEVICES=${ALLOCATED_CUDA_VISIBLE_DEVICES:-unset}"
echo "MODEL1_GPUS=${MODEL1_GPUS}"
echo "MODEL2_GPUS=${MODEL2_GPUS}"
echo "MASTER_PORT_MODEL1=${MASTER_PORT_MODEL1}"
echo "MASTER_PORT_MODEL2=${MASTER_PORT_MODEL2}"

run_model() {
    local model_id="$1"
    local yaml_file="$2"
    local gpu_ids="$3"
    local master_port="$4"
    local profile="$5"
    local profile_output="${PROFILE_OUTPUT_PREFIX}_m${model_id}_${SLURM_JOB_ID}"

    local cmd=(
        torchrun
        --master_port="${master_port}"
        --nproc_per_node="${NPROC_PER_MODEL}"
        -m fme.ace.train
        "${yaml_file}"
    )

    echo "Launching model${model_id} training on CUDA_VISIBLE_DEVICES=${gpu_ids}"
    if [[ "${profile}" == "1" || "${profile}" == "true" ]]; then
        CUDA_VISIBLE_DEVICES="${gpu_ids}" MASTER_PORT="${master_port}"             /usr/local/bin/nsys profile -o "${profile_output}" --force-overwrite true --trace cuda,nvtx,osrt "${cmd[@]}"
    else
        CUDA_VISIBLE_DEVICES="${gpu_ids}" MASTER_PORT="${master_port}" "${cmd[@]}"
    fi
}

run_model 1 "${YAML_FILE_MODEL1}" "${MODEL1_GPUS}" "${MASTER_PORT_MODEL1}" "${PROFILE_MODEL1}" &
pid_model1=$!
run_model 2 "${YAML_FILE_MODEL2}" "${MODEL2_GPUS}" "${MASTER_PORT_MODEL2}" "${PROFILE_MODEL2}" &
pid_model2=$!

set +e
wait "${pid_model1}"
status_model1=$?
wait "${pid_model2}"
status_model2=$?
set -e

if (( status_model1 != 0 || status_model2 != 0 )); then
    echo "Training failed: model1 exit=${status_model1}, model2 exit=${status_model2}" >&2
    exit 1
fi

echo "Both training models completed successfully."
