#!/bin/bash
#SBATCH --job-name=active_sampling
#SBATCH --output=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/slurm/loop_point_selection_%j.out
#SBATCH --error=/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/slurm/loop_point_selection_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=gvecchi
#SBATCH --reservation=hackathon


# Active learning driver: updates YAMLs, sbatch/waits for training and inference jobs.
# GPU is not required on this job; child jobs use their own SBATCH (--gres=gpu).

#set -euo pipefail

#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
#PYTHON_DIR="${CODE_DIR}/python"
# Override with: sbatch --export=CONFIG=/path/to/active_sampling.yaml this_script.sh
#CONFIG="${CONFIG:-${CODE_DIR}/yaml/active_sampling.yaml}"
CONFIG="/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/code/yaml/active_sampling.yaml"

module purge
module load anaconda3/2025.6
conda activate geoclim

export WANDB_JOB_TYPE=disabled

cd /scratch/gpfs/GVECCHI/el2358/ace/hurritrain/code/python
python loop_point_selection.py --config "${CONFIG}"
