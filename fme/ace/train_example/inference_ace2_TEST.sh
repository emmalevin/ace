#!/bin/bash
#SBATCH --job-name=ace2
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=gvecchi
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=el2358@princeton.edu

module purge
module load anaconda3/2025.6
conda activate geoclim

torchrun --nproc_per_node 1 -m fme.ace.train train_sample_subset.yaml