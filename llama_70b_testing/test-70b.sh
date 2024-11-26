#!/bin/sh -x
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus-per-node=3
#SBATCH --mem=300G
#SBATCH --partition=h100
#SBATCH --time=0-00:10:00


# Already copied the model weights over. 
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source /home/korchins/.bashrc
conda activate llama_nlp
python open-70B.py > test-70b.log