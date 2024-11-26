#!/bin/sh -x
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus-per-node=1



# Already copied the model weights over. 
# scp -r korchins@pcslsrv1.epfl.ch:~/.llama/checkpoints/Llama3.1-70B-Instruct /scratch/korchins/
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source /home/korchins/.bashrc
conda activate llama_nlp
cd /home/korchins/projects/llm_unmasking/environment_setup/transformers
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /scratch/korchins/Llama3.1-70B-Instruct --model_size 70B\
   --llama_version 3.1\
   --output_dir /scratch/korchins/llama_hf_weights/Llama3.1-70B-Instruct

#once the computation is done, we move it to the group store. 
cp -r /scratch/korchins/llama_hf_weights/Llama3.1-70B-Instruct /work/pcsl/model_weights/llama_hf_weights/