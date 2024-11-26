PARTITION=l40s
SUB_SCRIPT=convert-70B.sh
SLURMOUT_DIR=slurmout_convert-70B
mkdir -p $SLURMOUT_DIR

sbatch\
 -o ${SLURMOUT_DIR}/slurm-%A_%a.out \
 --time=0-01:00:00 \
 --mem=200G \
 -G 0 \
 -p ${PARTITION} \
 ${SUB_SCRIPT}


