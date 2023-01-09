#!/bin/bash

#SBATCH --job-name=bids2table
#SBATCH --partition=RM-shared
#SBATCH --ntasks=2
#SBATCH --mem=4000
#SBATCH --time=00:05:00
#SBATCH --array=0-19

eval "$(conda shell.bash hook)"
conda activate python39

bids2table -c config/openneuro_mriqc.yaml -y overrides.yaml \
    collection_id=2022-12-18-1900 \
    worker_id="$SLURM_ARRAY_TASK_ID" \
    num_workers=20
