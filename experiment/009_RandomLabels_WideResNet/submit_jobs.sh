#!/bin/bash

# Define the arrays
CORRUPT=(0. 21 24 28)

# Loop through each combination of parameters
for corruption_percentage in "${CORRUPT[@]}"; do
    
    # Create a unique job name based on the parameters
    JOB_NAME="train_corrupt${corruption_percentage}"
    OUTPUT_FILE="slurm_${JOB_NAME}_%j.out"
    ERROR_FILE="slurm_${JOB_NAME}_%j.err"
    
    # Create the SBATCH file content with the current parameters
    sbatch_script="#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=36GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=END
#SBATCH --mail-user=cm6627@nyu.edu
#SBATCH --output=${OUTPUT_FILE}
#SBATCH --error=${ERROR_FILE}

module purge

singularity exec --nv \
            --overlay /scratch/cm6627/diffeo_cnn/diffeo_singularity.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c \"source /ext3/env.sh; python ./fitting-random-labels/train.py --batch-size=128 --wrn-droprate=0.2 --wrn-depth=10 --label-corrupt-prob=${corruption_percentage}\""

    # Save the SBATCH script to a temporary file
    sbatch_file=$(mktemp)
    echo "${sbatch_script}" > "${sbatch_file}"
    
    # Submit the job
    sbatch "${sbatch_file}"
    
    # Remove the temporary file
    rm "${sbatch_file}"

done