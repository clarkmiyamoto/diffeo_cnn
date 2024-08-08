#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=36GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_corrupt_0p5
#SBATCH --mail-type=END
#SBATCH --mail-user=cm6627@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge

singularity exec --nv \
	    --overlay /scratch/cm6627/diffeo_cnn/diffeo_singularity.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python ./fitting-random-labels/train.py --wrn-depth=22 --label-corrupt-prob=0.0"
