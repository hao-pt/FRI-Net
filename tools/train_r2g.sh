#!/bin/bash
#SBATCH -J frinet_r2g              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=elor,gpu         # Request partition
#SBATCH --constraint="[a6000|a100|6000ada]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --cpus-per-gpu=4              # Number of CPU cores per gpu
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

JOB_NAME=frinet_r2g
for PHASE in 0 1 2
do
    python train_stru3d.py --phase=${PHASE} --job_name=${JOB_NAME} \
        --input_channels 3 \
        --img_folder data/R2G_hr_dataset_processed_v1/ \
        --occ_folder data/R2G_hr_dataset_processed_v1/occ \
        --ids_path null \
        --job_name ${JOB_NAME} \
        --batch_size 16 \
        --resume checkpoints/${JOB_NAME}/checkpoint_${PHASE}.pth
done

# python train_stru3d.py --phase=1 --job_name=train_stru3d
# python train_stru3d.py --phase=2 --job_name=train_stru3d