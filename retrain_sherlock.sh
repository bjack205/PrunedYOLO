#!/bin/bash
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#set a job name
#SBATCH --job-name=YOLO_retrain
#################
#a file for job output, you can check job progress
#SBATCH --output=YOLO_retrain.out
#################
# a file for errors from the job
#SBATCH --error=YOLO_retrain.err
#################
#time you think you need; default is one hour
#in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the faster your job will run.
# Default is one hour, this example will run in  less that 5 minutes.
#SBATCH --time=2:00:00
#################
# --gres will give you one GPU, you can ask for more, up to 8 (or how ever many are on the node/card)
#SBATCH --gres gpu:1
# We are submitting to the gpu partition, if you can submit to the hns partition, change this to -p hns_gpu.
#SBATCH -p gpu
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=16000
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=YourSUNetID@stanford.edu

module load py-tensorflow/1.5.0_py36
module load viz
module load py-matplotlib/2.1.2_py36
# module load py-keras/2.1.4  # This is the desired version
srun  python3 ~/PrunedYOLO/retrain.py
