#!/bin/bash
#SBATCH --job-name=megl_tnn_multi_gpu
#SBATCH --partition=gpuq                        # the DGX only belongs in the 'gpu' partition
#SBATCH --qos=gpu                               # need to select 'gpu' QoS
#SBATCH --output=/scratch/cleemart/MEGL_TNN/output/megl_tnn_output.%j.txt
#SBATCH --error=/scratch/cleemart/MEGL_TNN/error/megl_tnn_error.%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                     # up to 128;
#SBATCH --gres=gpu:A100.80gb:2                  # up to 8, here we are requesting 2 GPUs; A100.80gb:<num_gpu>
#SBATCH --mem-per-cpu=40GB                      # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL
#SBATCH --time=0-01:00:00                       # set to 1hr; please choose carefully
#SBATCH --mail-type=BEGIN,END,FAIL              # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=cleemart@gmu.edu            # Put your GMU email address here

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10
module load python
module load cuda/12.3.1
module load cudnn/8.9.7.29-12.3.1

source /home/cleemart/venv/bin/activate
python /home/cleemart/MEGL-Topology-of-Neural-Networks/src/WAB_Cifar10.py
