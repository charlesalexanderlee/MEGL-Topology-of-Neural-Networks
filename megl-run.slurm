#!/bin/bash
#SBATCH --job-name=megl_tnn_${USER}
#SBATCH --partition=gpuq               # the DGX only belongs in the 'gpu' partition
#SBATCH --qos=gpu                            # need to select 'gpu' QoS
#SBATCH --output=${USER_PATH}MEGL-Topology-of-Neural-Networks/SLURM_output/megl_output.%j.txt
#SBATCH --error=${USER_PATH}MEGL-Topology-of-Neural-Networks/SLURM_output/megl_error.%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # up to 128; 
#SBATCH --gres=gpu:A100.80gb:1                # up to 8; only request what you need
#SBATCH --mem-per-cpu=40GB                     # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=5-00:00:00                       # set to 1hr; please choose carefully
#SBATCH --mail-type=BEGIN,END,FAIL              # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=${USER}@gmu.edu            # Put your GMU email address here

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10
module load python
module load cudnn
module load cmake

pip install -r ${USER_PATH}MEGL-Topology-of-Neural-Networks/requirements.txt

python ${USER_PATH}MEGL-Topology-of-Neural-Networks/src/WAB_Cifer10.py
