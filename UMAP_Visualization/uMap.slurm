#!/bin/bash
#SBATCH --job-name=megl_tnn
#SBATCH --partition=gpuq                        # the DGX only belongs in the 'gpu' partition
#SBATCH --qos=gpu                               # need to select 'gpu' QoS
#SBATCH --output=/scratch/jjung43/MEGL-Topology-of-Neural-Networks/SLURM_output/megl_output.%j.txt
#SBATCH --error=/scratch/jjung43/MEGL-Topology-of-Neural-Networks/SLURM_output/megl_error.%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # up to 128; 
#SBATCH --gres=gpu:A100.40gb:1                # up to 8; only request what you need
#SBATCH --mem-per-cpu=80GB                     # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=0-00:30:00                       # set to 1hr; please choose carefully
#SBATCH --mail-type=BEGIN,END,FAIL              # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=jjung43@gmu.edu            # Put your GMU email address here

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10
module load python
module load cudnn
module load cmake


source /home/jjung43/venv/bin/activate

python /scratch/jjung43/MEGL-Topology-of-Neural-Networks/UMapCifar103D.py


