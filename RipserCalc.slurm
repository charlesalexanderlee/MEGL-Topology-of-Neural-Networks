#!/bin/bash
#SBATCH --job-name=megl_tnn
#SBATCH --partition=gpuq                        # the DGX only belongs in the 'gpu' partition
#SBATCH --qos=gpu                               # need to select 'gpu' QoS
#SBATCH --output=OUTPUT_PATH(REPLACE_ME)
#SBATCH --error=ERROR_PATH(REPLACE_ME)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # up to 128; 
#SBATCH --gres=gpu:1g.10gb:1               # up to 8; only request what you need
#SBATCH --mem-per-cpu=350GB                     # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=0-15:00:00                       # set to 1hr; please choose carefully
#SBATCH --mail-type=BEGIN,END,FAIL              # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=EMAIL_ADDRESS(REPLACE_ME)            # Put your email address here

set echo
umask 0027


module load gnu10
module load python
module load cudnn
module load cmake


source VIRTUAL_ENV_DIRECTORY

python '[USER_PATH]'/MEGL-Topology-of-Neural-Networks/RipserCalculations.py


