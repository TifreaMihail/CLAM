#!/bin/bash

#SBATCH --job-name=hyperopt_controller
#SBATCH --output=/home/mcs001/20181133/CLAM/job_logs/controller_%j.txt
#SBATCH --partition=tue.gpu.q
#SBATCH --time=2-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus=1

# Load environment
source /cm/shared/apps/Anaconda/2021.11/pth3.9/etc/profile.d/conda.sh
module load CUDA/11.7.0
conda activate thesis-conda-env6

# Change to project directory
cd /home/mcs001/20181133/CLAM/

# Check if configuration file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch run_bo_controller.sh <config_file>"
    exit 1
fi

CONFIG_FILE=$1

# Run the Bayesian optimization controller with the configuration file
python /home/mcs001/20181133/CLAM/scripts/Bayesian_opt/bo_controller_inline.py --config $CONFIG_FILE
