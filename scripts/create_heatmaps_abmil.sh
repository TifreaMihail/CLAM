#!/bin/bash

#SBATCH --job-name=heatmap_abmil_CAM16
#SBATCH --output=job_logs/heatmaps/abmil_%j.txt
#SBATCH --partition=tue.default.q
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G

source /cm/shared/apps/Anaconda/2021.11/pth3.9/etc/profile.d/conda.sh

module load CUDA/11.7.0
conda activate thesis-conda-env6
which python

# Open WSI-finetuning project
cd /home/mcs001/20181133/CLAM/

# export CUDA_VISIBLE_DEVICES=0
python create_heatmaps.py --config_file config_template_abmil.yaml
# python create_heatmaps.py --config config_template.yaml