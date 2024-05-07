#!/bin/bash

#SBATCH --job-name=heatmap_abmil_CAM16
#SBATCH --output=job_logs/heatmap_abmil%j.txt
#SBATCH --partition=tue.gpu.q
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=2

source /cm/shared/apps/Anaconda/2021.11/pth3.9/etc/profile.d/conda.sh

module load CUDA/11.7.0
conda activate thesis-conda-env6
which python

# Open WSI-finetuning project
cd /home/mcs001/20181133/CLAM/

export CUDA_VISIBLE_DEVICES=0, 1
python create_heatmaps.py --config config_template_abmil.yaml
# python create_heatmaps.py --config config_template.yaml