#!/bin/bash

#SBATCH --job-name=eval_abmil_CAM16
#SBATCH --output=job_logs/eval_abmil%j.txt
#SBATCH --partition=tue.gpu.q
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=1

source /cm/shared/apps/Anaconda/2021.11/pth3.9/etc/profile.d/conda.sh

module load CUDA/11.7.0
conda activate thesis-conda-env6
which python

# Open WSI-finetuning project
cd /home/mcs001/20181133/CLAM/

export CUDA_VISIBLE_DEVICES=0

python eval.py --k 5 --models_exp_code camelyon16_abmil_50_s1 --save_exp_code eval_camelyon16_abmil_50_s1_cv --task task_1_tumor_vs_normal --model_type abmil --results_dir results --data_root_dir /home/mcs001/20181133/CLAM/data_feat --splits_dir /home/mcs001/20181133/CLAM/splits/task_1_tumor_vs_normal_100