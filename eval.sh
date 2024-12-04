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

# learning_rates=(1e-3 1e-4 1e-5)
# regularizations=(1e-3 1e-4 1e-5 1e-6)
learning_rates=(1e-3)
regularizations=(1e-5)
for lr in "${learning_rates[@]}"
do
    for rg in "${regularizations[@]}"
    do
        model_code="camelyon16_abmil_50_lr${lr}_rg${rg}_wsi_s2021"
        python eval.py --k 5 --models_exp_code $model_code --save_exp_code ${model_code}_pl1_best --task task_1_tumor_vs_normal --model_type abmil --data_root_dir ./data_feat --root_sub_dir Camelyon16_patch256_ostu_res50_pl1_wsi --splits_dir /home/mcs001/20181133/CLAM/splits/task_camelyon16_wsi
    done
done
