#!/bin/bash

#SBATCH --job-name=train_abmil_CAM16_lr_1e-3
#SBATCH --output=/home/mcs001/20181133/CLAM/job_logs/gabmil/WINDOW_gabmil_lr1e-3%j_skip_norm.txt
#SBATCH --partition=tue.gpu.q
#SBATCH --time=2-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1

source /cm/shared/apps/Anaconda/2021.11/pth3.9/etc/profile.d/conda.sh
module load CUDA/11.7.0
conda activate thesis-conda-env6

# Open WSI-finetuning project
cd /home/mcs001/20181133/CLAM/

export CUDA_VISIBLE_DEVICES=0

# Regularizations to test
rg="1e-3"
ws=2

lr_code=1e-3
rg_code=${rg//./}
ws_code=${ws}

  
exp_code="camelyon16_gabmil_50_lr${lr_code}_rg${rg_code}_ws${ws_code}_global_200_test_simple"

python /home/mcs001/20181133/CLAM/main.py \
  --drop_out 0.25 \
  --lr 1e-3 \
  --reg $rg \
  --window_size $ws \
  --k 5 \
  --exp_code $exp_code \
  --weighted_sample \
  --bag_loss ce  \
  --use_grid \
  --task task_1_tumor_vs_normal \
  --model_type gabmil \
  --data_root_dir /home/mcs001/20181133/CLAM/data_feat \
  --root_sub_dir Camelyon16_patch256_ostu_res50_pl1_wsi \
  --results_dir /home/mcs001/20181133/CLAM/results/window_gabmil_final_skip_norm \
  --split_dir /home/mcs001/20181133/CLAM/splits/task_camelyon16_wsi \
  --embed_dim 1024
