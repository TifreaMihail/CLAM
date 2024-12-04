#!/bin/bash

#SBATCH --job-name=train_abmil_CAM16_lr_1e-3
#SBATCH --output=/home/mcs001/20181133/CLAM/job_logs/gabmil/WINDOW_gabmil_lr1e-5%j_both.txt
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=4-12:00:00
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

# Regularizations and window sizes to test
# window_sizes=(2 4 16 64)
window_sizes=(1)
regularizations=("1e-2" "1e-3" "1e-4" "1e-5")

for use_norm in True False; do
  for use_skip in True False; do
    for rg in "${regularizations[@]}"; do
      for ws in "${window_sizes[@]}"; do
        lr_code=1e-5
        rg_code=${rg//./}
        ws_code=${ws}

        # Set options and codes based on use_norm
        if [ "$use_norm" = "True" ]; then
          use_norm_option="--use_weight_norm"
          norm_code="norm"
        else
          use_norm_option=""
          norm_code="nonorm"
        fi

        # Set options and codes based on use_skip
        if [ "$use_skip" = "True" ]; then
          use_skip_option="--use_skip"
          skip_code="skip"
        else
          use_skip_option=""
          skip_code="noskip"
        fi

        exp_code="camelyon16_gabmil_50_lr${lr_code}_rg${rg_code}_ws${ws_code}_${norm_code}_${skip_code}"
        results_subdir="/home/mcs001/20181133/CLAM/results/TRAIN/BOTH/${norm_code}_${skip_code}"

        
        # Create the results directory if it doesn't exist
        mkdir -p "$results_subdir"

        # Run the experiment
        python /home/mcs001/20181133/CLAM/main.py \
          --drop_out 0.25 \
          --lr 1e-5 \
          --reg $rg \
          --window_size $ws \
          --k 1 \
          --exp_code $exp_code \
          --weighted_sample \
          --bag_loss ce  \
          $use_norm_option \
          $use_skip_option \
          --use_grid \
          --use_block \
          --task task_1_tumor_vs_normal \
          --model_type gabmil \
          --data_root_dir /home/mcs001/20181133/CLAM/data_feat \
          --root_sub_dir Camelyon16_patch256_ostu_res50_pl1_wsi \
          --results_dir $results_subdir \
          --split_dir /home/mcs001/20181133/CLAM/splits/task_camelyon16_wsi \
          --embed_dim 1024
      done
    done
  done
done
