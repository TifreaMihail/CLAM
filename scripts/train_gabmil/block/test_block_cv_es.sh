#!/bin/bash

#SBATCH --job-name=test_gabmil_CAM16_BLOCK
#SBATCH --output=/home/mcs001/20181133/CLAM/job_logs/gabmil/BLOCK_gabmil_test.txt
#SBATCH --partition=tue.gpu.q
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

# Define the top hyperparameter configurations

# ABLATION STUDY (ws1 and layer norm)
# declare -A top_configs
# top_configs[1]="1e-3 1e-4 2 False True"
# top_configs[2]="1e-3 1e-5 4 True True"
# top_configs[3]="1e-4 1e-5 4 False True"
# top_configs[4]="1e-3 1e-3 2 False True"
# top_configs[5]="1e-3 1e-3 1 False True"

# Hyperparameter tuning
declare -A top_configs
top_configs[1]="1e-3 1e-4 2 False True"
top_configs[2]="1e-4 1e-5 4 False True"
top_configs[3]="1e-3 1e-3 2 False True"
top_configs[4]="1e-4 1e-3 4 False True"
top_configs[5]="1e-3 1e-5 4 False True"

# Iterate through each top configuration
for i in {1..5}; do
  # Extract the hyperparameters
  IFS=' ' read -r lr reg window_size norm skip <<< "${top_configs[$i]}"
  
  # Set up directory and experiment codes
  lr_code=${lr//./}
  rg_code=${reg//./}
  ws_code=${window_size}
  
  # Set options based on normalization and skip connections
  if [ "$norm" = "True" ]; then
    use_norm_option="--use_weight_norm"
    norm_code="norm"
  else
    use_norm_option=""
    norm_code="nonorm"
  fi

  if [ "$skip" = "True" ]; then
    use_skip_option="--use_skip"
    skip_code="skip"
  else
    use_skip_option=""
    skip_code="noskip"
  fi

  # Define the experiment code and results directory
  exp_code="camelyon16_gabmil_50_lr${lr_code}_rg${rg_code}_ws${ws_code}_${norm_code}_${skip_code}"
  results_subdir="/home/mcs001/20181133/CLAM/results/TEST_CV_ES/BLOCK/${norm_code}_${skip_code}"

  # Create the results directory if it doesn't exist
  mkdir -p "$results_subdir"

  # Run the experiment with 5-fold cross-validation
  python /home/mcs001/20181133/CLAM/main.py \
    --drop_out 0.25 \
    --lr $lr \
    --reg $reg \
    --window_size $window_size \
    --k 5 \
    --exp_code $exp_code \
    --weighted_sample \
    --bag_loss ce \
    --early_stopping \
    $use_norm_option \
    $use_skip_option \
    --use_block \
    --task task_1_tumor_vs_normal \
    --model_type gabmil \
    --data_root_dir /home/mcs001/20181133/CLAM/data_feat \
    --root_sub_dir Camelyon16_patch256_ostu_res50_pl1_wsi \
    --results_dir "$results_subdir" \
    --split_dir /home/mcs001/20181133/CLAM/splits/task_camelyon16_CV_sametest \
    --embed_dim 1024
done
