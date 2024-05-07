#!/bin/bash

# Define learning rates to test
learning_rates=("1e-3" "1e-4" "1e-5")

# Create a job for each learning rate
for lr in "${learning_rates[@]}"; do
  # Create a unique script file for each learning rate
  script_file="train_lr_${lr//./}.sh"

  # Write a job script to handle this learning rate
  cat > $script_file <<EOL
#!/bin/bash

#SBATCH --job-name=train_abmil_CAM16_lr_${lr//./}
#SBATCH --output=job_logs/train_abmil_lr${lr//./}%j.txt
#SBATCH --partition=tue.gpu.q
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=1

source /cm/shared/apps/Anaconda/2021.11/pth3.9/etc/profile.d/conda.sh
module load CUDA/11.7.0
conda activate thesis-conda-env6

# Open WSI-finetuning project
cd /home/mcs001/20181133/CLAM/

export CUDA_VISIBLE_DEVICES=0

# Regularizations to test
regularizations=("1e-3" "1e-4" "1e-5" "1e-6")

for rg in "\${regularizations[@]}"; do
  lr_code=${lr//./}
  rg_code=\${rg//./}
  exp_code="camelyon16_abmil_50_lr\${lr_code}_rg\${rg_code}"

  python main.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr $lr \
    --reg \${rg} \
    --k 5 \
    --exp_code $exp_code \
    --weighted_sample \
    --bag_loss ce \
    --task task_1_tumor_vs_normal \
    --model_type abmil \
    --log_data True \
    --data_root_dir /home/mcs001/20181133/CLAM/data_feat \
    --embed_dim 1024
done
EOL

  # Submit the job script to the scheduler
  # sbatch $script_file
done
