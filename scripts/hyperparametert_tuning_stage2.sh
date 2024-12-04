#!/bin/bash

# Enable error handling
set -e

# List of job scripts to submit
scripts=(
    "./train_abmil/test_abmil_cv_es.sh"
    "./train_gabmil/block/test_block_cv_es.sh"
    "./train_gabmil/grid/test_grid_cv_es.sh"
    "./train_gabmil/both/test_both_cv_es.sh"
    "./train_transmil/test_transmil_cv_es.sh"
)

# Loop through each script and submit it using sbatch
for script in "${scripts[@]}"; do
    if [[ -f "$script" ]]; then
        echo "Submitting $script..."
        sbatch "$script"
    else
        echo "Warning: $script does not exist. Skipping."
    fi
done

echo "All jobs submitted!"