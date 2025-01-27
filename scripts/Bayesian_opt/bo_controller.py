from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import subprocess
import time
import os

# Define your search space
search_space = [
    Real(1e-5, 1e-2, "log-uniform", name="learning_rate"),  # Learning rate
    Real(1e-6, 1e-4, "log-uniform", name="regularization"),  # Regularization
    Integer(1, 100, name="window_size"),  # Window size
    Categorical([0, 1], name="use_grid"),  # Use grid flag (0 = False, 1 = True)
    Categorical([0, 1], name="use_block")  # Use block flag (0 = False, 1 = True)
]

# Path configurations
RESULTS_DIR = "/home/mcs001/20181133/CLAM/results/hyperopt"
JOB_SCRIPT = "/home/mcs001/20181133/CLAM/scripts/Bayesian_opt/train_gabmil.sh"

# Function to run a single experiment
def run_experiment(params):
    learning_rate, regularization, window_size, use_grid, use_block = params

    # Create a unique experiment ID
    exp_id = f"lr{learning_rate:.1e}_reg{regularization:.1e}_ws{window_size}_grid{use_grid}_block{use_block}".replace(".", "_")
    result_file = os.path.join(RESULTS_DIR, f"{exp_id}_result.txt")

    # Check if result file already exists to skip re-running the experiment
    if os.path.exists(result_file):
        print(f"Result already exists for {exp_id}, skipping...")
        with open(result_file, "r") as f:
            return float(f.read().strip())

    # Submit the job using sbatch
    try:
        SEED = 2021  # Fixed seed value (you can randomize this per job if needed)
        print(f"Submitting job for {exp_id} with seed {SEED}")
        

        # Submit the job using sbatch
        sbatch_cmd = [
            "sbatch", JOB_SCRIPT,
            str(learning_rate),
            str(regularization),
            str(window_size),
            str(use_grid),
            str(use_block),
            exp_id,
            result_file,
            str(SEED)
        ]

        output = subprocess.check_output(sbatch_cmd).decode("utf-8")
        print(f"Job submitted: {output.strip()}")
    except Exception as e:
        print(f"Job submission failed for {exp_id}: {e}")
        return 9999.0  # Return a high penalty value for failed submission

    # Wait for the result file to appear
    while not os.path.exists(result_file):
        print(f"Waiting for job {exp_id} to complete...")
        time.sleep(60)  # Check every 60 seconds

    # Read the validation loss from the result file
    try:
        with open(result_file, "r") as f:
            result = float(f.read().strip())
        print(f"Job {exp_id} completed with result: {result}")
    except Exception as e:
        print(f"Failed to read result for {exp_id}: {e}")
        return 9999.0  # Penalty value for unreadable results

    return result

# Run Bayesian optimization
print("Starting Bayesian Optimization...")
result = gp_minimize(
    run_experiment,
    search_space,
    n_calls=20,  # Number of function evaluations
    random_state=42,
    verbose=True
)

# Report the results
print("\nOptimization completed.")
print(f"Best parameters found: {result.x}")
print(f"Best validation loss: {result.fun}")

# Save the results
output_file = os.path.join(RESULTS_DIR, "bayesian_optimization_results.txt")
with open(output_file, "w") as f:
    f.write("Best Parameters:\n")
    f.write(f"Learning Rate: {result.x[0]}\n")
    f.write(f"Regularization: {result.x[1]}\n")
    f.write(f"Window Size: {result.x[2]}\n")
    f.write(f"Use Grid: {result.x[3]}\n")
    f.write(f"Use Block: {result.x[4]}\n")
    f.write(f"Best Validation Loss: {result.fun}\n")

print(f"Results saved to {output_file}")
