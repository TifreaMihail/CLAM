import json
import argparse
import pickle
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import subprocess
import os
import numpy as np

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists
        return super(NumpyEncoder, self).default(obj)

# Function to parse the configuration file
def load_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

# Parse arguments
parser = argparse.ArgumentParser(description="Bayesian Optimization Controller")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
args = parser.parse_args()

# Load the configuration
config = load_config(args.config)

MAIN_SCRIPT = config["MAIN_SCRIPT"]
RESULTS_DIR = config["RESULTS_DIR"]
SEED = config["SEED"]
K = config["K"]
DEFAULT_ARGS = config["DEFAULT_ARGS"]
HYPERPARAMETERS = config["HYPERPARAMETERS"]

# Build the search space for Bayesian optimization
search_space = []
for param, details in HYPERPARAMETERS.items():
    param_type = details["type"]
    if param_type == "log-uniform":
        search_space.append(Real(details["min"], details["max"], "log-uniform", name=param))
    elif param_type == "integer":
        search_space.append(Integer(details["min"], details["max"], name=param))
    elif param_type == "categorical":
        search_space.append(Categorical(details["values"], name=param))

# Function to run a single experiment
def run_experiment(params):
    # Extract hyperparameter values
    param_dict = {param.name: value for param, value in zip(search_space, params)}

    # Map hyperparameter names to main.py arguments
    mapped_params = {
        "learning_rate": "lr",
        "regularization": "reg",
        # Add other mappings if necessary
    }

    # Generate a unique experiment identifier
    exp_code = "_".join([f"{key}{value}" for key, value in param_dict.items()]).replace(".", "_")
    experiment_dir = os.path.join(RESULTS_DIR, f"{exp_code}_s{SEED}")

    # Create experiment directory if it doesn't exist
    os.makedirs(experiment_dir, exist_ok=True)

    # Construct the command to run main.py
    cmd = ["python", MAIN_SCRIPT]

    # Add hyperparameters to the command, mapping names as necessary
    for param, value in param_dict.items():
        mapped_param = mapped_params.get(param, param)  # Use mapped name if available
        if value == 1:  # Boolean flags
            cmd.append(f"--{mapped_param}")
        elif value == 0:  # Skip flags if 0
            continue
        else:
            cmd.extend([f"--{mapped_param}", str(value)])

    # Add default arguments to the command
    for key, value in DEFAULT_ARGS.items():
        if isinstance(value, bool):  # Handle boolean flags
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    # Add experiment-specific arguments
    cmd.extend([
        f"--exp_code", exp_code,
        f"--results_dir", RESULTS_DIR,
        f"--seed", str(SEED),
        f"--k", str(K)
    ])

    # Save the full configuration to a summary file
    config_summary = {
        "hyperparameters": param_dict,
        "default_args": DEFAULT_ARGS,
        "seed": SEED,
        "exp_code": exp_code
    }
    config_summary_file = os.path.join(experiment_dir, "config_summary.json")
    with open(config_summary_file, "w") as f:
        json.dump(config_summary, f, indent=4, cls=NumpyEncoder)

    print(f"Running experiment with command: {' '.join(cmd)}")

    # Run the command and capture the output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Print the standard output and error for debugging
        print(f"Standard Output:\n{result.stdout}")
        print(f"Standard Error:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        # Print the error details
        print(f"Experiment {exp_code} failed with error:")
        print(f"Standard Output:\n{e.stdout}")
        print(f"Standard Error:\n{e.stderr}")
        return 9999.0  # High penalty for failed runs

    # Read the validation loss from output_metric.txt
    val_loss_file = os.path.join(experiment_dir, "output_metric.txt")
    if os.path.exists(val_loss_file):
        with open(val_loss_file, "r") as f:
            val_loss = float(f.read().strip())
        print(f"Experiment {exp_code} completed with validation loss: {val_loss}")
        return val_loss
    else:
        print(f"Validation loss file not found for {val_loss_file}.")
        return 9999.0  # High penalty for missing results

# Check if a previous checkpoint exists
checkpoint_file = os.path.join(RESULTS_DIR, "bayesian_optimization_checkpoint.pkl")
if os.path.exists(checkpoint_file):
    print(f"Resuming from checkpoint: {checkpoint_file}")
    with open(checkpoint_file, "rb") as f:
        previous_result = pickle.load(f)
    # Resume optimization
    result = gp_minimize(
        run_experiment,
        search_space,
        n_calls=30,  # Add more experiments
        x0=previous_result.x_iters,  # Previously tested configurations
        y0=previous_result.func_vals,  # Corresponding losses
        random_state=42,
        verbose=True
    )
else:
    print("No checkpoint found. Starting a new optimization.")
    # Start fresh optimization
    result = gp_minimize(
        run_experiment,
        search_space,
        n_calls=30,
        random_state=42,
        verbose=True
    )

# Save results to a file
results_save_file = os.path.join(RESULTS_DIR, "bayesian_optimization_checkpoint.pkl")
with open(results_save_file, "wb") as f:
    pickle.dump(result, f)

print(f"Optimization checkpoint saved to {results_save_file}")

# Report the best results
print("\nOptimization completed.")
print(f"Best parameters found: {result.x}")
print(f"Best validation loss: {result.fun}")

# Save results to file
results_summary_file = os.path.join(RESULTS_DIR, "bayesian_optimization_results.txt")
with open(results_summary_file, "w") as f:
    f.write("Best Parameters:\n")
    for param, value in zip(search_space, result.x):
        f.write(f"{param.name}: {value}\n")
    f.write(f"Best Validation Loss: {result.fun}\n")

print(f"Results saved to {results_summary_file}")
