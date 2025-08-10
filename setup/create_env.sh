#!/bin/bash
#SBATCH --job-name=env_creation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=01:00:00

# Exit immediately if a command exits with a non-zero status
set -e

# Start timer
start_time=$(date +%s)

# Get the current date
current_date=$(date +%y%m%d)

# Create environment name with the current date
env_prefix=blt_$current_date

# Create the uv virtual environment
uv venv $env_prefix --python 3.12

# Activate the virtual environment
source $env_prefix/bin/activate

echo "Currently in env $(which python)"

# Install dependency groups (if using pyproject.toml)
uv pip install --group pre_build --no-build-isolation
uv pip install --group compile_xformers --no-build-isolation

# Sync all dependencies
uv sync

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"