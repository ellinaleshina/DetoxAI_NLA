import subprocess
from tqdm.auto import tqdm

# Define the noise levels to test
noise_levels = [1.3, 1.7, 2.5, 5, 7] 

# Define the base command
base_command = "python evaluation/edit_model.py --config_file gpt2-medium.ini"
log_folder = "logs"
# Create a logs directory if it doesn't exist
import os
os.makedirs(log_folder, exist_ok=True)

# Loop through noise levels with a progress bar
for noise_level in tqdm(noise_levels, desc="Running commands", unit="config"):
    # Prepare the full command with the current noise level
    command = f"noise_level={noise_level} {base_command}"
    
    # Define the log file name
    log_file = f"{log_folder}/noise_level_{noise_level}.log"
    
    # Run the command and save output to the log file
    with open(log_file, "w") as log:
        tqdm.write(f"Running: {command} -> Logging to: {log_file}")
        result = subprocess.run(command, shell=True, stdout=log, stderr=log)

        # Check return code and report errors if any
        if result.returncode != 0:
            tqdm.write(f"[Error] Command failed for noise_scale={noise_level}. Check {log_file} for details.")

print(f"All runs completed. Logs are saved in the {log_folder} directory.")
