import os
import re

# Directory containing log files
log_dir = "logs"

# Initialize lists for the data
noise_levels = []
perplexities = []
toxicities = []
sample_responses = []

# Regular expressions to extract Perplexity, Toxicity, and Sample Response
pattern_metrics = r"gpt2 - Perplexity: ([\d\.]+), Toxicity: ([\d\.]+)"
pattern_response = r"Generation:(.+)"  # Adjust this based on actual response format in logs

# Loop through log files in the directory
for log_file in os.listdir(log_dir):
    if log_file.endswith(".log"):
        # Extract noise level from the file name
        noise_level = float(log_file.split("_")[-1].replace(".log", ""))
        file_path = os.path.join(log_dir, log_file)

        # Read the log file
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Extract metrics from the last line
        last_line = lines[-1].strip()
        match_metrics = re.search(pattern_metrics, last_line)
        if match_metrics:
            perplexity = float(match_metrics.group(1))
            toxicity = float(match_metrics.group(2))

            # Add to corresponding lists
            noise_levels.append(noise_level)
            perplexities.append(perplexity)
            toxicities.append(toxicity)

        # Extract sample response (assume it's on a specific line or marked explicitly)
        sample_response = None
        for line in lines:
            match_response = re.search(pattern_response, line)
            if match_response:
                sample_response = match_response.group(1)
                break
        sample_responses.append(sample_response if sample_response else "No response found")

# Combine data and sort by Noise Level
sorted_data = sorted(
    zip(noise_levels, perplexities, toxicities, sample_responses),
    key=lambda x: x[0]
)

# Unpack sorted data into individual lists
noise_levels, perplexities, toxicities, sample_responses = zip(*sorted_data)

# Create the structured data dictionary
data = {
    "Noise Level": list(noise_levels),
    "Perplexity": list(perplexities),
    "Toxicity": list(toxicities),
    "Sample Response": list(sample_responses),
}

# Print the resulting dictionary
# print(data)


# quit()



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

noise_levels = data["Noise Level"]
perplexities = data["Perplexity"]
toxicities = data["Toxicity"]
sample_responses = data["Sample Response"]

# Font size variable
fontsize = 25

# Create a figure with GridSpec for the plot and text
fig = plt.figure(figsize=(12, 7))  # Make the figure narrower
gs = GridSpec(1, 2, width_ratios=[2, 1.5], wspace=0.6)

# Left plot: Perplexity and Toxicity vs Noise Level
ax1 = fig.add_subplot(gs[0])

# Plot Perplexity
color1 = "tab:blue"
ax1.set_xlabel("Noise Level", fontsize=fontsize)
ax1.set_ylabel("Perplexity", color=color1, fontsize=fontsize)
ax1.plot(noise_levels, perplexities, marker="o", color=color1, label="Perplexity")
ax1.tick_params(axis="y", labelcolor=color1, labelsize=fontsize // 2)
ax1.tick_params(axis="x", labelsize=fontsize // 2)
ax1.grid(True)

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel("Toxicity", color=color2, fontsize=fontsize)
ax2.plot(noise_levels, toxicities, marker="s", color=color2, linestyle="--", label="Toxicity")
ax2.tick_params(axis="y", labelcolor=color2, labelsize=fontsize // 2)

# Add title
ax1.set_title("Perplexity and Toxicity vs. Noise Level", fontsize=20)

# Right section: Textual visualization of Sample Responses
ax3 = fig.add_subplot(gs[1])
ax3.axis("off")  # Hide the axis
ax3.text(0.5, 1, "prompt: \"Holy\"", ha="center", va="bottom", fontsize=20)

# Create a table-like display
cell_text = [[f"{noise_levels[i]:.1f}", sample_responses[i]] for i in range(len(noise_levels))]
table = ax3.table(
    cellText=cell_text,
    colLabels=["Noise Level", "Sample Response"],
    loc="center",
    cellLoc="center",
    colLoc="center",
)
for key, cell in table.get_celld().items():
    cell.set_height(0.052)  # Increase height for each cell

# Adjust table font size
table.auto_set_font_size(True)
# table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(cell_text[0]))))
plt.tight_layout()
plt.savefig('toxicity_big.png')
plt.show()
