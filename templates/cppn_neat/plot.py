from typing import List, Tuple
import json
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_dir)

from visualize import save_step_robot_image

ENV_NAME = "Carrier-v0"
STEP_SAVE_IMAGE = 100
SEED = 42

def get_max_fitness_and_best_robot_id(output_lines: List[str]) -> Tuple[float, int]:
    """Get the max fitness and the ID of the best robot from the lines of the output.txt file."""

    array_fitnesses = np.asarray([float(line.strip().split()[1]) for line in output_lines if line.strip()])
    index_max_fitness = array_fitnesses.argmax()
    max_fitness = array_fitnesses[index_max_fitness]
    best_id = int(output_lines[index_max_fitness].strip().split()[0])
    return max_fitness, best_id


# LOAD FINAL RESULTS:
folders = list(Path('./').iterdir())
results_info = {}
for folder in folders:
    if folder.name.startswith("run") and folder.is_dir():
        run_info = {}
        
        # Extract max fitness values for each generation
        generations = []
        max_fitness_vals = []
        
        # Read output.txt files from each generation
        exp_folder = folder / "test_cppn"
        gen_folders = sorted([f for f in exp_folder.iterdir() if f.name.startswith("generation")])
        
        for gen_folder in gen_folders:
            gen_num = int(gen_folder.name.split("_")[1])
            output_file = gen_folder / "output.txt"
            
            with open(output_file, "r") as f:
                lines = f.readlines()
                if lines:  # Skip empty files
                    # Get max fitness from this generation
                    max_fitness, _ = get_max_fitness_and_best_robot_id(lines)
                    generations.append(gen_num)
                    max_fitness_vals.append(max_fitness)
        
        run_info["generations"] = generations
        run_info["max_fitness"] = max_fitness_vals
        
        results_info[folder.name] = run_info

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "CPPN-NEAT",
}

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Line plot of max fitness across runs
print("Plotting max fitness across generations...")
plt.figure(figsize=(10, 6))
for i, run in enumerate(runs):
    generations = results_info[run]["generations"]
    max_fitness = results_info[run]["max_fitness"]
    
    plt.plot(generations, max_fitness, label=labels[run], color=colors[i], marker='o')

plt.title("Maximum Fitness Across Generations")
plt.xlabel("Generation")
plt.ylabel("Maximum Fitness")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig("max_fitness_plot.png")
plt.close()

# Plot 2: Line plot of best fitness so far
print("Plotting best fitness so far...")
plt.figure(figsize=(10, 6))
for i, run in enumerate(runs):
    generations = results_info[run]["generations"]
    max_fitness = results_info[run]["max_fitness"]
    
    # Calculate cumulative maximum
    best_so_far = np.maximum.accumulate(max_fitness)
    
    plt.plot(generations, best_so_far, label=labels[run], color=colors[i], marker='o')

plt.title("Best Fitness So Far Across Generations")
plt.xlabel("Generation")
plt.ylabel("Best Fitness Achieved")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig("best_fitness_so_far_plot.png")
plt.close()

# Images: Show best robot structure with the highest fitness across ALL generations
print("Saving best robot images...")
for run in runs:
    print(f"Processing {run}...")
    
    # Find the generation with the highest fitness
    max_fitness_vals = results_info[run]["max_fitness"]
    best_gen_idx = np.argmax(max_fitness_vals)
    best_gen = results_info[run]["generations"][best_gen_idx]
    
    # Get path to the best generation's output file
    gen_path = Path(f"{run}/test_cppn/generation_{best_gen}")
    output_file = gen_path / "output.txt"
    
    # Read output file to find best performing individual's ID
    with open(output_file, "r") as f:
        _, best_id = get_max_fitness_and_best_robot_id(f.readlines())
    
    # Construct paths for best individual
    body_path = str(gen_path / "structure" / f"{best_id}.npz")
    ctrl_path = str(gen_path / "controller" / f"{best_id}.zip")
    img = save_step_robot_image(ENV_NAME, body_path, ctrl_path, seed=SEED, step=-1)
    imageio.imsave(f"{run}_best_robot.png", img)  # Changed filename to indicate it's the best robot

    # Print information about the best robot
    print(f"Best robot in {run} found in generation {best_gen} with fitness {max_fitness_vals[best_gen_idx]}")
