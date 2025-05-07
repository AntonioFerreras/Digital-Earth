import taichi as ti
import numpy as np
import time
import os
import random  # Add this import for random number generation
import argparse  # Add this import for command-line argument parsing
from renderer import Renderer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate uvwz_rgb data samples.')
parser.add_argument('--sample-sunset', action='store_true', default=False,
                    help='Sample sunset scenarios (limits y values to 0.5)')
args = parser.parse_args()

# Initialize Taichi
ti.init(arch=ti.vulkan)

# Create output directory if it doesn't exist
output_dir = "uvwz_rgb_data"
os.makedirs(output_dir, exist_ok=True)

# Initialize renderer
image_res = (1920, 1080)
up = (0, 1, 0)
renderer = Renderer(image_res, up)
renderer.copy_textures()  # Make sure textures are loaded

# Parameters
num_samples = 2 ** 23  # Total number of samples to generate
batch_size = 2 ** 16   # Number of samples per file
buffer = []            # Buffer to store results before writing to file

def generate_random_filename():
    """Generate a random filename with 8 digits"""
    random_number = random.randint(0, 99999999)
    filename = f"{random_number:08d}.txt"
    return filename

def write_buffer_to_file(buffer_data):
    """Write the buffer data to a file with a random name"""
    filename = os.path.join(output_dir, generate_random_filename())
    with open(filename, 'w') as f:
        for line in buffer_data:
            f.write(line + '\n')
    print(f"Wrote {len(buffer_data)} samples to {filename}")
    return []  # Return empty buffer

# Main sampling loop
start_time = time.time()
samples_processed = 0

print(f"Starting to generate {num_samples} samples...")

# Use the command-line argument instead of hardcoded value
sample_sunset = args.sample_sunset

print(f"Sampling with {'sunset mode' if sample_sunset else 'full range'}")

for i in range(num_samples):
    # Generate random uvwz in [0,1]
    uvwz = ti.Vector([np.random.random(), np.random.random(), 
                      np.random.random(), np.random.random()])
    
    if sample_sunset:
        uvwz.y *= 0.5;

    # Call batch path trace
    rgb = renderer.batch_path_trace(
        uvwz,
        renderer.CIE_LUT_tex
    )
    
    # Convert to numpy arrays for easier handling
    uvwz_np = uvwz.to_numpy()
    rgb_np = rgb.to_numpy()

    # avoid negative rgb values
    rgb_np = np.maximum(rgb_np, 0.0)
    
    # Format the line: uvwz (space-separated) followed by rgb (space-separated)
    line = f"{uvwz_np[0]:.8f} {uvwz_np[1]:.8f} {uvwz_np[2]:.8f} {uvwz_np[3]:.8f} {rgb_np[0]:.8f} {rgb_np[1]:.8f} {rgb_np[2]:.8f}"
    buffer.append(line)
    
    # Write to file when buffer is full
    if len(buffer) >= batch_size:
        buffer = write_buffer_to_file(buffer)
    
    # Update progress
    samples_processed += 1
    if samples_processed % 100 == 0:
        elapsed_time = time.time() - start_time
        samples_per_second = samples_processed / elapsed_time
        estimated_total_time = num_samples / samples_per_second
        remaining_time = estimated_total_time - elapsed_time
        
        print(f"Progress: {samples_processed}/{num_samples} ({samples_processed/num_samples*100:.2f}%)")
        print(f"Speed: {samples_per_second:.2f} samples/second")
        print(f"Estimated time remaining: {remaining_time/60:.2f} minutes")

# Write any remaining samples
if buffer:
    write_buffer_to_file(buffer)

print(f"Completed generating {num_samples} samples in {(time.time() - start_time)/60:.2f} minutes")