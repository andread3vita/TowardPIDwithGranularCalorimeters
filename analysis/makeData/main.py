import subprocess
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="Run commands to generate datasets")
parser.add_argument('x', type=int, help="Value for x")
parser.add_argument('y', type=int, help="Value for y")
parser.add_argument('z', type=int, help="Value for z")

# Parse the arguments
args = parser.parse_args()

# Get the values of x, y, z from the arguments
x = args.x
y = args.y
z = args.z

# Build the commands to execute
command1 = f"python make_dataset_particle.py {x} {y} {z} proton"
command2 = f"python make_dataset_particle.py {x} {y} {z} pion"
command3 = f"python make_fullDataset.py {x} {y} {z}"

# Function to run commands
def run_command(command):
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error while running the command: {e}")
        print(e.stdout)
        print(e.stderr)

# Run the commands in order
run_command(command1)
run_command(command2)
run_command(command3)

print("Command execution completed.")

