import subprocess
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="Run commands to generate datasets")
parser.add_argument('x', type=int, help="Value for x")
parser.add_argument('y', type=int, help="Value for y")
parser.add_argument('z', type=int, help="Value for z")
parser.add_argument('time_type', type=str, help="Value for time_type")

# Parse the arguments
args = parser.parse_args()

# Get the values of x, y, z from the arguments
x = args.x
y = args.y
z = args.z
time_type = args.time_type

# Build the commands to execute
command1 = f"python make_dataset_particle.py {x} {y} {z} {time_type} proton"
command2 = f"python make_dataset_particle.py {x} {y} {z} {time_type} pion"
command3 = f"python make_dataset_particle.py {x} {y} {z} {time_type} kaon"
command4 = f"python make_fullDataset.py {x} {y} {z} {time_type}"

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
run_command(command4)

print("Command execution completed.")

