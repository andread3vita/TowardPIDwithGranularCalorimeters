import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Ask the user for the column name to plot the histogram
column = input("Enter the name of the column for the histogram: ")
output_file = ""

# Load the TSV file
file_path = '../../dataset/results_100_100_100/final_combined.tsv'
output_file = f'../../results/features/plots/histogram_{column}.png'

df = pd.read_csv(file_path, sep='\t')

print("Feature analysis:")
print("Number of Entries:",len(df[column]))
print("Min:",np.min(df[column]))
print("Max:",np.max(df[column]))
print("Number of non-numeric elements:",np.sum([not np.isreal(x) for x in df[column]]))

# Ask the user for the x-axis range
print("\n\nPlot configuration")
x_min = float(input("Enter the minimum x value: "))
x_max = float(input("Enter the maximum x value: "))
scale = input("Y scale: ")

unique_label, counts_label = np.unique(df[column], return_counts=True)
class_distribution = dict(zip(unique_label, counts_label))
# plt.savefig("test.png")

# Ask the user for the number of bins
num_bins = int(input("Enter the number of bins: "))


# Check if the 'Class' column exists
if 'Class' not in df.columns:
    print("Error: The 'Class' column does not exist in the dataset.")
else:
    # Check if the selected column exists in the dataset
    if column not in df.columns:
        print(f"Error: The column '{column}' does not exist in the dataset.")
    else:
        # Set up the style with elements similar to ROOT
        sns.set_context("notebook")  # Set the context size
        plt.rcParams.update({
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "lines.linewidth": 1.5,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "font.family": "serif",
            "mathtext.fontset": "stix",  # LaTeX-like font for equations
        })
        
        # Create bin edges using the same binning for all classes
        bin_edges = np.linspace(x_min, x_max, num_bins + 1)

        # Create a histogram for each class (0, 1, 2) and overlay them
        classes = df['Class'].unique()
        colors = {0: 'blue', 1: 'green', 2: 'red'}
        particle = {0: 'proton', 1: 'pion', 2: 'kaon'}

        plt.figure(figsize=(10, 6))
        for class_value in classes:
            subset = df[df['Class'] == class_value]
            sns.histplot(subset[column], label=f'{particle[class_value]}', color=colors.get(class_value), kde=False, 
                         stat="density", common_norm=False, bins=bin_edges, element="step", fill=False)

        # Add title and labels
        plt.title(f'Histogram of {column} with Class distributions', fontsize=15, fontweight='bold')
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(frameon=False, fontsize=12)

        # Set grid style similar to ROOT
        plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

        if scale == 'log':
            plt.yscale('log')

        # Set the x-axis range based on user input
        plt.xlim(x_min, x_max)

        # Save the plot as a PNG file
        plt.savefig(output_file, dpi=300)
        print(f'Plot saved as {output_file}')

        # Show the plot
        plt.show()
