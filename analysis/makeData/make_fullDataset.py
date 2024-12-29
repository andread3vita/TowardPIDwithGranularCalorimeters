import pandas as pd
import sys
import numpy as np

seg_x = sys.argv[1]
seg_y = sys.argv[2]
seg_z = sys.argv[3]
time_type = sys.argv[4]

# Load the two files

df_proton = pd.DataFrame()
df_pion = pd.DataFrame()
df_kaon = pd.DataFrame()

if time_type == "d":
    df_proton = pd.read_csv(f'../../dataset/results_{seg_x}_{seg_y}_{seg_z}/final_proton_digi.tsv', sep='\t')
    df_pion = pd.read_csv(f'../../dataset/results_{seg_x}_{seg_y}_{seg_z}/final_pion_digi.tsv', sep='\t')
    df_kaon = pd.read_csv(f'../../dataset/results_{seg_x}_{seg_y}_{seg_z}/final_kaon_digi.tsv', sep='\t')
else:
    df_proton = pd.read_csv(f'../../dataset/results_{seg_x}_{seg_y}_{seg_z}/final_proton_smear.tsv', sep='\t')
    df_pion = pd.read_csv(f'../../dataset/results_{seg_x}_{seg_y}_{seg_z}/final_pion_smear.tsv', sep='\t')
    df_kaon = pd.read_csv(f'../../dataset/results_{seg_x}_{seg_y}_{seg_z}/final_kaon_smear.tsv', sep='\t')

# Remove the first two columns
df_proton = df_proton.iloc[:, 2:]
df_pion = df_pion.iloc[:, 2:]
df_kaon = df_kaon.iloc[:, 2:]

# Add the 'Class' column: 0 for final_proton, 1 for final_pion
df_proton['Class'] = 0
df_pion['Class'] = 1
df_kaon['Class'] = 2

# Combine the two dataframes
final_result = pd.concat([df_proton, df_pion,df_kaon], ignore_index=True)

# Save the result to a new file
if time_type == "d":
    final_result.to_csv(f'../../dataset/results_{seg_x}_{seg_y}_{seg_z}/final_combined_digi.tsv', sep='\t', index=False)
else:
    final_result.to_csv(f'../../dataset/results_{seg_x}_{seg_y}_{seg_z}/final_combined_smear.tsv', sep='\t', index=False)

# Function to check for NaN, Empty, and Inf
def check_for_nan_empty_inf(df):
    # Check for NaN (missing values)
    nan_values = df.isna().sum()  # Count NaN per column
    print("NaN values per column:")
    print(nan_values)

    # Check for empty values (empty strings)
    empty_values = (df == "").sum()  # Count empty strings per column
    print("\nEmpty values (empty strings) per column:")
    print(empty_values)

    # Check for infinite values (inf, -inf)
    inf_values = df.applymap(np.isinf).sum()  # Count infinite values per column
    print("\nInf values per column:")
    print(inf_values)

# Check for NaN, Empty, and Inf
check_for_nan_empty_inf(final_result)

print("Combined file created: final_combined.tsv")

