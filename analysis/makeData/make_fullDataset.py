import pandas as pd
import sys
import numpy as np
seg_x = sys.argv[1]
seg_y = sys.argv[2]
seg_z = sys.argv[3]

# Load the two files
df_proton = pd.read_csv(f'../../dataset/results_{seg_x}_{seg_y}_{seg_z}/final_proton.tsv', sep='\t')
df_pion = pd.read_csv(f'../../dataset/results_{seg_x}_{seg_y}_{seg_z}/final_pion.tsv', sep='\t')

# Remove the first two columns
df_proton = df_proton.iloc[:, 2:]
df_pion = df_pion.iloc[:, 2:]

# Add the 'Class' column: 0 for final_proton, 1 for final_pion
df_proton['Class'] = 0
df_pion['Class'] = 1

# Combine the two dataframes
final_result = pd.concat([df_proton, df_pion], ignore_index=True)

# Save the result to a new file
final_result.to_csv(f'../../dataset/results_{seg_x}_{seg_y}_{seg_z}/final_combined.tsv', sep='\t', index=False)

# Funzione per verificare NaN, Empty e Inf
def check_for_nan_empty_inf(df):
    # Verifica NaN (valori mancanti)
    nan_values = df.isna().sum()  # Conta i NaN per colonna
    print("NaN values per colonna:")
    print(nan_values)

    # Verifica valori vuoti (empty strings)
    empty_values = (df == "").sum()  # Conta le stringhe vuote per colonna
    print("\nEmpty values (stringhe vuote) per colonna:")
    print(empty_values)

    # Verifica valori infiniti (inf, -inf)
    inf_values = df.applymap(np.isinf).sum()  # Conta i valori infiniti per colonna
    print("\nInf values per colonna:")
    print(inf_values)

# Verifica la presenza di NaN, Empty e Inf
check_for_nan_empty_inf(final_result)

print("Combined file created: final_combined.tsv")
