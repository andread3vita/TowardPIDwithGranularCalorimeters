import pandas as pd
import sys

primary_folder = '../../dataset/'

seg_x = sys.argv[1]
seg_y = sys.argv[2]
seg_z = sys.argv[3]
time_type = sys.argv[4]
particle = sys.argv[5]

segmentation = f'results_{seg_x}_{seg_y}_{seg_z}/'

subdirectories = []

if time_type == "d":
    subdirectories = [
        'firstVertex/Digitalization',
        'generalFeatures/Digitalization',
        'missingEnergy',
        'spatialObservables',
        'speed/Digitalization',
        'topPeaks'
    ]
else:
    subdirectories = [
        'firstVertex/Smearing',
        'generalFeatures/Smearing',
        'missingEnergy',
        'spatialObservables',
        'speed/Smearing',
        'topPeaks'
    ]
    
# Construct the paths to the subdirectories
paths = [f"{primary_folder}{segmentation}{subdir}" for subdir in subdirectories]

common_cols = None
combined_dataset = pd.DataFrame()

# Process each subdirectory
for subdir in paths:
    file_path = f"{subdir}/{particle}.tsv"
    try:
        df = pd.read_csv(file_path, sep='\t',index_col=None)

        if df.shape[1] >= 2:
            # Extract specific columns (after the first two)
            specific_cols = df.iloc[:, 2:]

            # For the first file, save the common columns (first two) and combine with specific columns
            if common_cols is None:
                common_cols = df.iloc[:, :2]
                combined = pd.concat([common_cols, specific_cols], axis=1)
                combined_dataset = pd.concat([combined_dataset, combined], axis=1)
            else:
                # For subsequent files, only add specific columns
                combined_dataset = pd.concat([combined_dataset, specific_cols], axis=1)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Remove any columns with 'Unnamed' in their name
combined_dataset = combined_dataset.loc[:, ~combined_dataset.columns.str.contains('^Unnamed:', na=False)]

# Save the final combined dataset
if time_type == "d":
    combined_dataset.to_csv(f"{primary_folder}{segmentation}final_{particle}_digi.tsv", sep='\t', index=False)
else: 
    combined_dataset.to_csv(f"{primary_folder}{segmentation}final_{particle}_smear.tsv", sep='\t', index=False)
print(f"Combined file created: final_{particle}.tsv")
