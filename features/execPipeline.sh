#!/bin/bash

# Ensure the "log" directory exists
mkdir -p log

# Directory for .args files
args_dir="args"

#!/bin/bash

for file in src/*.cc; do
    # Ignora il file utils.cc
    if [[ "$file" == "src/utils.cc" ]]; then
        continue
    fi

    # Extract the file name without extension
    filename=$(basename "$file" .cc)

    # Compile the file into an executable
    g++ -std=c++11 -Ofast -g "$file" ./src/utils.cc `root-config --cflags` -o "$filename" \
    -L$(root-config --libdir) -Wl,-rpath,$(root-config --libdir) \
    -lCore -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint \
    -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc \
    -lROOTVecOps -pthread -lm -ldl -lSpectrum
    echo "Compiled $file -> $filename"
done


# Define executables
declare -a executables=(
    "./firstVertex"
    "./generalFeature"
    "./missingEnergy"
    "./spatialObservables"
    "./speed"
    "./topPeaks"
)

# Prompt for segmentation input
echo "Enter the segmentation (format x_y_z):"
read segmentation

if [[ -z "$segmentation" ]]; then
    echo "No segmentation provided. Exiting."
    exit 1
fi

# Loop through all .args files in the args/ directory
nd_count=0
for file in args/*.args; do
    if grep -q "^${segmentation}\s" "$file"; then
        # Check if there are columns with ND
        if grep -E "^${segmentation}\s.*\sND" "$file" > /dev/null; then
            echo "The file '$file' contains ND for segmentation $segmentation."
            nd_count=$((nd_count + 1))
        fi
    fi
done

# Check if any file contained ND
if [ $nd_count -gt 0 ]; then
    exit 0
fi

# Convert segmentation underscores to spaces
segmentation_args=$(echo "$segmentation" | tr '_' ' ')

# Iterate over each executable
for exe in "${executables[@]}"; do
    # Remove any "./" from the executable name
    exe_name=$(basename "$exe")
    
    # Argument file in the args directory
    args_file="${args_dir}/${exe_name}.args"
    
    # Check if the argument file exists
    if [[ -f $args_file ]]; then
        
        # Search for the line corresponding to the given segmentation
        args=$(grep -P "^$segmentation\t" "$args_file" | awk -F'\t' '{for (i=2; i<=NF; i++) printf $i " ";}')

        # If no matching line is found, skip this executable
        if [[ -z "$args" ]]; then
            echo "No matching segmentation '$segmentation' found in $args_file. Skipping $exe."
            continue
        fi

        # Execute the command for "proton"
        log_name_proton="log/${exe_name}_${segmentation}_proton.log"
        nohup $exe proton $segmentation_args $args > "$log_name_proton" 2>&1 &
        echo "Started $exe with 'proton', segmentation: '$segmentation_args', and arguments from $args_file (log: $log_name_proton)"
        
        # Execute the command for "pion"
        log_name_pion="log/${exe_name}_${segmentation}_pion.log"
        nohup $exe pion $segmentation_args $args > "$log_name_pion" 2>&1 &
        echo "Started $exe with 'pion', segmentation: '$segmentation_args', and arguments from $args_file (log: $log_name_pion)"
    else
        # Execute the command for "proton" without additional arguments
        log_name_proton="log/${exe_name}_${segmentation}_proton.log"
        nohup $exe proton $segmentation_args > "$log_name_proton" 2>&1 &
        echo "Started $exe with 'proton' and segmentation: '$segmentation_args' (log: $log_name_proton)"
        
        # Execute the command for "pion" without additional arguments
        log_name_pion="log/${exe_name}_${segmentation}_pion.log"
        nohup $exe pion $segmentation_args > "$log_name_pion" 2>&1 &
        echo "Started $exe with 'pion' and segmentation: '$segmentation_args' (log: $log_name_pion)"
    fi
done

echo "All available processes have been started in the background. Check the 'log' directory for log files."
