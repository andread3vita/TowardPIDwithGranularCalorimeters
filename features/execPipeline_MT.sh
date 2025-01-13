#!/bin/bash

# Ensure the "log" directory exists
mkdir -p log

# Directory for .args files
args_dir="args"

echo "Do you want to compile? (y/N):"
read compile_

if [ "$compile_" == "y" ]; then
    echo "Starting compilation..."
    
    for file in src/*.cc; do
        # Ignora il file utils.cc
        if [[ "$file" == "src/utils.cc" ]]; then
            continue
        elif [[ "$file" == "src/topPeaks_NOTSTABLE.cc" ]]; then
            continue
        elif [[ "$file" == "src/fileManager.cc" ]]; then
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
fi

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

        # Start in a new screen session for "proton"
        screen -dmS "${exe_name}_proton" bash -c "
            source ~/cmsenv_activate.sh;
            $exe proton $segmentation_args $args > log/${exe_name}_${segmentation}_proton.log 2>&1
        "
        echo "Started $exe with 'proton' in a screen session (${exe_name}_proton)."

        # Start in a new screen session for "pion"
        screen -dmS "${exe_name}_pion" bash -c "
            source ~/cmsenv_activate.sh;
            $exe pion $segmentation_args $args > log/${exe_name}_${segmentation}_pion.log 2>&1
        "
        echo "Started $exe with 'pion' in a screen session (${exe_name}_pion)."

    else
        # Start in a new screen session for "proton" without arguments
        screen -dmS "${exe_name}_proton" bash -c "
            source ~/cmsenv_activate.sh;
            $exe proton $segmentation_args > log/${exe_name}_${segmentation}_proton.log 2>&1
        "
        echo "Started $exe with 'proton' in a screen session (${exe_name}_proton)."

        # Start in a new screen session for "pion" without arguments
        screen -dmS "${exe_name}_pion" bash -c "
            source ~/cmsenv_activate.sh;
            $exe pion $segmentation_args > log/${exe_name}_${segmentation}_pion.log 2>&1
        "
        echo "Started $exe with 'pion' in a screen session (${exe_name}_pion)."
    fi
done

echo "All processes have been started in separate screen sessions."
