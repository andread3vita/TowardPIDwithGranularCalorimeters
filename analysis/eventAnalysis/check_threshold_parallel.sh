#!/bin/bash

# Set fixed parameters
particle="proton"
size_x=50
size_y=50
size_z=25
closeVertex=2
xyWindow=1
zWindow=1
smearing="y"

thresholds=(200 220 250 300 350 380)

first_script="/lustre/cmswork/aabhishe/TowardPIDwithGranularCalorimeters/features/firstVertex"

# Loop through each potential threshold values 
for threshold in "${thresholds[@]}"; do
    # Create a new screen session for each threshold
    screen -dmS threshold_$threshold bash -c "
        source ~/homeui/aabhishe/cmsenv_activate.sh;
        $first_script $particle $size_x $size_y $size_z $threshold $closeVertex $xyWindow $zWindow $smearing
    "
    echo "Started screen session for threshold $threshold"
done
