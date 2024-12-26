#!/bin/bash

for file in src/*.cc; do
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
