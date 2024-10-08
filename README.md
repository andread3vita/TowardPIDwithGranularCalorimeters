# TowardPIDwithGranularCalorimeters

This study investigates whether high-granularity hadronic calorimeters can differentiate between protons, charged pions, and kaons by analyzing detailed energy deposition patterns, with promising preliminary results from Geant4 simulations
![Screenshot](images/calorimeterRepresentation.png)
## Machine Learning Strategy

Our work proposes the use of XGBoost Boosted Decision Trees (BDTs) to analyse descriptive features for each event. The approach includes a preprocessing step that generates variables for each event, which are then input into the machine learning algorithm.
Hyperparameter optimization is conducted using GridSearch, exploring different configurations, including the choice of booster and tree method type.

## Meaningful Shower Features

To study particle interactions, identifying the primary interaction vertex is crucial, as it reveals key information about the particle. Detector segmentation, particularly longitudinal, enables detailed analysis of the shower's energy profile. A moving window algorithm helps locate the primary vertex near an energy peak. Further studies can focus on the energy around the vertex, its relationship to secondary vertices, and shower dimensions, including average size and asymmetries from non-interacting secondary particles.


# How to compile
```
cd features/src
g++ -Ofast -g <file>.cc utils.cc `root-config --cflags` -o <OutputName> -L$(root-config --libdir) -Wl,-rpath,$(root-config --libdir) -lCore -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTVecOps -pthread -lm -ldl -lSpectrum
```
