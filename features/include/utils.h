#ifndef UTILS_H
#define UTILS_H

#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH2.h>
#include <TMultiGraph.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TSystemDirectory.h>
#include <TTree.h>
#include <algorithm>
#include <cmath>
#include <dirent.h>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

using namespace std;

// Utils
std::string returnFilePath(std::string particleName);
std::vector<int> convertPos(int cub_idx, int cell_idx, std::vector<int> size);

// Angle and Aplanarity
std::vector<int> findClosePoints(const std::vector<std::vector<double>> &points,
                                 const std::vector<double> &referencePoint, double threshold);
std::vector<int> topEnergy_index(const std::vector<double> &arr);
std::vector<std::vector<double>> calculate_distance_and_angle(std::vector<std::vector<double>> &v);
double angleBetweenLineAndPlane(const std::vector<std::vector<double>> &vectors);

// Statistical Functions
double computeMean(const std::vector<double> &data);
double computeVariance(const std::vector<double> &data);

// Find Peak
int findPeak(const std::vector<double> &vec, double threshold = 60.0);

// Spatial Observables
std::vector<double> computeMeanFullStats(const std::vector<int> &x, const std::vector<double> &weights,
                                         double cell_size);
std::vector<double> computeRadius(const std::vector<int> &x, const std::vector<int> &y,
                                  const std::vector<double> &energy, double x_center, double y_center, bool isWeighted);
std::vector<double> computeLongitudinalShower(const std::vector<int> &z, const std::vector<double> &energy,
                                              double z_center, bool isWeighted);


#endif // UTILS_H
