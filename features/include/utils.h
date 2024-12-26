#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <numeric>

std::string returnFilePath(std::string particleName);

std::vector<int> convertPos(int cub_idx, int cell_idx, std::vector<int> size = {100,100,100});

int convert_to_index(int cub_idx, int cell_idx, std::vector<int> size);

std::vector<int> index_to_pos(int index, std::vector<int> size);

double computeMean(const std::vector<double>& data);

double computeVariance(const std::vector<double>& data);

int findPeak(const std::vector<double>& vec, double threshold);

std::vector<double> shift_time(std::string particleName, double distance,std::string smear);

double smearing_time(double min_time_0,double delta_t);

std::vector<double> computeMeanFullStats(const std::vector<int>& x, const std::vector<double>& weights,double cell_size);

std::vector<double> computeRadius(const std::vector<int>& x, const std::vector<int>& y,const std::vector<double>& energy, double x_center, double y_center, bool isWeighted);

std::vector<double> computeLongitudinalShower(const std::vector<int>& z, const std::vector<double>& energy, double z_center,bool isWeighted);

std::vector<int> findClosePoints(const std::vector<std::vector<double>>& points, 
                                 const std::vector<double>& referencePoint, 
                                 double threshold);

std::vector<int> topEnergy_index(const std::vector<double>& arr);

std::vector<std::vector<double>> calculate_distance_and_angle(std::vector<std::vector<double>>& v);

double angleBetweenLineAndPlane(const std::vector<std::vector<double>>& vectors);

#endif // UTILS_H