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

#include "../include/utils.h"

using namespace std;

///////////////////// Utils //////////////////////////

// Function to return the directory path for a given particle name
std::string returnFilePath(std::string particleName)
{
    std::string dirPath;
    if (particleName == "kaon")
    {
        dirPath = "/lustre/cmsdata/optCalData/kaon/";
    }
    else if (particleName == "pion")
    {
        dirPath = "/lustre/cmsdata/optCalData/pion/";
    }
    else if (particleName == "proton")
    {
        dirPath = "/lustre/cmsdata/optCalData/proton/";
    }
    else
    {
        std::cerr << "Invalid particle name!" << std::endl;
        std::cerr << "Valid particles: kaon, pion, proton" << std::endl;
    }
    return dirPath;
}

// Function to convert 1D cell indices into scaled 3D coordinates
std::vector<int> convertPos(int cub_idx, int cell_idx, std::vector<int> size)
{
    int layer_cublet_idx = cub_idx % 100;
    int x_cub = layer_cublet_idx % 10;
    int y_cub = 9 - layer_cublet_idx / 10;
    int z_cub = cub_idx / 100;

    int layer_cell_idx = cell_idx % 100;
    int x_cell_temp = layer_cell_idx % 10;
    int y_cell_temp = 9 - layer_cell_idx / 10;
    int z_cell_temp = cell_idx / 100;

    // Convert the 1D index into 3D coordinates (x, y, z)
    int x_cell = x_cub * 10 + x_cell_temp;
    int y_cell = y_cub * 10 + y_cell_temp;
    int z_cell = z_cub * 10 + z_cell_temp;

    // Scale the coordinates to the new grid size
    int new_x = x_cell / (100 / size[0]);
    int new_y = y_cell / (100 / size[1]);
    int new_z = z_cell / (100 / size[2]);

    return {new_x, new_y, new_z};
}

///////////////////// Angle and Aplanarity //////////////////////////

// Function to find points within a certain Euclidean distance from a reference point
std::vector<int> findClosePoints(const std::vector<std::vector<double>> &points,
                                 const std::vector<double> &referencePoint, double threshold)
{

    std::vector<int> result;

    // Iterate through each point in the vector
    for (size_t i = 0; i < points.size(); ++i)
    {
        const std::vector<double> &point = points[i];

        // Calculate the Euclidean distance
        double distance =
            std::sqrt(std::pow(point[0] - referencePoint[0], 2) + std::pow(point[1] - referencePoint[1], 2) +
                      std::pow(point[2] - referencePoint[2], 2));

        // Check if the distance is within the threshold
        if (distance < threshold)
        {
            result.push_back(i);
        }
    }

    return result;
}

// Function to get indices sorted by values in descending order
std::vector<int> topEnergy_index(const std::vector<double> &arr)
{

    std::vector<int> indices(arr.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        indices[i] = i;
    }

    // Sort indices in descending order based on the corresponding values in arr
    std::sort(indices.begin(), indices.end(), [&arr](int i1, int i2) { return arr[i1] > arr[i2]; });

    return indices;
}

// Function to calculate the distance and angle for each point relative to the first point
std::vector<std::vector<double>> calculate_distance_and_angle(std::vector<std::vector<double>> &v)
{

    std::vector<std::vector<double>> out;
    std::vector<double> first = v[0];

    for (size_t i = 0; i < v.size(); ++i)
    {
        double dist = std::sqrt(std::pow(first[0] - v[i][0], 2) + std::pow(first[1] - v[i][1], 2) +
                                std::pow(first[2] - v[i][2], 2));

        if (dist > 0)
        {
            double dot = first[0] * v[i][0] + first[1] * v[i][1] + first[2] * v[i][2];
            double normA = std::sqrt(first[0] * first[0] + first[1] * first[1] + first[2] * first[2]);
            double normB = std::sqrt(v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);
            double angle_rad = std::acos(dot / (normA * normB));
            double angle_deg = angle_rad * 180.0 / M_PI;
            out.push_back({dist, angle_deg});
        }
        else
        {
            out.push_back({dist, dist});
        }
    }

    return out;
}

// Function to calculate the angle between a line and a plane
double angleBetweenLineAndPlane(const std::vector<std::vector<double>> &vectors)
{

    // Define initial points and calculate points in the plane
    std::vector<std::vector<double>> points = {{0, 0, -0.5}};
    for (size_t g = 0; g < 3; g++)
    {
        std::vector<double> temp(3, 0.);
        temp[0] = vectors[g][0] - 49.5;
        temp[1] = vectors[g][1] - 49.5;
        temp[2] = vectors[g][2] + 0.5;
        points.push_back(temp);
    }

    // Line direction vector
    std::vector<double> d = {points[1][0] - points[0][0], points[1][1] - points[0][1], points[1][2] - points[0][2]};

    // Plane normal vector calculation using cross product
    std::vector<double> u = {points[2][0] - points[1][0], points[2][1] - points[1][1], points[2][2] - points[1][2]};
    std::vector<double> v = {points[3][0] - points[1][0], points[3][1] - points[1][1], points[3][2] - points[1][2]};
    std::vector<double> n = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};

    double normD = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    double normN = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    double dotDN = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];

    if (dotDN == 0 && (normD == 0 || normN == 0))
    {
        return 0.;
    }

    double cosAlpha = dotDN / (normD * normN);
    double alpha = std::acos(dotDN / (normD * normN));

    // Return the angle between the line and plane in degrees
    return std::abs(90.0 - (alpha * 180.0 / M_PI));
}

// Function to compute the mean of a vector of doubles
double computeMean(const std::vector<double> &data)
{
    if (data.empty())
    {
        return 0.0;
    }
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

// Function to compute the variance of a vector of doubles
double computeVariance(const std::vector<double> &data)
{
    if (data.empty())
    {
        return 0.0;
    }
    double mean = computeMean(data);
    double variance = 0.0;
    for (const auto &value : data)
    {
        variance += std::pow(value - mean, 2);
    }
    return variance / data.size();
}

///////////////////// Find Peak //////////////////////////

// Function to find the index of the first peak in a vector
int findPeak(const std::vector<double> &vec, double threshold = 60.0)
{
    for (size_t i = 1; i < vec.size() - 1; ++i)
    {
        if (vec[i] > threshold && vec[i] > vec[i - 1] && vec[i] > vec[i + 1])
        {
            return i;
        }
    }
    return -1; // Return -1 if no peak is found
}

///////////////////// Spatial Observables //////////////////////////

// Function to compute the weighted mean and mean position
std::vector<double> computeMeanFullStats(const std::vector<int> &x, const std::vector<double> &weights,
                                         double cell_size)
{

    double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total_weight == 0.0)
    {
        return {0.0, 0.0};
    }

    double weighted_sum = 0.0;
    double position_sum = 0.0;

    for (size_t i = 0; i < x.size(); ++i)
    {
        weighted_sum += weights[i] * cell_size * x[i];
        position_sum += cell_size * x[i];
    }

    double weighted_mean = weighted_sum / total_weight;
    double mean_position = position_sum / x.size();
    return {weighted_mean, mean_position};
}

// Function to compute the average radius and standard deviation
std::vector<double> computeRadius(const std::vector<int> &x, const std::vector<int> &y,
                                  const std::vector<double> &energy, double x_center, double y_center, bool isWeighted)
{

    double radius_sum = 0.0;
    double radius_sq_sum = 0.0;
    double total_weight = 0.0;

    for (size_t i = 0; i < x.size(); ++i)
    {
        double radius = std::sqrt(std::pow(x[i] - x_center, 2) + std::pow(y[i] - y_center, 2));
        if (isWeighted)
        {
            radius_sum += energy[i] * radius;
            radius_sq_sum += energy[i] * std::pow(radius, 2);
            total_weight += energy[i];
        }
        else
        {
            radius_sum += radius;
            radius_sq_sum += std::pow(radius, 2);
            total_weight += 1.0;
        }
    }

    double mean_radius = radius_sum / total_weight;
    double variance_radius = (radius_sq_sum / total_weight) - std::pow(mean_radius, 2);
    double stddev_radius = std::sqrt(variance_radius);

    return {mean_radius, stddev_radius};
}

// Function to compute the longitudinal shower profile
std::vector<double> computeLongitudinalShower(const std::vector<int> &z, const std::vector<double> &energy,
                                              double z_center, bool isWeighted)
{

    double z_sum = 0.0;
    double z_sq_sum = 0.0;
    double total_weight = 0.0;

    for (size_t i = 0; i < z.size(); ++i)
    {
        double z_dist = z[i] - z_center;
        if (isWeighted)
        {
            z_sum += energy[i] * z_dist;
            z_sq_sum += energy[i] * std::pow(z_dist, 2);
            total_weight += energy[i];
        }
        else
        {
            z_sum += z_dist;
            z_sq_sum += std::pow(z_dist, 2);
            total_weight += 1.0;
        }
    }

    double mean_z = z_sum / total_weight;
    double variance_z = (z_sq_sum / total_weight) - std::pow(mean_z, 2);
    double stddev_z = std::sqrt(variance_z);

    return {mean_z, stddev_z};
}
