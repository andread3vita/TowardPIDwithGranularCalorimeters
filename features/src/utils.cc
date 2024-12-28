#include <iostream>
#include <algorithm>
#include <TFile.h>
#include <TTree.h>
#include <string>
#include <vector>
#include <utility> 
#include <TSystem.h>
#include <TSystemDirectory.h>
#include <TH2.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TChain.h>
#include <sys/types.h>
#include <dirent.h>
#include <TGraph.h>
#include <TMultiGraph.h>
#include <dirent.h>

#include <numeric>
#include <random>
#include "/lustre/cmswork/aabhishe/TowardPIDwithGranularCalorimeters/features/include/utils.h"


using namespace std;

///////////////////// utils //////////////////////////
std::string returnFilePath(std::string particleName)
{
    std::string dirPath;
    if (particleName == "kaon")
    {
        dirPath = "/lustre/cmsdata/optCalData/kaon/";
    }
    else if(particleName == "pion")
    {
        dirPath = "/lustre/cmsdata/optCalData/pion/";

    }
    else if(particleName=="proton")
    {
    
        dirPath = "/lustre/cmsdata/optCalData/proton/";
    }
    else
    {
    
        std::cerr << "Invalid particle name!!"<<std::endl;
        std::cerr << "Valid particles: kaon , pion , proton " <<std::endl;
    }

    return dirPath;

}

std::vector<int> convertPos(int cub_idx, int cell_idx, std::vector<int> size){

    int layer_cublet_idx = cub_idx%100;

    int x_cub = layer_cublet_idx % 10;
    int y_cub = 9-layer_cublet_idx/10;
    int z_cub = cub_idx/100;

    int layer_cell_idx = cell_idx%100;

    int x_cell_temp = layer_cell_idx% 10;
    int y_cell_temp = 9-layer_cell_idx/10;
    int z_cell_temp = cell_idx/100;

    // Convert the 1D index into 3D coordinates (x, y, z)
    int x_cell = x_cub*10 + x_cell_temp;
    int y_cell = y_cub*10 + y_cell_temp;
    int z_cell = z_cub*10 + z_cell_temp;  

    // Scale the coordinates to the new grid size
    int new_x = x_cell / (100 / size[0]);
    int new_y = y_cell / (100 / size[1]);
    int new_z = z_cell / (100 / size[2]);

    return {new_x,new_y,new_z};

}

int convert_to_index(int cub_idx, int cell_idx, std::vector<int> size){

    int layer_cublet_idx = cub_idx%100;

    int x_cub = layer_cublet_idx % 10;
    int y_cub = 9-layer_cublet_idx/10;
    int z_cub = cub_idx/100;

    int layer_cell_idx = cell_idx%100;

    int x_cell_temp = layer_cell_idx% 10;
    int y_cell_temp = 9-layer_cell_idx/10;
    int z_cell_temp = cell_idx/100;

    // Convert the 1D index into 3D coordinates (x, y, z)
    int x_cell = x_cub*10 + x_cell_temp;
    int y_cell = y_cub*10 + y_cell_temp;
    int z_cell = z_cub*10 + z_cell_temp;  

    // Scale the coordinates to the new grid size
    int new_x = x_cell / (100 / size[0]);
    int new_y = y_cell / (100 / size[1]);
    int new_z = z_cell / (100 / size[2]);

    int index = new_x + size[1]*new_y + size[0]*size[1]*new_z;
    return index;

}

std::vector<int> index_to_pos(int index, std::vector<int> size){

    int z_layer = index / (size[0]*size[1]);

    int plane_index =  index % (size[0]*size[1]);

    int x_layer = (plane_index % size[0]);
    int y_layer = (size[1] - 1) - (plane_index / size[0]);

    return {x_layer,y_layer,z_layer};

}

double computeMean(const std::vector<double>& data) {
    if (data.empty()) {
        return 0.0;  // Return 0 if the vector is empty
    }
    // Accumulate the sum of elements and divide by the number of elements
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double computeVariance(const std::vector<double>& data) {
    if (data.empty()) {
        return 0.0;  // Return 0 if the vector is empty
    }
    
    double mean = computeMean(data);
    double variance = 0.0;

    // Compute the sum of squared differences from the mean
    for (const auto& value : data) {
        variance += std::pow(value - mean, 2);
    }

    // Divide by the number of elements to get the variance
    return variance / data.size();
}

std::vector<double> computeMeanFullStats(const std::vector<int>& x, const std::vector<double>& weights,double cell_size)
{

    double mean_w = 0.;
    double mean_p = 0.;
    double norm = 0.;
    for (size_t i = 0; i < x.size(); ++i)
    {
        double x_cell = x[i]*cell_size;

        mean_w += x_cell*weights[i];
        mean_p += x_cell;
        norm += weights[i];

    }

    mean_w /= norm;
    mean_p /= x.size();

    vector<double> out = {mean_w,mean_p};

    return out;
}

///////////////////// findPeak //////////////////////////
// Function to find the index of the first peak in a vector, based on a threshold
int findPeak(const std::vector<double> &vec, double threshold = 60.0)
{

    int size = vec.size() - 1; // Get the size of the vector minus one for safe access

    // First pass: Find the peak based on the initial threshold
    for (int i = 0; i < size; ++i)
    {
        if (vec[i] > threshold)
        {
            return i; // Return the index if the current value exceeds the threshold
        }
        // Check if the difference between consecutive elements exceeds the threshold
        else if (vec[i + 1] - vec[i] > threshold)
        {
            return i + 1; // Return the index of the next element
        }
    }

    // Second pass: Gradually decrease the threshold and repeat the search
    for (int j = 1; j < 5; ++j)
    {
        double thr = threshold - (j * 0.1) * threshold; // Reduce threshold by 10% each time

        for (int i = 0; i < size; ++i)
        {
            if (vec[i] > thr)
            {
                return i; // Return the index if the value exceeds the reduced threshold
            }
            else if (vec[i + 1] - vec[i] > thr)
            {
                return i + 1; // Return the index of the next element if the difference exceeds the reduced threshold
            }
        }
    }

    return 99; // Return 99 if no peak is found
}

///////////////////// smearing //////////////////////////
double smearing_time(double min_time_0,double delta_t)
{
    
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<double> gauss(min_time_0, delta_t);
    double time = gauss(gen);

    return time;
}

std::vector<double> shift_time(std::string particleName, double dist, std::string smear)
{
    double mass = 0.;
    
    std::string partNames[] = {"proton","pion","kaon"};
    double massList[] = {938.272, 139.570, 493.677};

    for (size_t i = 0; i< 3; i++)
    {   
        std::string particle = partNames[i];
        if (particleName == particle)
        {
            mass = massList[i];
        }
    }

    double c = 299792458;
    double energy = 1e5;
    double p = sqrt(pow(energy,2) - pow(mass,2));

    double v = (p/energy)*c;
    double t = (dist/v)*1e12;

    if (smear == "y")
    {   
        double t_3 = smearing_time(t,40)+(0.4/v)*1e12;
        double t_2_6 = smearing_time(t,40);
        return {t_2_6,t_3};
    }
    else
    {   
        double t_3 = t+(0.4/v)*1e12;
        double t_2_6 = t;
        return {t_2_6,t_3};
    }
    
}

///////////////////// spatial Observables //////////////////////////
std::vector<double> computeRadius(  const std::vector<int> &x, 
                                    const std::vector<int> &y,
                                    const std::vector<double> &energy, 
                                    double x_center, 
                                    double y_center, 
                                    bool isWeighted)
{
    double cell_size_x = 3.; // mm
    double cell_size_y = 3.; // mm

    double R = 0.;            // Average radius
    double sigma = 0.;        // Standard deviation
    double total_energy = 0.; // Total energy deposited

    // Loop over all positions
    for (size_t i = 0; i < x.size(); ++i)
    {
        // Calculate the difference from the center
        double dx = (x[i] - x_center);
        double dy = (y[i] - y_center);

        // Calculate the radius using Euclidean distance
        double radius = sqrt(pow(dx, 2) + pow(dy, 2));

        // Accumulate radius and squared radius for weighted or unweighted calculations
        if (isWeighted)
        {
            R += energy[i] * radius;              // Weighted sum of radii
            sigma += energy[i] * radius * radius; // Weighted sum of squared radii
        }
        else
        {
            R += radius;              // Simple sum of radii
            sigma += radius * radius; // Sum of squared radii
        }

        total_energy += energy[i]; // Accumulate total energy
    }

    // Compute average and standard deviation
    if (isWeighted)
    {
        R /= total_energy;                              // Average radius for weighted calculation
        sigma = sqrt(sigma / total_energy - pow(R, 2)); // Standard deviation
    }
    else
    {
        R /= x.size();                              // Average radius for unweighted calculation
        sigma = sqrt(sigma / x.size() - pow(R, 2)); // Standard deviation
    }

    vector<double> out = {R, sigma}; // Return results as a vector
    return out;
}

std::vector<double> computeLongitudinalShower(  const std::vector<int> &z, 
                                                const std::vector<double> &energy,
                                                double z_center, bool isWeighted)
{
    double cell_size_z = 12.; // mm

    double Z_0 = 0.;          // Average Z position
    double sigma = 0.;        // Standard deviation
    double total_energy = 0.; // Total energy deposited

    // Loop over all positions
    for (size_t i = 0; i < z.size(); ++i)
    {
        // Calculate the absolute difference from the center
        double dz = (std::abs(z[i] - z_center));

        // Accumulate for weighted or unweighted calculations
        if (isWeighted)
        {
            double z0 = dz * energy[i];   // Weighted contribution to Z_0
            sigma += dz * dz * energy[i]; // Weighted sum of squared distances

            Z_0 += z0; // Accumulate weighted Z_0
        }
        else
        {
            Z_0 += dz;        // Simple sum for Z_0
            sigma += dz * dz; // Sum of squared distances
        }

        total_energy += energy[i]; // Accumulate total energy
    }

    // Compute average and standard deviation
    if (isWeighted)
    {
        Z_0 /= total_energy;                              // Average Z for weighted calculation
        sigma = sqrt(sigma / total_energy - pow(Z_0, 2)); // Standard deviation
    }
    else
    {
        Z_0 /= z.size();                              // Average Z for unweighted calculation
        sigma = sqrt(sigma / z.size() - pow(Z_0, 2)); // Standard deviation
    }

    vector<double> out = {Z_0, sigma}; // Return results as a vector
    return out;
}

///////////////////// angle and aplanarity //////////////////////////
std::vector<int> findClosePoints(const std::vector<std::vector<double>>& points, 
                                 const std::vector<double>& referencePoint, 
                                 double threshold) {
    
    std::vector<int> result;
    
    // Itera su ogni elemento del vettore di vettori
    for (size_t i = 0; i < points.size(); ++i) {
        const std::vector<double>& point = points[i];
        
        // Calcola la distanza euclidea
        double distance = std::sqrt(std::pow(point[0] - referencePoint[0], 2) +
                                    std::pow(point[1] - referencePoint[1], 2) +
                                    std::pow(point[2] - referencePoint[2], 2));
        
        // Verifica se la distanza è inferiore alla soglia
        if (distance < threshold) {
            result.push_back(i);
        }
    }
    
    return result;
}

// Funzione che ritorna un vettore di indici degli elementi di arr in ordine decrescente
std::vector<int> topEnergy_index(const std::vector<double>& arr) {
    
    // Crea un vettore di indici [0, 1, 2, ..., n-1]
    std::vector<int> indices(arr.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Ordina gli indici in base ai valori corrispondenti nel vettore arr in ordine decrescente
    std::sort(indices.begin(), indices.end(),
              [&arr](int i1, int i2) { return arr[i1] > arr[i2]; });

    return indices;
}

std::vector<std::vector<double>> calculate_distance_and_angle(std::vector<std::vector<double>>& v) {
    
    std::vector<std::vector<double>> out;
    std::vector<double> first = v[0];

    for (size_t i = 0; i < v.size(); ++i) {
        double dist = std::sqrt(std::pow(first[0] - v[i][0], 2) + 
                                std::pow(first[1] - v[i][1], 2) + 
                                std::pow(first[2] - v[i][2], 2));

        if (dist>0)
        {
            double dot = first[0] * v[i][0] + first[1] * v[i][1] + first[2] * v[i][2];
            
            double normA = std::sqrt(first[0] * first[0] + first[1] * first[1] + first[2] * first[2]);
            double normB = std::sqrt(v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);

            double angle_rad = std::acos(dot / (normA * normB));
            double angle_deg = angle_rad * 180.0 / M_PI;  
            
            out.push_back({dist,angle_deg});
        }
        else
        {
            out.push_back({dist,dist});
        }
    }

    return out;
}

double angleBetweenLineAndPlane(const std::vector<std::vector<double>>& vectors) {
    
    // Define a vector of points, starting with a fixed point
    std::vector<std::vector<double>> points;
    points.push_back({0,0,-0.5});

    // // Initialize variables to find the point with minimum z-coordinate
    // double min_z = 0.;
    // int ind_min = 1;

    // Loop through the first three vectors to compute shifted points
    for (size_t g = 0; g < 3; g++) {
        std::vector<double> temp(3,0.);
        
        // Shift coordinates relative to (49.5, 49.5, 0.5)
        temp[0] = vectors[g][0] - 49.5;
        temp[1] = vectors[g][1] - 49.5;
        temp[2] = vectors[g][2] + 0.5;

        // // Find the point with the lowest z-coordinate
        // if (temp[2] < min_z) {
        //     min_z = temp[2];
        //     ind_min = g + 1;
        // }

        // Add the computed point to the points vector
        points.push_back(temp);
    }

    // Define the line using the first point and the one with minimum z
    std::vector<std::vector<double>> linePoints = { points[0], points[1] };

    // Calculate the direction vector of the line
    std::vector<double> d = {
        linePoints[1][0] - linePoints[0][0],
        linePoints[1][1] - linePoints[0][1],
        linePoints[1][2] - linePoints[0][2]
    };

    // Use the next three points to define the plane
    std::vector<std::vector<double>> planePoints = { points[1], points[2], points[3] };

    // Compute vectors u and v in the plane
    std::vector<double> u = {
        planePoints[1][0] - planePoints[0][0],
        planePoints[1][1] - planePoints[0][1],
        planePoints[1][2] - planePoints[0][2]
    };

    std::vector<double> v = {
        planePoints[2][0] - planePoints[0][0],
        planePoints[2][1] - planePoints[0][1],
        planePoints[2][2] - planePoints[0][2]
    };

    // Calculate the normal vector to the plane using the cross product of u and v
    std::vector<double> n = {
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    };

    // Calculate the norms (magnitudes) of the direction vector d and the normal vector n
    double normD = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    double normN = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

    // Calculate the dot product between the direction vector d and the normal vector n
    double dotDN = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];

    if (dotDN == 0 && (normD == 0 || normN == 0))
    {
        return 0.;
    }

    // Calculate the angle (in radians) between the line and the plane
    double cosAlpha = dotDN / (normD * normN);
    double alpha = std::acos(dotDN / (normD * normN));

    // Return the angle in degrees, converting from radians and adjusting to get the correct angle
    return std::abs(90.0 - (alpha * 180.0 / M_PI)); 
}

