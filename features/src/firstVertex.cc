#include "TCanvas.h"
#include "TF2.h"
#include "TH2.h"
#include "TMath.h"
#include "TROOT.h"
#include "TRandom.h"
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH2.h>
#include <TMultiGraph.h>
#include <TROOT.h>
#include <TSpectrum.h>
#include <TSystem.h>
#include <TSystemDirectory.h>
#include <TTree.h>
#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../include/utils.h" // Custom utility functions

using namespace std;

int Tentries; // Number of entries

// Declare vectors for cublet indices, cell indices, time, energy deposit, particle ID, and deltaE
vector<int> *Tcublet_idx;
vector<int> *Tcell_idx;
vector<double> *Ttime;
vector<double> *Tedep;
vector<double> *Tpdg;
vector<double> *Tdeltae;

// Define Key as a pair of integers and Value as a pair of doubles
using Key = std::pair<int, int>;
using Value = std::pair<double, double>;

// Define DataMap as an unordered map where the key is of type Key and the value is of type Value
using DataMap = std::unordered_map<Key, Value>;

// Parameters for the window size in x, y, and z directions
// int xyWindowSize_param = 2;  //  2 if 100 100 100 and 1 if 50 50 100
// int zWindowSize_param = 1;      //  1 if 100 100 100 and 1 if 50 50 100

// int close_vertex_radius_param = 5; // Maximum distance to consider a cell close to the primary vertex (5 if 100 100 100 and 3 if 50 50 100)

// Define macros for cursor movement and line clearing in terminal
#define CURSOR_TO_START "\033[1G"
#define CLEAR_LINE "\033[K"

// Directory creation function for both Windows and POSIX-compliant systems
int createDirectory(const std::string &path)
{
    #ifdef _WIN32
        return mkdir(path.c_str()); // Use _mkdir for Windows
    #else
        return mkdir(path.c_str(), 0777); // Use POSIX mkdir for Unix-like systems
    #endif
}

// Specialize the std namespace to hash a pair of integers
namespace std
{
    template <> struct hash<std::pair<int, int>>
    {
        // Custom hash function for std::pair<int, int>
        std::size_t operator()(const std::pair<int, int> &p) const
        {
            // Combine the hash values of the two integers using XOR and bit shifting
            return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
        }
    };
}

// Function to calculate the Euclidean distance between two 3D positions
double distance(std::vector<int> ref, std::vector<int> pos)
{
    // Calculate the Euclidean distance between the reference and target points
    double dist = sqrt(pow(ref[0] - pos[0], 2) + pow(ref[1] - pos[1], 2) + pow(ref[2] - pos[2], 2));
    return dist;
}

std::vector<double> vertexAnalysis(TString filePath,int eventNum, std::vector<int> size_cell,double threshold, int close_vertex_radius_param, int xyWindowSize_param, int zWindowSize_param)
{

    // Open the input file using ROOT's TFile class
    TFile* inputFile = TFile::Open(filePath); 

    // Retrieve the TTree from the input file by dynamically casting to TTree type
    TTree* Tree = dynamic_cast<TTree*>(inputFile->Get("outputTree"));

    // Set up the branches of the tree to access specific variables
    Tree->SetBranchAddress("Tinteractions_in_event", &Tentries);  // Total number of interactions in the event
    Tree->SetBranchAddress("Tcublet_idx", &Tcublet_idx);  // Cublet index in the 3D grid
    Tree->SetBranchAddress("Tcell_idx", &Tcell_idx);      // Cell index within the cublet
    Tree->SetBranchAddress("Tedep", &Tedep);              // Energy deposited in the cell
    Tree->SetBranchAddress("Tglob_t", &Ttime);            // Global timestamp of the event

    // Read the data for a specific event using the event number
    Tree->GetEntry(eventNum);

    // Define variables for storing the energy deposition data
    DataMap energyMap;  // Map to store energy and time for each cublet-cell pair
    double totalEnergy = 0.0;  // Total energy deposited in the event
    std::vector<double> en_position(size_cell[2], 0);  // Energy deposition per z-slice
    std::vector<std::pair<double, double>> timAndEnergy;  // List of time and energy pairs for each interaction
    double centralTowerFraction_cell = 0.0;  // Energy fraction in the central tower
    double time0 = 10e5;

    // Define parameters for the spatial window of the analysis
    int xyWindowSize = xyWindowSize_param;  // XY window size for the analysis
    int zWindowSize = zWindowSize_param;    // Z window size for the analysis
    int minWin = ((size_cell[0]-1)/2) - xyWindowSize;  // Minimum window boundary in XY plane
    int maxWin = ((size_cell[0]-1)/2) + xyWindowSize;  // Maximum window boundary in XY plane

    // Loop over each interaction in the event
    for (size_t j = 0; j < Tentries; j++) 
    {
        // Get the cublet and cell indices, energy, and time for the current interaction
        int cub_idx = (*Tcublet_idx)[j];
        int cell_idx = (*Tcell_idx)[j];
        double E = (*Tedep)[j];  // Energy deposited in the current cell
        double time = (*Ttime)[j];  // Timestamp of the current interaction

        // Convert the cublet and cell indices to a 3D position
        std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);
        
        // Create a key (cublet, cell) to store in the energy map
        Key key = std::make_pair(cub_idx, cell_idx);

        // Accumulate the total energy for the event and compute time0
        totalEnergy += E;
        
        if(time < time0)
        {
            time0 = time;
        }

        // Update the energy and time in the energy map
        if (energyMap.find(key) != energyMap.end()) {
            // If the key already exists, add the energy and update the time if it's more recent
            energyMap[key].first += E;
            if (energyMap[key].second < time) {
                energyMap[key].second = time;
            }
        } else {
            // If the key does not exist, create a new entry with the current energy and time
            energyMap[key] = std::make_pair(E, time);
        }

        // Check if the interaction is within the XY window range
        if((int_pos[0] >= minWin && int_pos[0] <= maxWin) && 
        (int_pos[1] >= minWin && int_pos[1] <= maxWin))
        {
            // Accumulate energy in the corresponding z-slice
            en_position[int_pos[2]] += E;
        }

        // Check if the interaction is in the central tower (specific x and y condition)
        if((int_pos[0] == (size_cell[0] / 2)) && 
        (int_pos[1] == (size_cell[1] / 2 - 1)))
        {
            // Accumulate the energy contribution to the central tower fraction
            centralTowerFraction_cell += E;
        }

        // Store the time and energy for the current interaction
        timAndEnergy.push_back(std::make_pair(time, E));
    }

    ////////////////////////////////////////////////////////////////
    ////////// VERTEX POSITION + PostVertexEnergyFraction //////////
    ////////////////////////////////////////////////////////////////

    //// Z_VERTEX
    // Initialize the smoothed energy positions with the original energy values
    vector<double> smooth_energy_pos = en_position;

    // Apply smoothing if the Z-window size is greater than 1
    if (zWindowSize > 1)
    {
        // Get the first value of en_position
        double first_val = en_position[0];

        // Pad the beginning of en_position with (zWindowSize-1) copies of the first value
        for (int k = 0; k < (zWindowSize - 1); ++k)
        {
            en_position.insert(en_position.begin(), first_val);  // Insert first_val at the beginning
        }

        // Perform moving average smoothing
        double sum = 0.0;
        // Loop through en_position and calculate the moving average over the zWindowSize window
        for (int i = 0; i < (en_position.size() - zWindowSize + 1); ++i) {
            // Calculate the sum of elements in the current window and divide by the window size
            sum = (std::accumulate(en_position.begin() + i, en_position.begin() + i + zWindowSize, 0.0)) / zWindowSize;
            
            // Update smooth_energy_pos with the smoothed value
            smooth_energy_pos[i] = sum;
        }
    }

    // Find the z-vertex (peak) in the smoothed energy profile using the specified threshold
    int z_vertex = findPeak(smooth_energy_pos, threshold);


    //// X_VERTEX and Y_VERTEX + PostVertexEnergyFraction
    // Vector to store tuples of time and (x, y) coordinates of interactions at the z_vertex
    std::vector<std::tuple<double, std::pair<int, int>>> inter_timings;

    // Variable to store the fraction of energy deposited after the z_vertex
    double PostVertexEnergyFraction = 0.0;

    // Loop through all interactions in the event
    for (size_t j = 0; j < Tentries; j++) {

        // Get the cublet and cell indices, time, and energy for the current interaction
        int cub_idx = (*Tcublet_idx)[j];
        int cell_idx = (*Tcell_idx)[j];
        double time = (*Ttime)[j];
        double E = (*Tedep)[j];

        // Convert cublet and cell indices to a 3D position
        std::vector<int> int_pos = convertPos(cub_idx, cell_idx,size_cell);
        
        // Check if the interaction occurs at the z_vertex
        if (int_pos[2] == z_vertex)
        {
            // Store the time and the (x, y) coordinates in the inter_timings vector
            inter_timings.emplace_back(time, std::make_pair(int_pos[0], int_pos[1]));
        }

        // Accumulate energy if the interaction occurs at or after the z_vertex
        if (int_pos[2] >= z_vertex)
        {
            PostVertexEnergyFraction += E;
        }
    }

    // Find the (x, y) pair with the earliest time at the z_vertex
    std::pair<int, int> min_pair;  // Will store the (x, y) coordinates with the minimum time
    double min_time = 10e5;  // Initialize min_time with a large value

    // Loop through all the entries in inter_timings to find the minimum time
    for (const auto& elem : inter_timings) {
        double value;
        std::pair<int, int> pair;

        // Extract the time and (x, y) pair from the tuple
        std::tie(value, pair) = elem;

        // Update min_time and min_pair if the current time is smaller
        if (value < min_time) {
            min_time = value;
            min_pair = pair;
        }
    }

    // Extract the (x, y) coordinates corresponding to the minimum time
    int x_peak = min_pair.first;
    int y_peak = min_pair.second;

    // Store the (x, y, z) coordinates of the vertex
    std::vector<int> vertex_pos = {x_peak, y_peak, z_vertex};

    ////////////////////////////////////////////////////////////////
    ////////// FIRST AND SECOND MAX + numCellBeforeVertex //////////    
    ////////////////////////////////////////////////////////////////

    // Find the maximum and second maximum value of E (energy)
    Key maxKey, secondMaxKey;  // Variables to store the keys corresponding to max and second max energy
    double maxE = -1.0, secondMaxE = -1.0;  // Initialize max and second max energies to -1
    int numCellBeforeVertex = 0;  // Counter for the number of cells before the vertex (z_vertex)

    // Loop through each entry in the energyMap
    for (const auto& entry : energyMap) {
        double E = entry.second.first;  // Retrieve the stored energy for the current entry

        // Check if the current energy is greater than the maximum energy found so far
        if (E > maxE) {
            // Before updating the maximum, update the second maximum with the previous max
            secondMaxE = maxE;
            secondMaxKey = maxKey;

            // Update the maximum energy and its corresponding key
            maxE = E;
            maxKey = entry.first;
        } else if (E > secondMaxE) {
            // If the current energy is not the max but greater than the second max, update second max
            secondMaxE = E;
            secondMaxKey = entry.first;
        }
            
        // Convert the (cublet, cell) indices into a 3D position
        std::vector<int> pos = convertPos(entry.first.first, entry.first.second,size_cell);
        
        // Check if the position's z-coordinate is before the z_vertex
        if (pos[2] < z_vertex) {
            ++numCellBeforeVertex;  // Increment the counter if the cell is before the vertex
        }
    }


    //////////////////////////////
    ////////// TIME 50% //////////    
    //////////////////////////////

    // Sort the timAndEnergy vector based on the time (the first element of each pair)
    // The lambda function compares the first element (time) of two pairs
    sort(timAndEnergy.begin(), timAndEnergy.end(), [](const pair<double, double>& a, const pair<double, double>& b) {
        return a.first < b.first;
    });

    // Initialize cumulative sum to track the running total of energy
    double cumulativeSum = 0.0;

    // Loop through each pair in timAndEnergy
    for (auto& pair : timAndEnergy) {
        cumulativeSum += pair.second;  // Add the energy (second element) to the cumulative sum
        pair.second = cumulativeSum;   // Replace the energy with the cumulative sum (for cumulative distribution)
    }

    // Normalize the cumulative sums by dividing each one by the total cumulative sum
    for (auto& pair : timAndEnergy) {
        pair.second /= cumulativeSum;  // Normalize each cumulative energy value by the total energy
    }

    // Find the time corresponding to when the cumulative energy exceeds 50% (median energy)
    double time50 = 0.0;
    for (const auto& pair : timAndEnergy) {
        if (pair.second > 0.5) {  // Check if the normalized cumulative energy exceeds 50%
            time50 = pair.first;  // Store the corresponding time (first element of the pair)
            break;  // Stop once the 50% mark is reached
        }
    }

    ////////////////////////////////////////////////////////
    ////////// energyCloseVertex + maxCloseVertex //////////    
    ////////////////////////////////////////////////////////

    // Initialize variables to track energy close to the vertex and the maximum energy near the vertex
    double energyCloseVertex = 0;
    double maxCloseVertex = 0;

    // Loop through each entry in the energyMap, which stores energy for each (cublet, cell) pair
    for (const auto& entry : energyMap) {

        // Convert the (cublet, cell) indices into a 3D position
        std::vector<int> pos = convertPos(entry.first.first, entry.first.second,size_cell);

        // Check if the distance between the current position and the vertex position is within a radius of 5 units
        if (distance(vertex_pos, pos) <= close_vertex_radius_param)
        {
            double energy = entry.second.first;  // Retrieve the stored energy for the current entry
            
            // Accumulate energy for positions close to the vertex
            energyCloseVertex += energy;

            // Update the maximum energy near the vertex
            if (energy > maxCloseVertex)
            {
                maxCloseVertex = energy;
            }
        }
    }

    // Calculate the fraction of total energy that is close to the vertex
    double EnergyFractionCloseToVertex = energyCloseVertex / totalEnergy;


    ///////////////////////////////////////////////
    ////////// energyVarianceCloseVertex //////////    
    ///////////////////////////////////////////////

    // Initialize a vector to store energy values for slices near the vertex
    vector<double> slice_energy;

    // Loop through each entry in the energyMap, which stores energy for each (cublet, cell) pair
    for (const auto& entry : energyMap) {

        // Retrieve the cublet and cell indices
        int cub_idx = entry.first.first;
        int cell_idx = entry.first.second;

        // Convert the cublet and cell indices into a 3D position
        std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);

        // Check if the current position is within a window near the vertex in the xy-plane and within a z-range of 2 units
        if ((abs(int_pos[0] - vertex_pos[0]) < xyWindowSize) && 
            (abs(int_pos[1] - vertex_pos[1]) < xyWindowSize) && 
            abs(int_pos[2] - vertex_pos[2]) < 2)
        {
            // Push the energy value into the slice_energy vector
            double E = entry.second.first;
            slice_energy.push_back(E); 
        }
    }
    double en_variance = computeVariance(slice_energy);
                
    // z_vertex
    double VertexTime =  min_time-time0;
    // PostVertexEnergyFraction = PostVertexEnergyFraction/totalEnergy;
    // numCellBeforeVertex
    double deltaT = (time50-time0) - VertexTime;
    double TotalEnergyCloseToVertex = energyCloseVertex;
    // EnergyFractionCloseToVertex
    // maxCloseVertex
    double VarianceAtVertex = en_variance;
    std::vector<int> maxCoord = convertPos(maxKey.first,maxKey.second,size_cell); double distanceMaxFromVertex = distance(maxCoord,vertex_pos);
    centralTowerFraction_cell = centralTowerFraction_cell/totalEnergy;

    inputFile->Close();
    delete inputFile;
    
    std::vector<double> out(11,0);

    out[0] = z_vertex*1.0;
    out[1] =VertexTime;
    out[2] =PostVertexEnergyFraction/totalEnergy;
    out[3] =numCellBeforeVertex;
    out[4] =deltaT;
    out[5] =TotalEnergyCloseToVertex;
    out[6] = EnergyFractionCloseToVertex;
    out[7] =maxCloseVertex;
    out[8] =VarianceAtVertex;
    out[9] =distanceMaxFromVertex;
    out[10] =centralTowerFraction_cell;

    return out;
}

void fillTable(std::string particleName,std::vector<int> size_cell = {100,100,100},double threshold=125.,int close_vertex_radius_param = 5, int xyWindowSize_param = 2, int zWindowSize_param = 1)
{

    std::string outFile = "./results_" + std::to_string(size_cell[0]) + "_" + std::to_string(size_cell[1]) + "_" + std::to_string(size_cell[2]) + "__" + std::to_string(int(threshold)) + "/" + particleName + ".tsv";
    std::ofstream oFile(outFile, std::ios::out);

    oFile << "FileName\t";
    oFile << "EventNum\t";

    oFile << "Z_vertex\t";
    oFile << "VertexTime\t";

    oFile << "PostVertexEnergyFraction\t";
    oFile << "numCellBeforeVertex\t";
    
    oFile << "DeltaT\t";
    
    oFile << "TotalEnergyCloseToVertex\t";
    oFile << "EnergyFractionCloseToVertex\t";
    oFile << "MaxEnergyCloseVertex\t";
     
    oFile << "VarianceAtVertex\t";

    oFile << "distanceMaxFromVertex\t";

    oFile << "centralTowerFraction_cell";

    oFile << std::endl;

    std::string dirPath = returnFilePath(particleName);

    TSystemDirectory dir(dirPath.c_str(), dirPath.c_str());
    TList *filesList = dir.GetListOfFiles();

    int totEv = 0;

    if (filesList) {
        TSystemFile *file;
        TString fileName;
        TIter next(filesList);

        auto start = std::chrono::high_resolution_clock::now();

        while ((file=(TSystemFile*)next())) {

            if (!file->IsDirectory()) {
                
                TString fileName = dirPath + file->GetName();
                for (size_t i=0; i<1000;i++)
                {
                    
                    totEv += 1;

                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> duration = end - start;

                    std::cout << "Processing: " << file->GetName() <<"\tEvent: " << i << "\t\ttime[min]: " << (duration.count()/1000)/60 << "\t\tProgress: " << totEv/50e3*100 << "%" << std::flush;
                    
                    std::vector<double> info = vertexAnalysis(fileName,i,size_cell,threshold,close_vertex_radius_param, xyWindowSize_param, zWindowSize_param);
                    oFile << file->GetName() << "\t" << i << "\t" << info[0] << "\t" << info[1] <<  "\t" << info[2] <<  "\t" << info[3] << "\t" << info[4] <<  "\t" << info[5] << "\t" << info[6] << "\t" << info[7] <<  "\t" << info[8] << "\t" << info[9] << "\t" << info[10] << std::endl;

                    std::cout << CURSOR_TO_START << CLEAR_LINE;
                                
                }

            }
        }                   
    }

    oFile.close();
}

int main(int argc, char* argv[]) {
    
    // Check if the correct number of arguments is provided
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " <particle> <size_x> <size_y> <size_z> <threshold> <closeVertex> <xyWindow> <zWindow>" << std::endl;
        return 1;
    }


    // Retrieve and store the particle type from the first argument
    std::string particle = argv[1];
    
    // Convert the second, third, and fourth arguments to integers for the grid size
    int size_x = std::stoi(argv[2]);
    int size_y = std::stoi(argv[3]);
    int size_z = std::stoi(argv[4]);
    
    // Convert the fifth argument to a floating-point number (threshold)
    double threshold = std::stod(argv[5]);
    int rad = std::stoi(argv[6]);

    int xyWin = std::stoi(argv[7]);
    int zWin = std::stoi(argv[8]);

    // Check if the folder exists and create it if it doesn't
    std::string folderPath = "./results_" + std::to_string(size_x) + "_" + std::to_string(size_y) + "_" + std::to_string(size_z)+"__" + std::to_string(int(threshold));
    // Try to create the folder
    if (createDirectory(folderPath) == 0) {
        std::cout << "Directory created successfully: " << folderPath << std::endl;
    } else {
        std::cout << "Directory already exists or couldn't be created: " << folderPath << std::endl;
    }

    // Call a function that fills the table with the given particle type
    fillTable(particle,{size_x,size_y,size_z},threshold,rad, xyWin , zWin);

    return 0;
}
