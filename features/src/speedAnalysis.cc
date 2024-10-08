#include "TCanvas.h"
#include "TF2.h"
#include "TH2.h"
#include "TMath.h"
#include "TROOT.h"
#include "TRandom.h"
#include <TSpectrum.h>

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
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

#include "utils.h"
using namespace std;

int Tentries; // Number of entries in the event

// Vectors to store data from the ROOT tree
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
int xyWindowSize_param = 2;
int zWindowSize_param = 1;

// Parameter for defining a radius threshold for identifying "close start"
int close_start_radius_param = 3;

// Define terminal macros for cursor movement and line clearing
#define CURSOR_TO_START "\033[1G"
#define CLEAR_LINE "\033[K"

// Cross-platform directory creation function
#ifdef _WIN32
#include <direct.h> // Windows-specific header for directory creation
#define mkdir _mkdir
#else
#include <sys/stat.h> // POSIX-compliant systems (Linux, macOS)
#endif

// Function to create a directory
int createDirectory(const std::string &path)
{
#ifdef _WIN32
    return mkdir(path.c_str()); // Use _mkdir for Windows
#else
    return mkdir(path.c_str(), 0777); // Use POSIX mkdir for Unix-like systems
#endif
}

namespace std
{
// Specialization of std::hash for std::pair<int, int>
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

std::vector<double> speedAnalysis(TString filePath,int eventNum, std::vector<int> size_cell,double threshold)
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
    std::vector<double> en_position(size_cell[2], 0);  // Energy deposition per z-slice

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

        // Update the energy and time in the energy map
        if (energyMap.find(key) != energyMap.end()) {
            // If the key already exists, add the energy and update the time if it's more recent
            energyMap[key].first += E;
            energyMap[key].second += E*time;
            // if (energyMap[key].second < time) {
            //     energyMap[key].second = time;
            // }
        } else {
            // If the key does not exist, create a new entry with the current energy and time
            energyMap[key] = std::make_pair(E, E*time);
        }

        // Check if the interaction is within the XY window range
        if((int_pos[0] >= minWin && int_pos[0] <= maxWin) && 
        (int_pos[1] >= minWin && int_pos[1] <= maxWin))
        {
            // Accumulate energy in the corresponding z-slice
            en_position[int_pos[2]] += E;
        }
    }

    ///////////////////////////////////////////////////////////////
    ////////// VERTEX POSITION  + FIRST INTERACTION TIME //////////
    ///////////////////////////////////////////////////////////////

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

    //// X_VERTEX and Y_VERTEX
    // Vector to store tuples of time and (x, y) coordinates of interactions at the z_vertex and in the first layer
    std::vector<std::tuple<double, std::pair<int, int>>> inter_timings;
    std::vector<std::tuple<double, std::pair<int, int>>> inter_timings_0;

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

        // Check if the interaction occurs close to the starting point
        if(int_pos[2]==0 && abs(int_pos[0]-((size_cell[0]-1)/2))<close_start_radius_param && abs(int_pos[1]-((size_cell[0]-1)/2))<close_start_radius_param)
        {
            inter_timings_0.emplace_back(time, std::make_pair(int_pos[0],int_pos[1]));
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

    // Initialize VertexTime to 0, this will store the maximum time associated with the vertex position.
    double VertexTime = 0.;
    double totEnVertex = 0.;
    // Loop through each entry in the energyMap to find the maximum time for the vertex position.
    for (const auto &entry : energyMap)
    {
        // Extract cublet and cell indices from the key (a pair of integers)
        int cub_idx = entry.first.first;
        int cell_idx = entry.first.second;

        // Convert the cublet and cell indices to a 3D integer position (x, y, z)
        std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);

        // Check if the current position matches the vertex position in the x and y axes,
        // and is within a threshold of 2 in the z-axis.
        if (int_pos[0] == vertex_pos[0] && int_pos[1] == vertex_pos[1] && abs(int_pos[2] - vertex_pos[2]) < 2)
        {
            totEnVertex += entry.second.first;
            VertexTime += entry.second.second;

            // // Update VertexTime with the maximum time value found for the vertex.
            // if (VertexTime < entry.second.second)
            // {
            //     VertexTime = entry.second.second;
            // }
        }
    }

    VertexTime = VertexTime/totEnVertex;

    // Initialize min_time_0 to the largest possible double value (used to track the minimum time).
    double min_time_0 = std::numeric_limits<double>::max();

    // Loop through the inter_timings_0 (a collection of pairs) to find the minimum timing value.
    for (const auto &elem : inter_timings_0)
    {
        double value;
        std::pair<int, int> pair;

        // Unpack the tuple, extracting the timing value and the associated pair of integers.
        std::tie(value, pair) = elem;

        // Update min_time_0 with the minimum timing value found.
        if (value < min_time_0)
        {
            min_time_0 = value;
        }
    }

    // Subtract min_time_0 from VertexTime to normalize the vertex time relative to the minimum time.
    VertexTime = VertexTime - min_time_0;

    inputFile->Close();
    delete inputFile;
    
    std::vector<double> out(2,0);

    out[0] = min_time_0;
    out[1] = (double(z_vertex+1))/(VertexTime);
   
    return out;
}

void fillTable(std::string particleName, std::vector<int> size_cell,double threshold)
{

    std::string outFile = "./results_" + std::to_string(size_cell[0]) + "_" + std::to_string(size_cell[1]) + "_" + std::to_string(size_cell[2])+ "/" + particleName + ".tsv";
    std::ofstream oFile(outFile, std::ios::out);

    oFile << "FileName\t";
    oFile << "EventNum\t";

    oFile << "time0\t";
    oFile << "speed";

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
                    

                    std::vector<double> info = speedAnalysis(fileName,i,size_cell,threshold);
                    oFile << file->GetName() << "\t" << i << "\t" << info[0] << "\t" << info[1] << std::endl;

                    std::cout << CURSOR_TO_START << CLEAR_LINE;
                                
                }

            }
        }                   
    }

    oFile.close();
}

int main(int argc, char* argv[]) {

    // Check if the correct number of arguments is provided
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <particle> <size_x> <size_y> <size_z> <threshold>" << std::endl;
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

    // Check if the folder exists and create it if it doesn't
    std::string folderPath = "./results_" + std::to_string(size_x) + "_" + std::to_string(size_y) + "_" + std::to_string(size_z);
    // Try to create the folder
    if (createDirectory(folderPath) == 0) {
        std::cout << "Directory created successfully: " << folderPath << std::endl;
    } else {
        std::cout << "Directory already exists or couldn't be created: " << folderPath << std::endl;
    }

    // Call a function that fills the table with the given particle type
    fillTable(particle,{size_x,size_y,size_z},threshold);

}
