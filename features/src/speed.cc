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

#include "../include/utils.h"
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
using Key = int;
using Value = std::tuple<double, double, double>;

// Define DataMap as an unordered map where the key is of type Key and the value is of type Value
using DataMap = std::unordered_map<Key, Value>;

// Define terminal macros for cursor movement and line clearing
#define CURSOR_TO_START "\033[1G"
#define CLEAR_LINE "\033[K"
double DELTA_SMEARING = 30.;

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

std::vector<double> speedAnalysis(  std::string particleName, 
                                    TString filePath,
                                    int eventNum, 
                                    std::vector<int> size_cell,
                                    double threshold, 
                                    int xyWindowSize_param,
                                    int zWindowSize_param,
                                    std::string smear)
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

    // Shift due to the presence of a tracker (2.6 + 0.4 m)
    std::vector<double> deltaT_TL = shift_time(particleName,2.6,smear);

    for (size_t j = 0; j < Tentries; j++) 
    {
        // Get the cublet and cell indices, energy, and time for the current interaction
        int cub_idx = (*Tcublet_idx)[j];
        int cell_idx = (*Tcell_idx)[j];
        double E = (*Tedep)[j];  // Energy deposited in the current cell

        if (E>0)
        {
            double time = (*Ttime)[j];  // Timestamp of the current interaction
            time = time*1000 + deltaT_TL[0];

            if (smear == "d")
            {
                time = int(time / DELTA_SMEARING);
            }
            
            // Convert the cublet and cell indices to a 3D position
            std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);
            int index_cell = convert_to_index(cub_idx, cell_idx, size_cell);
            
            // Create a key (cublet, cell) to store in the energy map
            Key key = index_cell;

            // Update the energy and time in the energy map
            if (energyMap.find(key) != energyMap.end()) {
                
                // If the key already exists, add the energy and update the time if it's more recent
                std::get<0>(energyMap[key]) += E;
                std::get<1>(energyMap[key]) += E*time;
                std::get<2>(energyMap[key]) +=pow(E,2);
            

            } else {
                // If the key does not exist, create a new entry with the current energy and time
                energyMap[key] = std::make_tuple(E, E*time,pow(E,2));
            }

            // Check if the interaction is within the XY window range
            if((int_pos[0] >= minWin && int_pos[0] <= maxWin) && 
            (int_pos[1] >= minWin && int_pos[1] <= maxWin))
            {
                // Accumulate energy in the corresponding z-slice
                en_position[int_pos[2]] += E;
            }
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

    if (z_vertex >= 0)
    {
        //// X_VERTEX and Y_VERTEX + PostVertexEnergyFraction + VertexTime + numCellBeforeVertex + MAX and SECONDMAX

        // Variable to store the fraction of energy deposited after the z_vertex
        double PostVertexEnergyFraction = 0.0;
        int numCellBeforeVertex = 0;  // Counter for the number of cells before the vertex (z_vertex)
        double VertexTime = 10e5;
        int x_vertex = 0;
        int y_vertex = 0;
        int vertex_index = 0;

        double squareEnergySum = 0.0;                                  // To hold squareEnergySum
        double totEnergyVertex = 0.0;                                  // To hold totEnergyVertex

        for (const auto& entry : energyMap) {
            
            int index = entry.first;
            int z_layer = index / (size_cell[0]*size_cell[1]);

            if (z_layer == z_vertex)
            {   
                double weightedTime = std::get<1>(entry.second) / std::get<0>(entry.second);

                if (weightedTime < VertexTime)
                {   

                    double E2 = std::get<2>(entry.second);
                    squareEnergySum = E2;
                    totEnergyVertex = std::get<0>(entry.second);

                    VertexTime = weightedTime;

                    int plane_index =  index % (size_cell[0]*size_cell[1]);

                    x_vertex = (plane_index % size_cell[0]);
                    y_vertex = (size_cell[1] - 1) - (plane_index / size_cell[0]);

                    vertex_index = index;

                }
            }
            
        }

        if (smear == "y")
        {   
            double sigma_verteTime = (sqrt(squareEnergySum)/totEnergyVertex)*DELTA_SMEARING;
            VertexTime = smearing_time(VertexTime,sigma_verteTime);
        }

        // Store the (x, y, z) coordinates of the vertex and find the traveled distance
        std::vector<int> vertex_pos = {x_vertex, y_vertex, z_vertex};
        std::vector<int> central_cell = {size_cell[0]/2 , size_cell[1]/2, 0};
        double traveled_distance = sqrt(pow(vertex_pos[0]*1.-central_cell[0]*1.,2)+pow(vertex_pos[1]*1.-central_cell[1]*1.,2)+pow(z_vertex*1.+1.,2));
        // double traveled_distance = z_vertex + 1.;
        
        // Subtract min_time_0 from VertexTime to normalize the vertex time relative to the minimum time.
        double travel_time = (VertexTime - deltaT_TL[1] > 0) ? VertexTime - deltaT_TL[1] : 0.1;
        
        inputFile->Close();
        delete inputFile;
        
        std::vector<double> out(2,0);

        out[0] = deltaT_TL[1];
        out[1] = traveled_distance/(travel_time)*100.; //scaled speed (scaling factor = 100)
    
        return out;
    }
    else
    {

        inputFile->Close();
        delete inputFile;

        std::vector<double> out(2,0);

        out[0] = deltaT_TL[1];
        out[1] = -1;
    
        return out;

    }
}

void fillTable( std::string particleName, 
                std::vector<int> size_cell,
                double threshold,
                int xyWindowSize_param, 
                int zWindowSize_param,
                std::string smear="y", 
                std::string folderPath="")
{

    std::string outFile = folderPath + "/" + particleName + ".tsv";
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
                        

                    std::vector<double> info = speedAnalysis(particleName,fileName,i,size_cell,threshold, xyWindowSize_param, zWindowSize_param,smear);
                    oFile << file->GetName() << "\t" << i << "\t" << info[0] << "\t" << info[1] << std::endl;

                    std::cout << CURSOR_TO_START << CLEAR_LINE;          
                }

            }
        }                   
    }

    oFile.close();
}

/*

Values:
    - threshold :       125 ( 100 100 100 ) - 125   ( 50 50 100 ) - 200 ( 25 25 100 ) - 175  ( 100 100 50 ) - 250 ( 100 100 25 )
    - xyWindow :        2   ( 100 100 100 ) - 1     ( 50 50 100 ) - 1   ( 25 25 100 ) - 1   ( 100 100 50 ) - 2  ( 100 100 25 )
    - zWindow :         1   ( 100 100 100 ) - 1     ( 50 50 100 ) - 1   ( 25 25 100 ) - 1   ( 100 100 50 ) - 1  ( 100 100 25 )

*/

int main(int argc, char* argv[]) {

    // Check if the correct number of arguments is provided
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " <particle> <size_x> <size_y> <size_z> <threshold> <xyWindow> <zWindow> <smearing>" << std::endl;
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
    int xyWin = std::stoi(argv[6]);
    int zWin = std::stoi(argv[7]);

    std::string smear = argv[8];

    // Check if the folder exists and create it if it doesn't
    std::string folderPath = "../dataset/";
    
    // Try to create the folder
    if (createDirectory(folderPath) == 0) {
        std::cout << "Directory created successfully: " << folderPath << std::endl;
    } else {
        std::cout << "Directory already exists or couldn't be created: " << folderPath << std::endl;
    }
    
    folderPath += "results_" + std::to_string(size_x) + "_" + std::to_string(size_y) + "_" + std::to_string(size_z);

    // Try to create the folder
    if (createDirectory(folderPath) == 0) {
        std::cout << "Directory created successfully: " << folderPath << std::endl;
    } else {
        std::cout << "Directory already exists or couldn't be created: " << folderPath << std::endl;
    }

    folderPath += "/speed/";

    // Try to create the folder
    if (createDirectory(folderPath) == 0) {
        std::cout << "Directory created successfully: " << folderPath << std::endl;
    } else {
        std::cout << "Directory already exists or couldn't be created: " << folderPath << std::endl;
    }
    
    if (smear == "y")
    {
        folderPath += "Smearing";
    }
    else if (smear == "d")
    {
        folderPath += "Digitalization";
    }
    else 
    {
        folderPath += "noSmearing";
    }

    // Try to create the folder
    if (createDirectory(folderPath) == 0) {
        std::cout << "Directory created successfully: " << folderPath << std::endl;
    } else {
        std::cout << "Directory already exists or couldn't be created: " << folderPath << std::endl;
    }

    // Call a function that fills the table with the given particle type
    fillTable(  particle, 
                {size_x,size_y,size_z},
                threshold,
                xyWin, 
                zWin,
                smear,
                folderPath
                );
}
