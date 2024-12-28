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
#include <unordered_map>
#include <utility>
#include <vector>

#include "../include/utils.h"

using namespace std;

// Global variable to hold the number of entries in the tree
int Tentries;

// Pointers to vectors that hold various types of data from the ROOT tree
vector<int> *Tcublet_idx = nullptr; // Cublet indices
vector<int> *Tcell_idx = nullptr;   // Cell indices
vector<double> *Ttime = nullptr;    // Time values
vector<double> *Tedep = nullptr;    // Deposited energy values
vector<double> *Tpdg = nullptr;     // PDG codes
vector<double> *Tdeltae = nullptr;  // Delta energy values

// Define Key as a pair of integers and Value as a pair of doubles
using Key = int;
using Value = std::tuple<double, double, double>;

// Define DataMap as an unordered map where the key is of type Key and the value is of type Value
using DataMap = std::unordered_map<Key, Value>;

// Define macros for cursor control in console output
#define CURSOR_TO_START "\033[1G"
#define CLEAR_LINE "\033[K"
double DELTA_SMEARING = 40.;

// OS-specific directory creation
#ifdef _WIN32
#include <direct.h>  // Windows-specific header for directory creation
#define mkdir _mkdir // Define mkdir for Windows
#else
#include <sys/stat.h> // POSIX-compliant systems (Linux, macOS)
#endif

// Function to create a directory at the specified path
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
} // namespace std

// Function to calculate the Euclidean distance between two 3D positions
double distance(std::vector<int> ref, std::vector<int> pos)
{
    // Calculate the Euclidean distance between the reference and target points
    double dist = sqrt(pow(ref[0] - pos[0], 2) + pow(ref[1] - pos[1], 2) + pow(ref[2] - pos[2], 2));
    return dist;
}

std::vector<double> generalFeature(std::string particleName, TString filePath, int eventNum,std::vector<int> size_cell, int Efraction_cell_radius_param,std::string smear)
{
    TFile* inputFile = TFile::Open(filePath);
    TTree* Tree = dynamic_cast<TTree*>(inputFile->Get("outputTree"));

    // Set branch addresses
    Tree->SetBranchAddress("Tinteractions_in_event", &Tentries);
    Tree->SetBranchAddress("Tcublet_idx", &Tcublet_idx);
    Tree->SetBranchAddress("Tcell_idx", &Tcell_idx);
    Tree->SetBranchAddress("Tedep", &Tedep);
    Tree->SetBranchAddress("Tglob_t", &Ttime);

    // Get the specified entry from the tree
    Tree->GetEntry(eventNum);

    // Define variables for storing the energy deposition data
    DataMap energyMap;  // Map to store energy and time for each cublet-cell pair
    double totalEnergy = 0.0;  // Total energy deposited in the event

    double deltaT_TL = shift_time(particleName,2.6,smear)[0];
    for (size_t j = 0; j < Tentries; j++) 
    {
        // Get the cublet and cell indices, energy, and time for the current interaction
        int cub_idx = (*Tcublet_idx)[j];
        int cell_idx = (*Tcell_idx)[j];
        double E = (*Tedep)[j];  // Energy deposited in the current cell
        double time = (*Ttime)[j];  // Timestamp of the current interaction
        time = time*1000 + deltaT_TL;

        // Convert the cublet and cell indices to a 3D position
        std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);
        
        int index_cell = convert_to_index(cub_idx, cell_idx, size_cell);
        
        // Create a key (cublet, cell) to store in the energy map
        Key key = index_cell;

        // Accumulate the total energy for the event and compute time0
        totalEnergy += E;
        
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
    }
    
    ///////////////////////////////////////////////////
    ////////// FIRST AND SECOND MAX DISTANCE //////////    
    ///////////////////////////////////////////////////

    Key maxKey, secondMaxKey;  // Variables to store the keys corresponding to max and second max energy
    double maxE = -1.0, secondMaxE = -1.0;  // Initialize max and second max energies to -1

    for (const auto& entry : energyMap) {
        
        int index = entry.first;
        
        int z_layer = index / (size_cell[0]*size_cell[1]);
        
        double E = std::get<0>(entry.second);
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
    }

    std::vector<int> pos_max = index_to_pos(maxKey,size_cell);
    std::vector<int> pos_secondMax = index_to_pos(secondMaxKey,size_cell);

    double distanceFirstSecondMaxEnergy = distance(pos_max,pos_secondMax);

    /////////////////////////////////////////////////////////////////////
    ////////// WEIGHTED TIME  + EfractionCell + numUniqueCells //////////    
    /////////////////////////////////////////////////////////////////////

    int unique_cells = energyMap.size();
    double EfractionCell_num = 0.0;   // Numerator for energy fraction within a specific radius
    double EfractionCell_denom = 0.0; // Denominator for energy fraction within a distance of 1

    double weightedTime = 0.0;
    double squareEnergySum = 0.0;                                  // To hold squareEnergySum
    double totEnergyVertex = 0.0;                                  // To hold totEnergyVertex

    // Iterate over each entry in the energyMap, which maps keys (cublet, cell) to energy and time values
    for (const auto &entry : energyMap)
    {

        int index = entry.first; 
        std::vector<int> pos = index_to_pos(entry.first,size_cell);
        double energy = std::get<0>(entry.second); 

        double dist = distance(pos_max, pos);
        if (dist <= Efraction_cell_radius_param)
        {

            EfractionCell_num += energy; // Accumulate energy for the numerator
        }
        // Check if the distance is within a distance of 1 for the denominator
        if (dist <= 1)
        {
            EfractionCell_denom += energy; // Accumulate energy for the denominator
        }

        weightedTime += std::get<1>(entry.second)/totalEnergy;

        squareEnergySum += std::get<2>(entry.second);

    }

    if (smear == "y")
    {   
        double sigma_weightedTime = (sqrt(squareEnergySum)/totalEnergy)*DELTA_SMEARING;
        weightedTime = smearing_time(weightedTime,sigma_weightedTime);
    }

    // Calculate the energy fraction (normalized) based on the numerator and denominator
    double EfractionCell = EfractionCell_num / EfractionCell_denom - 1.0;

    ///////////////////////////////////////////////////////
    ////////// RatioEcell + deltaEcell_secondmax //////////    
    ///////////////////////////////////////////////////////

    double RatioEcell = (maxE - secondMaxE) / (maxE + secondMaxE);
    double deltaEcell_secondmax = maxE - secondMaxE;

    inputFile->Close();
    delete inputFile;

    std::vector<double> out(9,0);
    out[0] = totalEnergy;
    out[1] = unique_cells;
    out[2] = maxE;
    out[3] = secondMaxE;
    out[4] = distanceFirstSecondMaxEnergy;
    out[5] = RatioEcell;
    out[6] = deltaEcell_secondmax;
    out[7] = EfractionCell;
    out[8] = weightedTime;

    return out;
}

void fillTable( std::string particleName,
                std::vector<int> size_cell, 
                int Efraction_cell_radius_param,
                std::string smear="y", 
                std::string folderPath="")
{

    std::string outFile = folderPath + "/" + particleName + ".tsv";
    std::ofstream oFile(outFile, std::ios::out);

    oFile << "FileName\t";
    oFile << "EventNum\t";

    oFile << "TotalEnergy\t";
    oFile << "NumberOfUniqueCells\t";

    oFile << "MaxEnergyInCell\t";
    oFile << "SecondMaxEnergyInCell\t";
    oFile << "distanceFirstSecondMaxEnergy\t";
    oFile << "RatioEcell\t"; // (E_max - E_2ndmax) / (E_max + E_2ndmax) [ will be 1 if no E_2ndmax found ]
    oFile << "DeltaEcell_secondMax\t"; // E_max - E_2ndmax
    oFile << "EfractionCell\t"; // E(within up to +-N cells around E_max) / E(within up to +-1 cells around E_max) - 1.0  

    oFile << "weightedTime"; 
    
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
                    

                    std::vector<double> info = generalFeature(particleName,fileName,i,size_cell, Efraction_cell_radius_param,smear);

                    oFile << file->GetName() << "\t" << i << "\t" << info[0] << "\t" << info[1] <<  "\t" << info[2] <<  "\t" << info[3] << "\t" << info[4] <<  "\t" << info[5] << "\t" << info[6] << "\t" << info[7] <<  "\t" << info[8] << std::endl;
                    std::cout << CURSOR_TO_START << CLEAR_LINE;
                                
                }

            }
        }                   
    }

    oFile.close();
}

/*

Values:
    - Efraction :       5 ( 100 100 100 ) - 3   ( 50 50 100 ) - 2 ( 25 25 100 ) - 3 ( 100 100 50 ) - 2 ( 100 100 25 ) - 2 ( 10 10 100 )   

*/

int main(int argc, char* argv[]) {

    // Check if the correct number of arguments is provided
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <particle> <size_x> <size_y> <size_z> <Efraction> <smearing>" << std::endl;
        return 1;
    }

    // Retrieve and store the particle type from the first argument
    std::string particle = argv[1];
    
    // Convert the second, third, and fourth arguments to integers for the grid size
    int size_x = std::stoi(argv[2]);
    int size_y = std::stoi(argv[3]);
    int size_z = std::stoi(argv[4]);

    int rad = std::stoi(argv[5]);

    std::string smear = argv[6];


    // Check if the folder exists and create it if it doesn't
    std::string folderPath = "./results/";
    
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

    folderPath += "/generalFeatures/";

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
    fillTable(particle,{size_x,size_y,size_z},rad,smear,folderPath);

}
