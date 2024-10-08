#include <TSpectrum.h>
#include "TCanvas.h"
#include "TRandom.h"
#include "TH2.h"
#include "TF2.h"
#include "TMath.h"
#include "TROOT.h"

#include <iostream>
#include <chrono>
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
#include <fstream>

#include "../include/utils.h" // Include custom utilities header

using namespace std; // Use standard namespace

// Global variables for data entries
int Tentries;

// Pointers to vectors for various data types, used to hold tree branches data
vector<int> *Tcublet_idx; // Vector for cublet indices
vector<int> *Tcell_idx;   // Vector for cell indices
vector<double> *Ttime;    // Vector for timing data
vector<double> *Tedep;    // Vector for deposited energy
vector<double> *Tpdg;     // Vector for particle ID
vector<double> *Tdeltae;  // Vector for delta energy

// Escape codes for terminal cursor manipulation
#define CURSOR_TO_START "\033[1G" // Move cursor to the beginning of the line
#define CLEAR_LINE "\033[K"       // Clear the current line

#ifdef _WIN32
#include <direct.h>  // Windows-specific header for directory creation
#define mkdir _mkdir // Define mkdir for Windows
#else
#include <sys/stat.h> // Include POSIX-compliant system header for directory operations
#endif

// Function to create a directory with the specified path
int createDirectory(const std::string &path)
{
#ifdef _WIN32
    return mkdir(path.c_str()); // Use _mkdir for Windows systems
#else
    return mkdir(path.c_str(), 0777); // Use POSIX mkdir for Unix-like systems, setting permissions to 0777
#endif
}

std::vector<double> SpatialObservables(TString filePath, int eventNum, std::vector<int> size_cell)
{
    // Open the ROOT file specified by filePath
    TFile *inputFile = TFile::Open(filePath);

    // Get the TTree named "outputTree" from the file
    TTree *Tree = dynamic_cast<TTree *>(inputFile->Get("outputTree"));

    // Set the branch addresses to read data from the tree
    Tree->SetBranchAddress("Tinteractions_in_event", &Tentries);
    Tree->SetBranchAddress("Tcublet_idx", &Tcublet_idx);
    Tree->SetBranchAddress("Tcell_idx", &Tcell_idx);
    Tree->SetBranchAddress("Tedep", &Tedep);

    // Read the specific event data from the TTree
    Tree->GetEntry(eventNum);

    // Vectors to store the positions and energies
    std::vector<int> x_pos;     // X positions of interactions
    std::vector<int> y_pos;     // Y positions of interactions
    std::vector<int> z_pos;     // Z positions of interactions
    std::vector<double> ENERGY; // Energies deposited in each interaction

    // Loop through all interactions in the event
    for (size_t j = 0; j < Tentries; j++)
    {
        // Get cublet and cell indices for the current interaction
        int cub_idx = (*Tcublet_idx)[j];
        int cell_idx = (*Tcell_idx)[j];
        double E = (*Tedep)[j];

        // Convert indices to spatial positions
        std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);

        // Store positions and energy
        x_pos.push_back(int_pos[0]);
        y_pos.push_back(int_pos[1]);
        z_pos.push_back(int_pos[2]);
        ENERGY.push_back(E);
    }

    // Calculate mean positions weighted by energy
    vector<double> x_mean = computeMean(x_pos, ENERGY, 1);
    vector<double> y_mean = computeMean(y_pos, ENERGY, 1);
    vector<double> z_mean = computeMean(z_pos, ENERGY, 1.);

    // Calculate radial distributions
    vector<double> rad = computeRadius(x_pos, y_pos, ENERGY, x_mean[0], y_mean[0], true);
    vector<double> rad_plain = computeRadius(x_pos, y_pos, ENERGY, x_mean[0], y_mean[0], false);

    // Calculate longitudinal shower distributions
    vector<double> z_long = computeLongitudinalShower(z_pos, ENERGY, z_mean[0], true);
    vector<double> z_long_plain = computeLongitudinalShower(z_pos, ENERGY, z_mean[0], false);

    // Close the input file and release the resource
    inputFile->Close();
    delete inputFile;

    // Return the computed observables
    return {rad[0], rad[1], z_long[0], z_long[1], rad_plain[0], rad_plain[1], z_long_plain[0], z_long_plain[1]};
}

void fillTable(std::string particleName,std::vector<int> size_cell = {100,100,100})
{

    string outFile =  "./results_" + std::to_string(size_cell[0]) + "_" + std::to_string(size_cell[1]) + "_" + std::to_string(size_cell[2]) + "/" + particleName + ".tsv";
    std::ofstream oFile(outFile, std::ios::out);

    oFile << "FileName\t";
    oFile << "EventNum\t";

    oFile << "radius\t";
    oFile << "radialSigma\t";

    oFile << "length\t";
    oFile << "longitudinalSigma\t";

    oFile << "radius_plain\t";
    oFile << "radialSigma_plain\t";

    oFile << "length_plain\t";
    oFile << "longitudinalSigma_plain";

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
                    

                    std::vector<double> info = SpatialObservables(fileName,i,size_cell);

                    oFile << file->GetName() << "\t" << i << "\t" << info[0] << "\t" << info[1] <<  "\t" << info[2] <<  "\t" << info[3] << "\t" << info[4] <<  "\t" << info[5] << "\t" << info[6] << "\t" << info[7] << std::endl;

                    std::cout << CURSOR_TO_START << CLEAR_LINE;
                                
                }

            }
        }                   
    }

    oFile.close();
}

int main(int argc, char* argv[]) {

    // Check if the correct number of arguments is provided
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <particle> <size_x> <size_y> <size_z>" << std::endl;
        return 1;
    }

    // Retrieve and store the particle type from the first argument
    std::string particle = argv[1];
    
    // Convert the second, third, and fourth arguments to integers for the grid size
    int size_x = std::stoi(argv[2]);
    int size_y = std::stoi(argv[3]);
    int size_z = std::stoi(argv[4]);

    // Check if the folder exists and create it if it doesn't
    std::string folderPath = "./results_" + std::to_string(size_x) + "_" + std::to_string(size_y) + "_" + std::to_string(size_z);
    // Try to create the folder
    if (createDirectory(folderPath) == 0) {
        std::cout << "Directory created successfully: " << folderPath << std::endl;
    } else {
        std::cout << "Directory already exists or couldn't be created: " << folderPath << std::endl;
    }

    // Call a function that fills the table with the given particle type
    fillTable(particle,{size_x,size_y,size_z});

}