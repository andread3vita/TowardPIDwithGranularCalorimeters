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
#include <sys/types.h>
#include <utility>
#include <vector>

#include "../include/utils.h" // Custom utility functions
using namespace std;

int Tentries; // Number of entries in the event

// Vectors to store data from the ROOT tree
vector<int> *Tcublet_idx;
vector<int> *Tcell_idx;
vector<double> *Ttime;
vector<double> *Tedep;
vector<double> *Tpdg;
vector<double> *Tdeltae;

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

int createDirectory(const std::string &path)
{
#ifdef _WIN32
    return mkdir(path.c_str()); // Use _mkdir for Windows
#else
    return mkdir(path.c_str(), 0777); // Use POSIX mkdir for Unix-like systems
#endif
}

// Function to compute the asymmetry using rescaled coordinates for both x and y axes
std::vector<double> missingEnergy(TString filePath, int eventNum, std::vector<int> size_cell)
{
    TFile *inputFile = TFile::Open(filePath);
    TTree *Tree = dynamic_cast<TTree *>(inputFile->Get("outputTree"));

    // Set branch addresses to read data from the tree
    Tree->SetBranchAddress("Tinteractions_in_event", &Tentries);
    Tree->SetBranchAddress("Tcublet_idx", &Tcublet_idx);
    Tree->SetBranchAddress("Tcell_idx", &Tcell_idx);
    Tree->SetBranchAddress("Tedep", &Tedep);

    Tree->GetEntry(eventNum); // Get the data for the specific event

    double asy_x = 0.0;
    double asy_y = 0.0;

    // Loop through each interaction in the event
    for (size_t j = 0; j < Tentries; j++)
    {
        int cub_idx = (*Tcublet_idx)[j]; // Cublet index
        int cell_idx = (*Tcell_idx)[j];  // Cell index
        double E = (*Tedep)[j];          // Energy deposited

        // Convert indices into 3D position
        std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);

        // Rescale the x and y coordinates
        double rescaled_x = int_pos[0] - (size_cell[0] - 1) / 2;
        double rescaled_y = int_pos[1] - (size_cell[1] - 1) / 2;

        // Compute the asymmetry as energy-weighted displacements
        asy_x += E * rescaled_x;
        asy_y += E * rescaled_y;
    }

    // Compute the overall asymmetry based on the x and y components
    double Asymmetry = sqrt(pow(asy_x, 2) + pow(asy_y, 2));

    // Close the file and clean up
    inputFile->Close();
    delete inputFile;

    return {std::abs(asy_x), std::abs(asy_y), Asymmetry}; // Return the x, y, and total asymmetry
}

// Function to compute asymmetry based on simple positive/negative x and y energy deposits
std::vector<double> missingEnergyPlain(TString filePath, int eventNum, std::vector<int> size_cell)
{
    TFile *inputFile = TFile::Open(filePath);
    TTree *Tree = dynamic_cast<TTree *>(inputFile->Get("outputTree"));

    // Set branch addresses to read data from the tree
    Tree->SetBranchAddress("Tinteractions_in_event", &Tentries);
    Tree->SetBranchAddress("Tcublet_idx", &Tcublet_idx);
    Tree->SetBranchAddress("Tcell_idx", &Tcell_idx);
    Tree->SetBranchAddress("Tedep", &Tedep);

    Tree->GetEntry(eventNum); // Get the data for the specific event

    double asy_x = 0.0;
    double asy_y = 0.0;

    // Loop through each interaction in the event
    for (size_t j = 0; j < Tentries; j++)
    {
        int cub_idx = (*Tcublet_idx)[j]; // Cublet index
        int cell_idx = (*Tcell_idx)[j];  // Cell index
        double E = (*Tedep)[j];          // Energy deposited

        // Convert indices into 3D position
        std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);

        // Rescale the x and y coordinates
        double rescaled_x = int_pos[0] - (size_cell[0] - 1) / 2;
        double rescaled_y = int_pos[1] - (size_cell[1] - 1) / 2;

        // Adjust the asymmetry calculation based on the sign of the coordinates
        if (rescaled_x > 0)
        {
            asy_x += E; // Positive x contribution
        }
        else if (rescaled_x < 0)
        {
            asy_x -= E; // Negative x contribution
        }

        if (rescaled_y > 0)
        {
            asy_y += E; // Positive y contribution
        }
        else if (rescaled_y < 0)
        {
            asy_y -= E; // Negative y contribution
        }
    }

    // Compute the overall asymmetry based on the x and y components
    double Asymmetry = sqrt(pow(asy_x, 2) + pow(asy_y, 2));

    // Close the file and clean up
    inputFile->Close();
    delete inputFile;

    return {std::abs(asy_x), std::abs(asy_y), Asymmetry}; // Return the x, y, and total asymmetry
}

void fillTable(std::string particleName, std::vector<int> size_cell = {100, 100, 100})
{

    string outFile = "./results_" + std::to_string(size_cell[0]) + "_" + std::to_string(size_cell[1]) + "_" +
                     std::to_string(size_cell[2]) + "/" + particleName + ".tsv";
    std::ofstream oFile(outFile, std::ios::out);

    oFile << "FileName\t";
    oFile << "EventNum\t";
    oFile << "AsymmetryX\t";
    oFile << "AsymmetryY\t";
    oFile << "Asymmetry\t";
    oFile << "AsymmetryX_plain\t";
    oFile << "AsymmetryY_plain\t";
    oFile << "Asymmetry_plain";

    oFile << std::endl;

    std::string dirPath = returnFilePath(particleName);

    TSystemDirectory dir(dirPath.c_str(), dirPath.c_str());
    TList *filesList = dir.GetListOfFiles();

    int totEv = 0;
    if (filesList)
    {
        TSystemFile *file;
        TString fileName;
        TIter next(filesList);

        auto start = std::chrono::high_resolution_clock::now();
        while ((file = (TSystemFile *)next()))
        {

            if (!file->IsDirectory())
            {

                TString fileName = dirPath + file->GetName();
                for (size_t i = 0; i < 1000; i++)
                {

                    totEv += 1;

                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> duration = end - start;

                    std::cout << "Processing: " << file->GetName() << "\tEvent: " << i
                              << "\t\ttime[min]: " << (duration.count() / 1000) / 60
                              << "\t\tProgress: " << totEv / 50e3 * 100 << "%" << std::flush;

                    std::vector<double> info = missingEnergy(fileName, i, size_cell);
                    std::vector<double> info_plain = missingEnergyPlain(fileName, i, size_cell);

                    oFile << file->GetName() << "\t" << i << "\t" << info[0] << "\t" << info[1] << "\t" << info[2]
                          << "\t" << info_plain[0] << "\t" << info_plain[1] << "\t" << info_plain[2] << std::endl;

                    std::cout << CURSOR_TO_START << CLEAR_LINE;
                }
            }
        }
    }

    oFile.close();
}

int main(int argc, char *argv[])
{

    // Check if the correct number of arguments is provided
    if (argc != 5)
    {
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
    std::string folderPath =
        "./results_" + std::to_string(size_x) + "_" + std::to_string(size_y) + "_" + std::to_string(size_z);
    // Try to create the folder
    if (createDirectory(folderPath) == 0)
    {
        std::cout << "Directory created successfully: " << folderPath << std::endl;
    }
    else
    {
        std::cout << "Directory already exists or couldn't be created: " << folderPath << std::endl;
    }

    // Call a function that fills the table with the given particle type
    fillTable(particle, {size_x, size_y, size_z});
}