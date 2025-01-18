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
#include <hdf5.h>

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

// Define macros for cursor control in console output
#define CURSOR_TO_START "\033[1G"
#define CLEAR_LINE "\033[K"
double DELTA_SMEARING = 30.;

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

template <typename T>
class Array3D {
    std::vector<float> data;
    int X, Y, Z;

public:
    Array3D(int x, int y, int z) : X(x), Y(y), Z(z), data(x * y * z, 0.0f) {}

    float& operator()(int x, int y, int z) {
        return data[x * Y * Z + y * Z + z];
    }

    const float& operator()(int x, int y, int z) const {
        return data[x * Y * Z + y * Z + z];
    }

    const std::vector<float>& getData() const { return data; }
    int getX() const { return X; }
    int getY() const { return Y; }
    int getZ() const { return Z; }
};

template <typename T>
void saveArrayToHDF5(const Array3D<T>& array, int label, const std::string& filename) {
    // Apri o crea il file HDF5
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Dati dell'array
    const std::vector<T>& data = array.getData();
    hsize_t dims[3] = {static_cast<hsize_t>(array.getX()),
                       static_cast<hsize_t>(array.getY()),
                       static_cast<hsize_t>(array.getZ())};

    // Crea lo spazio dei dati
    hid_t dataspace_id = H5Screate_simple(3, dims, nullptr);

    // Crea il dataset per i dati dell'array
    hid_t dataset_id = H5Dcreate2(file_id, "array_data", H5T_IEEE_F32LE, dataspace_id,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Scrivi i dati dell'array
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

    // Crea un dataset per il label
    hid_t labelspace_id = H5Screate(H5S_SCALAR); // Spazio per un singolo valore
    hid_t label_id = H5Dcreate2(file_id, "label", H5T_STD_I32LE, labelspace_id,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Scrivi il label
    H5Dwrite(label_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &label);

    // Chiudi tutti gli handle HDF5
    H5Dclose(label_id);
    H5Sclose(labelspace_id);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
}


Array3D<float> make3Dmap(  std::string particleName, 
                                TString filePath, 
                                int eventNum,
                                std::vector<int> size_cell,
                                std::string smear)
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
    Array3D<float> energyMap(size_cell[0], size_cell[1], size_cell[2]);
    double deltaT_TL = shift_time(particleName,2.6,smear)[0];

    for (size_t j = 0; j < Tentries; j++) 
    {
        // Get the cublet and cell indices, energy, and time for the current interaction
        int cub_idx = (*Tcublet_idx)[j];
        int cell_idx = (*Tcell_idx)[j];
        double E = (*Tedep)[j];  // Energy deposited in the current cell

        if (E>0)
        {
            double time = (*Ttime)[j];  // Timestamp of the current interaction
            time = time*1000 + deltaT_TL;

            if (smear == "d")
            {
                time = int(time / DELTA_SMEARING);
            }

            // Convert the cublet and cell indices to a 3D position
            std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);
            energyMap(int_pos[0],int_pos[1],int_pos[2]) = E;
            
        }
    }
    

    inputFile->Close();
    delete inputFile;

    return energyMap;
}

void fillTable( std::string particleName,
                std::vector<int> size_cell, 
                std::string smear="y", 
                std::string folderPath="")
{

    std::string oFile = folderPath + "/" + particleName + ".h5";

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
                    

                    Array3D<float> info = make3Dmap(particleName,fileName,i,size_cell,smear);

                    int label = 0;

                    if (particleName == "proton")
                    {
                        label = 0;
                    }
                    else if (particleName == "pion")
                    {
                        label = 1;
                    }
                    else if (particleName == "kaon")
                    {
                        label = 2;
                    }
                    else
                    {
                        label = -1;
                    }
                    saveArrayToHDF5(info, label, oFile); 

                    std::cout << CURSOR_TO_START << CLEAR_LINE;
                                
                }

            }
        }                   
    }
}

/*

Values:
    - Efraction :       5 ( 100 100 100 ) - 3   ( 50 50 100 ) - 2 ( 25 25 100 ) - 3 ( 100 100 50 ) - 2 ( 100 100 25 ) - 2 ( 10 10 100 )   

*/

int main(int argc, char* argv[]) {

    // Check if the correct number of arguments is provided
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <particle> <size_x> <size_y> <size_z> <smearing>" << std::endl;
        return 1;
    }

    // Retrieve and store the particle type from the first argument
    std::string particle = argv[1];
    
    // Convert the second, third, and fourth arguments to integers for the grid size
    int size_x = std::stoi(argv[2]);
    int size_y = std::stoi(argv[3]);
    int size_z = std::stoi(argv[4]);

    std::string smear = argv[5];


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

    folderPath += "/3Dmaps/";

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
    fillTable(particle,{size_x,size_y,size_z},smear,folderPath);

}
