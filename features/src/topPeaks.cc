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
 
#include "../include/utils.h"
using namespace std;

// Variables for tree entries and data storage
int Tentries; // Number of tree entries

// Vectors to store data for each tree entry
vector<int>* Tcublet_idx;   // Indices for cublets
vector<int>* Tcell_idx;     // Indices for cells
vector<double>* Ttime;      // Time data
vector<double>* Tedep;      // Energy deposition data
vector<double>* Tpdg;       // PDG (particle ID) data
vector<double>* Tdeltae;    // Delta energy data

// ANSI escape sequences for cursor movement and clearing lines (used in terminal output)
#define CURSOR_TO_START "\033[1G"
#define CLEAR_LINE "\033[K"

#ifdef _WIN32
    #include <direct.h>  // Windows-specific header for directory creation
    #define mkdir _mkdir // Alias to match POSIX-style function signature
#else
    #include <sys/stat.h>  // POSIX-compliant systems (Linux, macOS)
#endif

// Function to create a directory based on the operating system
int createDirectory(const std::string& path) {
    #ifdef _WIN32
        return mkdir(path.c_str());  // Use _mkdir for Windows
    #else
        return mkdir(path.c_str(), 0777);  // Use POSIX mkdir for Unix-like systems with full read-write-execute permissions
    #endif
}


std::vector<double> findPeaks_withDeco(TString filePath, int eventNum, std::vector<int> size_cell, int close_peaks_radius_param)
{
    // Controlla che size_cell abbia almeno 3 elementi
    if (size_cell.size() < 3) {
        std::cerr << "Errore: size_cell deve contenere almeno 3 elementi." << std::endl;
        return {-1};
    }

    // Apri il file ROOT
    TFile* inputFile = TFile::Open(filePath);
    if (!inputFile || inputFile->IsZombie()) {
        std::cerr << "Errore: impossibile aprire il file " << filePath << std::endl;
        return {-1};
    }

    // Ottieni l'albero
    TTree* Tree = dynamic_cast<TTree*>(inputFile->Get("outputTree"));
    if (!Tree) {
        std::cerr << "Errore: impossibile trovare 'outputTree'." << std::endl;
        inputFile->Close();
        delete inputFile;
        return {-1};
    }

    // Imposta i branch
    Tree->SetBranchAddress("Tinteractions_in_event", &Tentries);
    Tree->SetBranchAddress("Tcublet_idx", &Tcublet_idx);
    Tree->SetBranchAddress("Tcell_idx", &Tcell_idx);
    Tree->SetBranchAddress("Tedep", &Tedep);

    // Crea istogrammi
    TH2D hist_cell_zy("zy", "", size_cell[2], 0, size_cell[2], size_cell[1], 0, size_cell[1]);
    TH2D hist_cell_zx("zx", "", size_cell[2], 0, size_cell[2], size_cell[0], 0, size_cell[0]);

    // Carica l'evento specificato
    Tree->GetEntry(eventNum);

    // Loop sugli eventi
    for (size_t j = 0; j < Tentries; j++) 
    {
        int cub_idx = (*Tcublet_idx)[j];
        int cell_idx = (*Tcell_idx)[j];
        double E = (*Tedep)[j];

        if (E>0)
        {

            std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);

            hist_cell_zy.Fill(int_pos[2], int_pos[1], E);
            hist_cell_zx.Fill(int_pos[2], int_pos[0], E);
        }
    }

    // Proiezione sull'asse Z
    TH1D* projZ = hist_cell_zy.ProjectionX("projZ");
    int nBins = projZ->GetNbinsX();

    // Ricerca dei picchi
    TSpectrum spectrum;
    Int_t nfoundZ = spectrum.Search(projZ, 2, "nobackground nodraw", 0.1);

    // Estrazione dei picchi
    std::vector<double> peaksZ;
    Double_t* xpeaksZ = spectrum.GetPositionX();
    peaksZ.assign(xpeaksZ, xpeaksZ + nfoundZ);

    // Ordina il vettore in ordine crescente
    std::sort(peaksZ.begin(), peaksZ.end());

    // Pulizia
    delete projZ;
    
    double E1;
    double R1;
    if (nfoundZ > 0) {

        // Get number of bins for X and Y axis projections from the histograms
        int nBinsX_y = hist_cell_zy.GetNbinsX();
        int nBinsY_y = hist_cell_zy.GetNbinsY();
        int nBinsX_x = hist_cell_zx.GetNbinsX();
        int nBinsY_x = hist_cell_zx.GetNbinsY();

        // Create arrays of histograms to store projections of each peak
        size_t num_peaks = nfoundZ;
        TH1D* projX[num_peaks];
        TH1D* projY[num_peaks];

        // Initialize histograms for projections
        for (int i = 0; i < num_peaks; ++i) {
            std::string hist_name_x = "proj_x_" + std::to_string(i);
            std::string hist_name_y = "proj_y_" + std::to_string(i);
            projX[i] = new TH1D(hist_name_x.c_str(), "", size_cell[0], 0, size_cell[0]);
            projY[i] = new TH1D(hist_name_y.c_str(), "", size_cell[1], 0, size_cell[1]);
        }

        // Store the X, Y, Z peaks
        std::vector<std::vector<double>> out_peaks;
        double peak_pos = 0;

        // Loop through each Z peak
        for (size_t s = 0; s < num_peaks; s++) 
        {    
            peak_pos = peaksZ[s];  // Current Z threshold

            // Filter the ZX plane histogram for values around the current Z peak
            TH2D* h2d_filtered_x = new TH2D("FILTER_X", "", size_cell[2], 0, size_cell[2], size_cell[0], 0, size_cell[0]);
            for (int i = 1; i <= nBinsX_x; ++i) {
                double x = hist_cell_zx.GetXaxis()->GetBinCenter(i);

                // Select bins within the threshold range
                if (x > peak_pos - 2 && x < peak_pos + 2) {
                    for (int j = 1; j <= nBinsY_x; ++j) {
                        double content = hist_cell_zx.GetBinContent(i, j);
                        h2d_filtered_x->SetBinContent(i, j, content);

                        double error = hist_cell_zx.GetBinError(i, j);
                        h2d_filtered_x->SetBinError(i, j, error);
                    }
                }
            }

            

            // Project the filtered ZX histogram onto the Y-axis
            std::string hist_name_x = "proj_x_" + std::to_string(s);
            projX[s] = h2d_filtered_x->ProjectionY(hist_name_x.c_str());

            TSpectrum *spectrumX = new TSpectrum();

            double x_sm = 3;
            double x_thr = 0.1;
            Int_t nfoundX = spectrumX->Search(projX[s], x_sm, "nodraw nobackground", x_thr);

            // Adjust search parameters if no peaks found
            int iter = 0;
            while (!nfoundX) {
                if (x_thr < 0.01) {
                    x_thr = 0.1;
                    x_sm = 2;
                } else {
                    x_thr /= 2;
                }

                nfoundX = spectrumX->Search(projX[s], x_sm, "nodraw nobackground", x_thr);
                iter += 1;

                if (iter > 5)
                {   

                    x_thr = 0.1;
                    x_sm = 1;
                    nfoundX = spectrumX->Search(projX[s], x_sm, "nodraw nobackground", x_thr);
                }
                
            }

            // Retrieve X peak positions
            Double_t* xpeaksX = spectrumX->GetPositionX();
            std::vector<double> peaksX(xpeaksX, xpeaksX + nfoundX);

            // Find the most prominent X peak
            double peak_x = 0;
            if (peaksX.size() > 1) {
                double max_val = 0;
                for (double peak : peaksX) {
                    double temp = projX[s]->GetBinContent(projX[s]->FindBin(peak));
                    if (temp > max_val) {
                        peak_x = peak;
                        max_val = temp;
                    }
                }
            } else {
                peak_x = peaksX[0];
            }


            // Filter the ZY plane histogram for values around the current Z peak
            TH2D* h2d_filtered_y = new TH2D("FILTER_Y", "", size_cell[2], 0, size_cell[2], size_cell[1], 0, size_cell[1]);
            for (int i = 1; i <= nBinsX_y; ++i) {
                double x = hist_cell_zy.GetXaxis()->GetBinCenter(i);

                // Select bins within the threshold range
                if (x > peak_pos - 2 && x < peak_pos + 2) {
                    for (int j = 1; j <= nBinsY_y; ++j) {
                        double content = hist_cell_zy.GetBinContent(i, j);
                        h2d_filtered_y->SetBinContent(i, j, content);

                        double error = hist_cell_zy.GetBinError(i, j);
                        h2d_filtered_y->SetBinError(i, j, error);
                    }
                }
            }

            // Project the filtered ZY histogram onto the Y-axis
            std::string hist_name_y = "proj_y_" + std::to_string(s);
            projY[s] = h2d_filtered_y->ProjectionY(hist_name_y.c_str());

            TSpectrum *spectrumY = new TSpectrum();

            // Adjust search parameters if no peaks found
            double y_thr = 0.1;
            int y_sm = 3;
            Int_t nfoundY = spectrumY->Search(projY[s], y_sm, "nodraw nobackground", y_thr);

            iter = 0;
            while (!nfoundY) {
                if (y_thr < 0.01) {
                    y_thr = 0.1;
                    y_sm = 2;
                } else {
                    y_thr /= 2;
                }
                
                nfoundY = spectrumY->Search(projY[s], y_sm, "nodraw nobackground", y_thr);
                iter += 1;

                if (iter > 5)
                {  

                    y_thr = 0.1;
                    y_sm = 1;
                    nfoundY = spectrumY->Search(projY[s], y_sm, "nodraw nobackground", y_thr);
                }
            }

            // Retrieve Y peak positions
            Double_t* xpeaksY = spectrumY->GetPositionX();
            std::vector<double> peaksY(xpeaksY, xpeaksY + nfoundY);

            // Find the most prominent Y peak
            double peak_y = 0.0;
            if (peaksY.size() > 1) {
                double max_val = 0;
                for (double peak : peaksY) {
                    double temp = projY[s]->GetBinContent(projY[s]->FindBin(peak));
                    if (temp > max_val) {
                        peak_y = peak;
                        max_val = temp;
                    }
                }
            } else {
                peak_y = peaksY[0];
            }

            // Store the detected peaks (X, Y, Z) in the output vector
            out_peaks.push_back({peak_x, peak_y, peak_pos});
        
            // Clean up memory for current peak
            delete spectrumX;
            delete h2d_filtered_x;
            delete spectrumY;
            delete h2d_filtered_y;
        }

        // Vector to store energy accumulated around each peak
        std::vector<double> energy_aroundPeaks(nfoundZ, 0.0);
        double totEn = 0.0;

        // Loop through all energy depositions
        for (size_t j = 0; j < Tentries; j++) 
        {
            int cub_idx = (*Tcublet_idx)[j];
            int cell_idx = (*Tcell_idx)[j];
            double E = (*Tedep)[j]; 

            totEn += E;

            // Convert cub_idx and cell_idx to 3D positions
            std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);
            std::vector<double> double_pos(3, 0.0);
            for (size_t g = 0; g < 3; g++) {
                double_pos[g] += int_pos[g] + 0.5;  // Adjust position to center of cell
            }

            // Find which peaks are close to this point
            std::vector<int> isClose = findClosePoints(out_peaks, double_pos, close_peaks_radius_param);

            // Add energy to the corresponding peaks
            for (auto el : isClose) {
                energy_aroundPeaks[el] += E;
            }
        }   

        // Sort peaks by energy
        std::vector<int> energy_index = topEnergy_index(energy_aroundPeaks);

        // Store sorted energies and energy ratios
        std::vector<double> sorted_energies;
        std::vector<double> sorted_ratios;
        for (size_t g = 0; g < nfoundZ; g++) {
            sorted_energies.push_back(energy_aroundPeaks[energy_index[g]]);
            sorted_ratios.push_back(energy_aroundPeaks[energy_index[g]] / totEn);
        }

        for (size_t i = 0; i < num_peaks; i++) {
            if (projY[i]) delete projY[i];
            if (projX[i]) delete projX[i];
        }

        E1 = sorted_energies[0];
        R1 = sorted_ratios[0];
        
    }


    
    
    inputFile->Close();
    delete inputFile;  

    double numPeaks = nfoundZ;
    //E1
    //R1
    return {numPeaks,E1,R1};
}


void fillTable(std::string particleName,std::vector<int> size_cell = {100,100,100}, int close_peaks_radius_param = 5,std::string folderPath="")
{

     std::string outFile = folderPath + "/" + particleName + ".tsv";
    std::ofstream oFile(outFile, std::ios::out);

    oFile << "FileName\t";
    oFile << "EventNum\t";

    oFile << "numPeaks\t";
    oFile << "E1\t";
    oFile << "R1";


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
                    std::vector<double> info = findPeaks_withDeco(fileName,i,size_cell, close_peaks_radius_param);

                            
                    oFile << file->GetName() << "\t" << i << "\t" << info[0] << "\t" << info[1] << "\t" << info[2] << std::endl;
                    std::cout << CURSOR_TO_START << CLEAR_LINE;
                                
                }

            }
        }                   
    }

    oFile.close();
}

/*

Values:
    - closePeak :       5 ( 100 100 100 ) - 3   ( 50 50 100 ) - 2 ( 25 25 100 ) - 3 ( 100 100 50 ) - 2 ( 100 100 25 )

*/

int main(int argc, char* argv[]) {

    // Check if the correct number of arguments is provided
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <particle> <size_x> <size_y> <size_z> <closePeak>" << std::endl;
        return 1;
    }

    // Retrieve and store the particle type from the first argument
    std::string particle = argv[1];
    
    // Convert the second, third, and fourth arguments to integers for the grid size
    int size_x = std::stoi(argv[2]);
    int size_y = std::stoi(argv[3]);
    int size_z = std::stoi(argv[4]);

    int rad = std::stoi(argv[5]);

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

    folderPath += "/topPeaks/";

    // Try to create the folder
    if (createDirectory(folderPath) == 0) {
        std::cout << "Directory created successfully: " << folderPath << std::endl;
    } else {
        std::cout << "Directory already exists or couldn't be created: " << folderPath << std::endl;
    }

    // Call a function that fills the table with the given particle type
    fillTable(particle,{size_x,size_y,size_z},rad,folderPath);

}
