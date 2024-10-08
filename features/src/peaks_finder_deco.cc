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
 
#include "../include/utils.h" // Custom utility functions or classes
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

// Parameter for peak detection, used in spectrum analysis (for identifying nearby peaks)
int close_peaks_radius_param = 3; // 100 100 100 = 5 ; 50 50 100 = 3

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


std::vector<double> findPeaks_withDeco(TString filePath, int eventNum, std::vector<int> size_cell)
{
    // Open the input ROOT file using the provided file path
    TFile* inputFile = TFile::Open(filePath); 

    // Get the "outputTree" from the file and cast it to a TTree object
    TTree* Tree = dynamic_cast<TTree*>(inputFile->Get("outputTree"));

    // Set branch addresses to point to the variables that will hold the tree data
    Tree->SetBranchAddress("Tinteractions_in_event", &Tentries);  // Number of interactions in the event
    Tree->SetBranchAddress("Tcublet_idx", &Tcublet_idx);          // Cublet indices for each interaction
    Tree->SetBranchAddress("Tcell_idx", &Tcell_idx);              // Cell indices for each interaction
    Tree->SetBranchAddress("Tedep", &Tedep);                      // Energy deposition for each interaction

    // Create 2D histograms for energy deposition in ZY and ZX planes
    TH2D *hist_cell_zy = new TH2D("zy", "", size_cell[2], 0, size_cell[2], size_cell[1], 0, size_cell[1]);
    TH2D *hist_cell_zx = new TH2D("zx", "", size_cell[2], 0, size_cell[2], size_cell[0], 0, size_cell[0]);

    // Load the specified event (eventNum) from the tree
    Tree->GetEntry(eventNum);

    // Loop through all interactions within the event
    for (size_t j = 0; j < Tentries; j++) 
    {
        // Retrieve the cublet and cell indices for the current interaction
        int cub_idx = (*Tcublet_idx)[j];
        int cell_idx = (*Tcell_idx)[j];
        
        // Retrieve the energy deposited in this interaction
        double E = (*Tedep)[j]; 

        // Convert the cublet and cell indices to 3D position coordinates
        std::vector<int> int_pos = convertPos(cub_idx, cell_idx, size_cell);

        // Fill the 2D histograms with energy deposition data for ZY and ZX planes
        hist_cell_zy->Fill(int_pos[2], int_pos[1], E);  // Fill histogram for ZY plane
        hist_cell_zx->Fill(int_pos[2], int_pos[0], E);  // Fill histogram for ZX plane
    }

    // Parameters for spectrum analysis (used for detecting peaks in the Z axis projection)
    const Int_t nbins = size_cell[2];   // Number of bins based on Z-axis size
    Double_t xmin = 0;                  // Minimum value of the histogram range
    Double_t xmax = nbins;              // Maximum value of the histogram range
    Double_t source[nbins], dest[nbins]; // Arrays to store histogram contents

    // Create a 1D projection of the ZY histogram onto the Z axis (collapsing Y)
    TH1D *projZ = hist_cell_zy->ProjectionX("projZ");
    for (int i = 0; i < nbins; i++) 
        source[i] = projZ->GetBinContent(i + 1);  // Copy bin content to the source array

    int res = 100;  // Resolution parameter for peak detection

    // Perform high-resolution peak search using TSpectrum class
    TSpectrum *spectrumZ = new TSpectrum(20, res);  // Initialize spectrum with up to 20 peaks

    // Set parameters for Gaussian smoothing and deconvolution
    double sigma = size_cell[2] / 20;  // Standard deviation for smoothing
    int ndeconv = 10;                  // Number of deconvolution iterations
    int window = 2;                    // Window size for peak search
    double threshold = 1.5;            // Threshold for peak detection

    // Perform deconvolution of the source data and store the result in the dest array
    int nfoundZ = spectrumZ->SearchHighRes(source, dest, nbins, sigma, threshold, kFALSE, ndeconv, kFALSE, window);

    // Update the projection histogram with the deconvolved data
    for (int i = 0; i < nbins; i++) 
        projZ->SetBinContent(i + 1, dest[i]);

    // Perform the final peak search on the deconvolved histogram
    nfoundZ = spectrumZ->Search(projZ, 3, "nodraw", 0.06);

    // Retrieve the positions of the detected peaks along the Z axis
    Double_t* xpeaksZ = spectrumZ->GetPositionX();

    // Store the peak positions in a vector and return it
    std::vector<double> peaksZ(xpeaksZ, xpeaksZ + nfoundZ);

    // Clean up dynamically allocated memory
    delete projZ;
    delete spectrumZ;

    // Initialize output vector with a size of 14, filled with -1 values
    std::vector<double> out_vector(14, -1.);
    if (nfoundZ > 0) 
    {
        // Sort detected Z peaks
        std::sort(peaksZ.begin(), peaksZ.end());

        // Get number of bins for X and Y axis projections from the histograms
        int nBinsX_y = hist_cell_zy->GetNbinsX();
        int nBinsY_y = hist_cell_zy->GetNbinsY();
        int nBinsX_x = hist_cell_zx->GetNbinsX();
        int nBinsY_x = hist_cell_zx->GetNbinsY();

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
        double threshold = 0;

        // Loop through each Z peak
        for (size_t s = 0; s < num_peaks; s++) 
        {    
            threshold = peaksZ[s];  // Current Z threshold

            // Filter the ZX plane histogram for values around the current Z peak
            TH2D* h2d_filtered_x = new TH2D("FILTER_X", "", size_cell[2], 0, size_cell[2], size_cell[0], 0, size_cell[0]);
            for (int i = 1; i <= nBinsX_x; ++i) {
                double x = hist_cell_zx->GetXaxis()->GetBinCenter(i);

                // Select bins within the threshold range
                if (x > threshold - 2 && x < threshold + 2) {
                    for (int j = 1; j <= nBinsY_x; ++j) {
                        double content = hist_cell_zx->GetBinContent(i, j);
                        h2d_filtered_x->SetBinContent(i, j, content);

                        double error = hist_cell_zx->GetBinError(i, j);
                        h2d_filtered_x->SetBinError(i, j, error);
                    }
                }
            }

            // Project the filtered ZX histogram onto the Y-axis
            std::string hist_name_x = "proj_x_" + std::to_string(s);
            projX[s] = h2d_filtered_x->ProjectionY(hist_name_x.c_str());

            // Perform peak search on the projection
            for (int i = 0; i < nbins; i++) source[i] = projX[s]->GetBinContent(i + 1);
            TSpectrum *spectrumX = new TSpectrum(20, res);
            Int_t nfoundX = spectrumX->SearchHighRes(source, dest, nbins, sigma, threshold, kFALSE, ndeconv, kFALSE, window);
            for (int i = 0; i < nbins; i++) projX[s]->SetBinContent(i + 1, dest[i]);

            double x_thr = 0.1;
            int x_sm = 3;
            nfoundX = spectrumX->Search(projX[s], x_sm, "nodraw", x_thr);

            
            // Adjust search parameters if no peaks found
            int iter = 0;
            while (!nfoundX) {
                if (x_thr < 0.01) {
                    x_thr = 0.1;
                    x_sm = 2;
                } else {
                    x_thr /= 2;
                }

                nfoundX = spectrumX->Search(projX[s], x_sm, "nodraw", x_thr);
                iter += 1;

                if (iter == 5)
                {   
                    for (int i = 0; i < nbins; i++) projX[s]->SetBinContent(i + 1, source[i]);

                    x_thr = 0.1;
                    x_sm = 1;
                    nfoundX = spectrumX->Search(projX[s], x_sm, "nodraw", x_thr);
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
                double x = hist_cell_zy->GetXaxis()->GetBinCenter(i);

                // Select bins within the threshold range
                if (x > threshold - 2 && x < threshold + 2) {
                    for (int j = 1; j <= nBinsY_y; ++j) {
                        double content = hist_cell_zy->GetBinContent(i, j);
                        h2d_filtered_y->SetBinContent(i, j, content);

                        double error = hist_cell_zy->GetBinError(i, j);
                        h2d_filtered_y->SetBinError(i, j, error);
                    }
                }
            }

            // Project the filtered ZY histogram onto the Y-axis
            std::string hist_name_y = "proj_y_" + std::to_string(s);
            projY[s] = h2d_filtered_y->ProjectionY(hist_name_y.c_str());

            // Perform peak search on the Y projection
            for (int i = 0; i < nbins; i++) source[i] = projY[s]->GetBinContent(i + 1);
            TSpectrum *spectrumY = new TSpectrum(20, res);
            int nfoundY = spectrumY->SearchHighRes(source, dest, nbins, sigma, threshold, kFALSE, ndeconv, kFALSE, window);
            for (int i = 0; i < nbins; i++) projY[s]->SetBinContent(i + 1, dest[i]);

            // Adjust search parameters if no peaks found
            double y_thr = 0.1;
            int y_sm = 3;
            nfoundY = spectrumY->Search(projY[s], y_sm, "nodraw", y_thr);

            iter = 0;
            while (!nfoundY) {
                if (y_thr < 0.01) {
                    y_thr = 0.1;
                    y_sm = 2;
                } else {
                    y_thr /= 2;
                }
                
                nfoundY = spectrumY->Search(projY[s], y_sm, "nodraw", y_thr);
                iter += 1;

                if (iter > 5)
                {   
                    for (int i = 0; i < nbins; i++) projY[s]->SetBinContent(i + 1, source[i]);

                    y_thr = 0.1;
                    y_sm = 1;
                    nfoundY = spectrumY->Search(projY[s], y_sm, "nodraw", y_thr);
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
            out_peaks.push_back({peak_x, peak_y, threshold});

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

        // Store sorted peaks and corresponding energy values
        std::vector<std::vector<double>> sorted_peaks;
        for (size_t g = 0; g < nfoundZ; g++) {
            sorted_peaks.push_back(out_peaks[energy_index[g]]);
        }

        // Calculate distances and angles between peaks
        std::vector<std::vector<double>> d_and_angle = calculate_distance_and_angle(sorted_peaks);

        // Store sorted energies and energy ratios
        std::vector<double> sorted_energies;
        std::vector<double> sorted_ratios;
        for (size_t g = 0; g < nfoundZ; g++) {
            sorted_energies.push_back(energy_aroundPeaks[energy_index[g]]);
            sorted_ratios.push_back(energy_aroundPeaks[energy_index[g]] / totEn);
        }

        // Populate the out_vector with the results
        if(nfoundZ>2)
        {

            for(size_t k =0; k<3;k++)
            {
                out_vector[0+4*k] = sorted_energies[k];
                out_vector[1+4*k] = d_and_angle[k][1];
                out_vector[2+4*k] = d_and_angle[k][0];
                out_vector[3+4*k] = sorted_ratios[k];

            } 

            std::vector<std::vector<double>> topThreeVertices = {sorted_peaks[0],sorted_peaks[1],sorted_peaks[2]};
            double aplanarity = angleBetweenLineAndPlane(topThreeVertices);
            out_vector[12] = aplanarity;
            out_vector[13] = nfoundZ;
        }
        else
        {   

            for(size_t k =0; k<nfoundZ;k++)
            {
                out_vector[0+4*k] = sorted_energies[k];
                out_vector[1+4*k] = d_and_angle[k][1];
                out_vector[2+4*k] = d_and_angle[k][0];
                out_vector[3+4*k] = sorted_ratios[k];

            } 

            out_vector[12] = -1.; // Aplanarity is undefined for fewer than 3 peaks
            out_vector[13] = nfoundZ;

        }

        for (size_t i = 0; i < num_peaks; i++) {
            if (projY[i]) delete projY[i];
            if (projX[i]) delete projX[i];
        }

    }

    inputFile->Close();
    delete inputFile;

    return out_vector;

}

void fillTable(std::string particleName,std::vector<int> size_cell = {100,100,100})
{

    string outFile =  "./results_" + std::to_string(size_cell[0]) + "_" + std::to_string(size_cell[1]) + "_" + std::to_string(size_cell[2]) + "/" + particleName + ".tsv";
    std::ofstream oFile(outFile, std::ios::out);

    oFile << "FileName\t";
    oFile << "EventNum\t";
    oFile << "NumPeaks\t";

    oFile << "E1\t";
    oFile << "R1\t";

    oFile << "E2\t";
    oFile << "theta2\t";
    oFile << "d2\t";
    oFile << "R2\t";

    oFile << "E3\t";
    oFile << "theta3\t";
    oFile << "d3\t";
    oFile << "R3\t";
    
    oFile << "Aplanarity";

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
                    std::vector<double> info = findPeaks_withDeco(fileName,i,size_cell);

                        
                    oFile << file->GetName() << "\t" << i << "\t" << info[13] << "\t"
                                    << info[0] <<  "\t" << info[3]  <<  "\t"
                                    << info[4] << "\t" << info[5] <<  "\t" << info[6]  <<  "\t" << info[7]  <<  "\t" 
                                    << info[8]  <<  "\t" << info[9] << "\t" << info[10] <<  "\t" << info[11]  <<  "\t" 
                                    << info[12]  << std::endl;

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
