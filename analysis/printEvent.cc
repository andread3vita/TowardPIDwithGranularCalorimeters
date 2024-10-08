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
 
using namespace std;

int Tentries;

vector<int>* Tcublet_idx;
vector<int>* Tcell_idx;
vector<double>* Ttime;
vector<double>* Tedep;
vector<double>* Tpdg;
vector<double>* Tdeltae;

std::vector<int> convertPos(int cub_idx, int cell_idx)
{
    int layer_cublet_idx = cub_idx%100;
    int x_cub = layer_cublet_idx % 10;
    int y_cub = 9-layer_cublet_idx/10;
    int z_cub = cub_idx/100;
        
    int layer_cell_idx = cell_idx%100;
    int x_cell_temp = layer_cell_idx% 10;
    int y_cell_temp = 9-layer_cell_idx/10;
    int z_cell_temp = cell_idx/100;

    int x_cell = x_cub*10 + x_cell_temp;
    int y_cell = y_cub*10 + y_cell_temp;
    int z_cell = z_cub*10 + z_cell_temp;  

    std::vector<int> vec = {x_cell,y_cell,z_cell};
    return vec;
}

void SingleEventAnalysis(std::string particleName,int fileNum, int eventNum)
{

    TH2F *hist_cell_xy= new TH2F("histoxy", "Cell-wise", 100, 0, 100, 100, 0, 100);
    TH2F *hist_cell_zy= new TH2F("histozy", "Cell-wise", 100, 0, 100, 100, 0, 100);

    TString dirPath;
    if (particleName == "kaon")
    {
        dirPath = "/lustre/cmsdata/optCalData/kaon/kaon_";
        dirPath += std::to_string(fileNum);
        dirPath += ".root";

    }
    else if(particleName == "pion")
    {
        dirPath = "/lustre/cmsdata/optCalData/pion/pion_";
        dirPath += std::to_string(fileNum);
        dirPath += ".root";
  
    }
    else if(particleName=="proton")
    {
    
        dirPath = "/lustre/cmsdata/optCalData/proton/proton_";
        dirPath += std::to_string(fileNum);
        dirPath += ".root";
    }
    else
    {
      
        std::cerr << "Invalid particle name!!"<<std::endl;
        std::cerr << "Valid particles: kaon , pion , proton " <<std::endl;

        return;
    }

    //Get the tree "Edep" and set branch addresses
    TFile* inputFile = TFile::Open(dirPath); 
    TTree* Tree = dynamic_cast<TTree*>(inputFile->Get("outputTree"));

    // Create a vector of integers
    std::vector<double> pos;
    for (int i = 0; i < 100; ++i) {
        pos.push_back(i*1.);
    }
    
    // Set branch addresses
    Tree->SetBranchAddress("Tinteractions_in_event", &Tentries);
            
    Tree->SetBranchAddress("Tcublet_idx", &Tcublet_idx);
    Tree->SetBranchAddress("Tcell_idx", &Tcell_idx);
    Tree->SetBranchAddress("Tedep", &Tedep);

    std::vector<double>en_position(100,0);
    Tree->GetEntry(eventNum);
    for (size_t j = 0; j < Tentries; j++) {
                            
        double E = (*Tedep)[j]; 

        int cub_idx = (*Tcublet_idx)[j];
        int cell_idx =(*Tcell_idx)[j];

        std::vector<int> int_pos = convertPos(cub_idx,cell_idx);   

        hist_cell_zy->Fill(int_pos[2], int_pos[1], E);   
        hist_cell_xy->Fill(int_pos[0], int_pos[1], E);   

        en_position[int_pos[2]] +=E;
                      
    }

    std::vector<double> cumulative_energy(100,0);
    cumulative_energy[0] = en_position[0];
    for(size_t j = 1; j < 100; j++)
    {
        cumulative_energy[j] = cumulative_energy[j-1]+ en_position[j];
    }
    for(size_t j = 0; j < 100; j++)
    {
        cumulative_energy[j] /= cumulative_energy[99];
    }

    for(size_t j = 0; j < 100; j++)
    {
        std::cout << en_position[j] << "\t" << cumulative_energy[j] << std::endl;
    }
    

    TH1F *hist = new TH1F("hist", "Energy profile", 100, 0, 100);
    TH1F *hist_cumul = new TH1F("hist_cum", "Cumulative Energy profile", 100, 0, 100);

    for (size_t i = 0; i<100;i++)
    {
        hist->Fill(i,en_position[i]);
    }

    for (size_t i = 0; i<100;i++)
    {
        hist_cumul->Fill(i,cumulative_energy[i]);
    }


    TCanvas* canvas = new TCanvas("canvas", "Scatter Plot", 2000, 2000);
    canvas->Divide(2, 2);

    canvas->cd(1); 

    hist_cell_xy->Draw("COLZ");
    hist_cell_xy->GetXaxis()->SetTitle("x [a.u.]");
    hist_cell_xy->GetYaxis()->SetTitle("y [a.u.]");
    gPad->SetLogz();
    // gStyle->SetPalette(kLightTemperature);
    hist_cell_xy->SetStats(kFALSE);

    canvas->cd(2); 

    hist_cell_zy->Draw("COLZ");
    hist_cell_zy->GetXaxis()->SetTitle("z [a.u.]");
    hist_cell_zy->GetYaxis()->SetTitle("y [a.u.]");
    gPad->SetLogz();
    // gStyle->SetPalette(kLightTemperature);
    hist_cell_zy->SetStats(kFALSE);

    canvas->cd(3);
    hist->GetXaxis()->SetTitle("z [a.u.]");
    hist->GetYaxis()->SetTitle("Energy [MeV]");
    hist->SetStats(kFALSE);
    hist->SetTitleOffset(1.5, "Y");
    hist->SetFillStyle(3021);
    hist->SetFillColorAlpha(kBlue,0.8);
    hist->Draw("HIST");

    canvas->cd(4);
    hist_cumul->GetXaxis()->SetTitle("z [a.u.]");
    hist_cumul->GetYaxis()->SetTitle("Energy fraction");
    hist_cumul->SetStats(kFALSE);
    hist_cumul->SetTitleOffset(1.5, "Y");
    hist_cumul->SetFillStyle(3021);
    hist_cumul->SetFillColorAlpha(kBlue,0.8);
    hist_cumul->Draw("HIST");

    // Draw a horizontal red line at y = 0.5
    TLine *line = new TLine(0, 0.5, 100, 0.5); // Line from x = 0 to x = 100 at y = 0.5
    line->SetLineColor(kRed); // Set the line color to red
    line->SetLineWidth(2); // Optionally set the line width
    line->Draw("same"); // Draw the line on the same canvas

    canvas->Update();
    canvas->Draw();

    TString fileOut= "/lustre/cmswork/adevita/calOpt/features/EventAnalysis_";
    fileOut += particleName;
    fileOut += "_";
    fileOut += fileNum;
    fileOut += "_";
    fileOut += eventNum;
    fileOut += ".png";
    canvas->SaveAs(fileOut);

    delete canvas;

}