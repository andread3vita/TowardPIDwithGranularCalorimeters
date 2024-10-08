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

// TCanvas *c1 = new TCanvas();

int Tentries;

vector<int>* Tcublet_idx;
vector<int>* Tcell_idx;
vector<double>* Ttime;
vector<double>* Tedep;
vector<double>* Tpdg;
vector<double>* Tdeltae;

#define CURSOR_TO_START "\033[1G"
#define CLEAR_LINE "\033[K"

std::vector<int> isVertex(TString filePath,int eventNum,std::string particleName)
{

    TFile* inputFile = TFile::Open(filePath); 
    TTree* Tree = dynamic_cast<TTree*>(inputFile->Get("outputTree"));
    
    Tree->SetBranchAddress("Tinteractions_in_event", &Tentries);
            
    Tree->SetBranchAddress("Tcublet_idx", &Tcublet_idx);
    Tree->SetBranchAddress("Tcell_idx", &Tcell_idx);
    Tree->SetBranchAddress("Tedep", &Tedep);

    Tree->SetBranchAddress("Tdeltae", &Tdeltae);
    Tree->SetBranchAddress("Tpdg", &Tpdg);

    Tree->GetEntry(eventNum);

    int wanted = 0;
    if (particleName == "proton")
    {
        wanted = 2212;
    }
    else if (particleName == "pion")
    {
        wanted = 211;
    }
    else if (particleName == "kaon")
    {
        wanted = 321;
    }
    else
    {
        return {-1,-1};
    }

    int isVertex_flag = -1;
    bool peak = false;
    std::vector<int> true_peak_pos = {-1,-1,-1};
    for (size_t j = 0; j < Tentries; j++) 
    {
        
        int cub_idx = (*Tcublet_idx)[j];
        int cell_idx =(*Tcell_idx)[j];
        double E = (*Tedep)[j]; 

        std::vector<int> int_pos = convertPos(cub_idx,cell_idx);

        double deltaE = (*Tdeltae)[j];
        int pdg = (*Tpdg)[j];

        if ((deltaE < -50e3)&&(pdg==wanted)&&!peak)
        {
            peak = true;
            true_peak_pos = int_pos;
        }

                    
    }
    if(peak)
    {
        isVertex_flag = 1;
    }

    

    inputFile->Close();
    delete inputFile;

    return {isVertex_flag,true_peak_pos[2]};

}

void fillTable(std::string particleName)
{

    string outFile =  "./results/" + particleName + ".tsv";
    std::ofstream oFile(outFile, std::ios::out);

    oFile << "FileName\t";
    oFile << "EventNum\t";
    oFile << "isVertex\t";
    oFile << "trueZvertex";

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
                    

                    std::vector<int> info = isVertex(fileName,i,particleName);

                    oFile << file->GetName() << "\t" << i << "\t" << info[0] << "\t" << info[1] << std::endl;

                    std::cout << CURSOR_TO_START << CLEAR_LINE;
                                
                }

            }
        }                   
    }

    oFile.close();
}

int main(int argc, char* argv[]) {

    std::string particle = argv[1];
    fillTable(particle);

}