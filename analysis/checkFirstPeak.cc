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

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

// Funzione per ottenere l'indice di una colonna dato il nome
int getColumnIndex(const string &filename, const string &columnName) {
    ifstream file(filename);
    string headerLine;
    
    if (getline(file, headerLine)) {
        stringstream ss(headerLine);
        string column;
        int index = 0;

        while (getline(ss, column, '\t')) {
            if (column == columnName) {
                return index;
            }
            index++;
        }
    }
    cerr << "Errore: Colonna \"" << columnName << "\" non trovata in " << filename << endl;
    return -1;
}

// Funzione per leggere una colonna specifica da un file .tsv dato l'indice
vector<double> readColumn(const string &filename, int columnIndex) {
    ifstream file(filename);
    vector<double> columnData;
    string line;

    // Salta l'header
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        int col = 0;

        while (getline(ss, value, '\t')) {
            if (col == columnIndex) {
                columnData.push_back(stod(value));
                break;
            }
            col++;
        }
    }
    return columnData;
}

void ResidualsVertex(std::string filename_ref,std::string filename_check)
{

    // Trova l'indice della colonna per entrambi i file
    int columnIndex1 = getColumnIndex(filename_ref,"trueZvertex");
    int columnIndex2 = getColumnIndex(filename_check, "Z_vertex");

    int columnIndex_check = getColumnIndex(filename_ref,"isVertex");
    vector<double> columnData_check = readColumn(filename_ref, columnIndex_check);

    // Leggi le colonne dai file
    vector<double> columnData1 = readColumn(filename_ref, columnIndex1);
    vector<double> columnData2 = readColumn(filename_check, columnIndex2);

    std::vector<double> full_diff;
    std::vector<double> true_diff;

    int count = 0;
    int count_0 = 0;
    for (size_t i = 0; i < columnData1.size(); i++) {

        if (columnData_check[i] > 0)
        {
            true_diff.push_back(columnData1[i] - columnData2[i]);

            if (abs(columnData1[i] - columnData2[i])>2)
            {
                count +=1;
            }

            if (abs(columnData1[i] - columnData2[i])== 0)
            {
                count_0 += 1;
            }

        }

        full_diff.push_back(columnData1[i] - columnData2[i]);

    }

    std::cout << "Number of events with diff == 0: " << count_0 << "/" << true_diff.size() << " = " << (count_0*1.)/(true_diff.size()*1.)*100 << "%" << std::endl;
    std::cout << "Number of events with diff > 2: " << count << "/" << true_diff.size() << " = " << (count*1.)/(true_diff.size()*1.)*100 << "%" << std::endl;
    TH1D *hist = new TH1D("hist", "With -1 events", 200, -100, 100);
    for (double value : full_diff) {
        hist->Fill(value);
    }
    TH1D *hist_true = new TH1D("hist_t", "Without -1 events", 200, -100, 100);
    for (double value : true_diff) {
        hist_true->Fill(value);
    }
    
    TCanvas* canvas = new TCanvas("canvas", "Scatter Plot", 1100, 800);
    canvas->Divide(2, 1);

    canvas->cd(1); 

    hist_true->GetXaxis()->SetTitle("diff");
    hist_true->GetYaxis()->SetTitle("Counts");
    hist_true->SetStats(kFALSE);
    gPad->SetLogy();
    hist_true->Draw();

    canvas->cd(2); 

    hist->GetXaxis()->SetTitle("diff");
    hist->GetYaxis()->SetTitle("Counts");
    hist->SetStats(kFALSE);
    gPad->SetLogy();
    hist->Draw();

    canvas->Update();
    canvas->Draw();

    TString fileOut= "/lustre/cmswork/adevita/calOpt/features/firstPeak_performance.png";
    canvas->SaveAs(fileOut);

    delete hist;
    delete hist_true;
    delete canvas;

}