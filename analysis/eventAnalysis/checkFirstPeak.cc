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
#include <regex>
 
using namespace std;

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

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

std::pair<double, double> residuals(std::string filename_check)
{
    
    
    // Regular expression to extract the number after "proton"
    std::regex pattern("proton(\\d+(\\.\\d+)?)");

    // Variable to store the match
    std::smatch match;

    // Search for the pattern in the filename
    std::regex_search(filename_check, match, pattern) ;
    // The number is in the first capture group
    std::string extracted_value = match[1];
   
    std::string filename_ref="/lustre/cmswork/aabhishe/TowardPIDwithGranularCalorimeters/features/results/trueVertex/results_100_100_25/proton.tsv";
    int columnIndex1 = getColumnIndex(filename_ref,"trueZvertex");
    int columnIndex2 = getColumnIndex(filename_check, "Z_vertex");

    int columnIndex_check = getColumnIndex(filename_ref,"isVertex");
    vector<double> columnData_check = readColumn(filename_ref, columnIndex_check);

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

            if (abs(columnData1[i] - columnData2[i])<2)
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
    std::cout << "Number of events with diff < 2: " << count << "/" << true_diff.size() << " = " << (count*1.)/(true_diff.size()*1.)*100 << "%" << std::endl;
    double a=(count_0*1.)/(true_diff.size()*1.)*100;
    double b = (count*1.)/(true_diff.size()*1.)*100;
    std::ofstream csv_file("threshold_"+extracted_value+".csv");
    if (csv_file.is_open()) {
        csv_file << "Condition,Count,Total,Percentage\n"; // Header row
        csv_file << "diff == 0," << count_0 << "," << true_diff.size() << "," 
                 << (count_0 * 1.0) / (true_diff.size() * 1.0) * 100 << "%\n";
        csv_file << "diff < 2," << count << "," << true_diff.size() << "," 
                 << (count * 1.0) / (true_diff.size() * 1.0) * 100 << "%\n";
        csv_file.close();
        std::cout << "Results saved to output.csv" << std::endl;
    } 


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

 return std::make_pair(a, b);
}

void checkFirstPeak() {
    std::string path_main = "/lustre/cmswork/aabhishe/exp_scripts/results/results_50_50_25/firstVertex/Smearing/";

    // Paths to the input files
    std::string paths[] = {
        path_main + "proton150.000000.tsv",
        path_main + "proton180.000000.tsv",
        path_main + "proton200.000000.tsv",
        path_main + "proton220.000000.tsv",
        path_main + "proton250.000000.tsv",
        path_main + "proton300.000000.tsv",
        path_main + "proton350.000000.tsv",
        path_main + "proton380.000000.tsv"
    };

    // Thresholds (x-axis)
    double thresholds[] = {150,180,200, 220, 250, 300, 350, 380};

    // Vectors to store results for the two graphs
    std::vector<double> count1;
    std::vector<double> count2;

    // Loop through each file and compute residuals
    for (int i = 0; i < 6; i++) {
        std::pair<double, double> result = residuals(paths[i]);
        count1.push_back(result.first);
        count2.push_back(result.second);
    }

    // Create ROOT graphs
    TGraph* graph1 = new TGraph(6, thresholds, count1.data());
    TGraph* graph2 = new TGraph(6, thresholds, count2.data());

    // Customize the first graph
    graph1->SetTitle("50_50_25;Threshold (MeV); Accuracy (%)");
    graph1->SetMarkerStyle(20);
    graph1->SetMarkerColor(kBlue);
    graph1->SetLineColor(kBlue);

    // Customize the second graph
    graph2->SetTitle("Graph 2;Threshold (MeV);Accuracy (%)");
    graph2->SetMarkerStyle(21);
    graph2->SetMarkerColor(kRed);
    graph2->SetLineColor(kRed);

    double y_min = 80;  // Minimum value for the Y-axis
    double y_max = 95; // Maximum value for the Y-axis
    graph1->GetYaxis()->SetRangeUser(y_min, y_max);
    // Draw the graphs on a canvas
    TCanvas* c = new TCanvas("c", "Threshold vs Percentage", 800, 600);
    c->SetGrid(1);
    graph1->Draw("APL"); // Draw the first graph with Axes, Points, and Lines
    graph2->Draw("LP"); // Overlay the second graph

    // Add a legend
    auto legend = new TLegend(0.1, 0.7, 0.2, 0.8);
    legend->AddEntry(graph1, "Count1", "lp");
    legend->AddEntry(graph2, "Count2", "lp");
    legend->Draw();

    // Save the canvas as a file
    c->SaveAs("threshold_vs_percentage.png");
}
