#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/stat.h>

class ArgsFileManager {
public:
    ArgsFileManager() {}

    void generateArgsFile(const std::string& executableName, 
                          const std::vector<std::string>& params, 
                          const std::vector<int>& segmentationValues, 
                          const std::string& defaultValue = "ND")
    {   
        
        std::string argsFilePath = args_dir + "/" + executableName + ".args";

        // Check if the file already exists
        struct stat buffer;
        if (stat(argsFilePath.c_str(), &buffer) == 0) {
            std::cout << "File " << argsFilePath << " already exists. Skipping file creation.\n";
            return;  // Skip file creation if the file already exists
        }

        std::ofstream outfile(args_dir + "/" + executableName + ".args");
        outfile << "segmentation";
        for (const auto& param : params) {
            outfile << "\t" << param;
        }
        outfile << "\n";

        for (auto& seg_x : segmentationValues) {
            for (auto& seg_z : segmentationValues) {
                
                std::ostringstream segmentation;
                segmentation << seg_x << "_" << seg_x << "_" << seg_z;

                outfile << segmentation.str();
                for (size_t i = 0; i < params.size(); ++i) {
                    outfile << "\t" << defaultValue;
                }
                outfile << "\n";
            
            }
        }

        outfile.close();
        std::cout << "File generated: " << args_dir + "/" + executableName + ".args" << "\n";
    }

    void updateArgsRow(const std::string& executableName, 
                       const std::string& segmentation, 
                       const std::vector<std::string>& values) 
    {
        std::string argsFile = args_dir + "/" + executableName + ".args";

        std::ifstream infile(argsFile);
    
        std::string header;
        std::getline(infile, header);

        std::vector<std::string> lines;
        lines.push_back(header);

        std::string line;
        bool updated = false;

        while (std::getline(infile, line)) {
            auto tokens = split(line, '\t');
            if (!tokens.empty() && tokens[0] == segmentation) {
                // Update the row
                std::string updatedLine = segmentation + "\t" + join(values, "\t");
                lines.push_back(updatedLine);
                updated = true;
            } else {
                lines.push_back(line);
            }
        }

        infile.close();

        std::ofstream outfile(argsFile);
        for (const auto& l : lines) {
            outfile << l << "\n";
        }
        outfile.close();
        std::cout << "Segmentation updated: " << segmentation << "\n";
    }

private:
    std::string args_dir = "args"; // Directory of args files

    // Helper to split a string
    std::vector<std::string> split(const std::string& str, char delimiter) const {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }

    // Helper to join the elements into a string
    std::string join(const std::vector<std::string>& elements, const std::string& delimiter) const {
        std::ostringstream os;
        for (size_t i = 0; i < elements.size(); ++i) {
            os << elements[i];
            if (i != elements.size() - 1) os << delimiter;
        }
        return os.str();
    }
};

// Main function to handle user input
int main() {
    try {
        std::string executableName;
        std::cout << "Enter the name of the executable: ";
        std::cin >> executableName;

        // Select parameters based on the executable name
        std::vector<std::string> params;
        if (executableName == "firstVertex") {
            params = {"threshold", "closeVertex","xyWindow","zWindow","smearing"};
        }
        else if (executableName == "generalFeature") {
            params = {"Efraction", "smearing"};
        }
        else if (executableName == "missingEnergy") {
            std::cout << "It has no configurable parameters.\n";

            return 1;
        }
        else if (executableName == "spatialObservables") {
            std::cout << "It has no configurable parameters.\n";

            return 1;
        }
        else if (executableName == "speed") {
            params = {"threshold","xyWindow","zWindow","smearing"};
        }
        else if (executableName == "topPeaks") {
            params = {"closePeak"};
        }
        else {
            std::cout << "Executable not recognized.\n";
            return 1;  // Exit if executable is not recognized
        }

        std::cout << "Here are the executable arguments: ";
        for (auto& param : params)
        {
            std::cout << param << " , ";
        }
        std::cout << " " << std::endl;

        // Ask the user to input segmentation values
        std::vector<int> segmentationValues = {5, 10, 20, 25, 50, 100}; // Possible values

        // Create an instance of ArgsFileManager
        ArgsFileManager manager;

        // Generate the initial args file
        manager.generateArgsFile(executableName, params, segmentationValues);

        std::string segmentation;
        bool continueInput = true;

        while (continueInput) {
            std::cout << "Enter the segmentation to update (format x_y_z): ";
            std::cin >> segmentation;

            if (segmentation.empty()) {
                continueInput = false;
                break;
            }

            // Ask for the values to update for the segmentation
            std::vector<std::string> values;
            std::string value;
            std::cout << "Enter the values to associate with the segmentation (separated by space): ";
            while (std::cin >> value) {
                values.push_back(value);
                if (std::cin.peek() == '\n') break;
            }

            // Update the row for that segmentation
            manager.updateArgsRow(executableName, segmentation, values);

            // Ask if the user wants to continue
            std::string continueResponse;
            std::cout << "Do you want to update another segmentation? (y/N): ";
            std::cin >> continueResponse;

            if (continueResponse != "y") {
                continueInput = false;
            }
        }

        std::cout << "Operation completed!" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}
