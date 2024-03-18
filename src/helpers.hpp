#pragma once
#include <vector>
#include <iostream>
#include <map>
#include <fstream>


const uint32_t LABEL_MAGIC_NUMBER = 0x801;

// takes a path to a file and returns the number of items in the file
static size_t getItemCount(const std::string& path) {
    std::ifstream input_file(path, std::ios::binary);
    assert(input_file.is_open());

    uint32_t magic_number, num_items;
    input_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);
    input_file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
    num_items = __builtin_bswap32(num_items);

    assert(magic_number == LABEL_MAGIC_NUMBER);

    return num_items;
}

static void logPrediction(int prediction, int label, int image_index, const std::string& filename) {
    // Open the file in append mode
    std::ofstream file(filename, std::ios::app);
    if (!file) {
        std::cerr << "Unable to open file for writing.\n";
        return;
    }

    file << " - image " << image_index << ": Prediction=" << prediction << ". Label=" << label << "\n";

    file.close();
}

static std::map<std::string, std::string> parseConfigfile(std::ifstream& configfile) {
    std::string line;
    std::map<std::string, std::string> config;
    while(std::getline(configfile, line)) {
        if(line.empty() || line[0] == '\\') {
            continue; // skip empty lines
        }
        std::string key, value;
        size_t equalPosition = line.find('=');
        if(equalPosition == std::string::npos) {
            continue; // ignore lines without =
        }
        key = line.substr(0, equalPosition);
        value = line.substr(equalPosition + 1);

        // remove all leading and trailing whitespace
        key.erase(0, key.find_first_not_of(" \t\r\n"));
        key.erase(key.find_last_not_of(" \t\r\n") + 1);
        value.erase(0, value.find_first_not_of(" \t\r\n"));
        value.erase(value.find_last_not_of(" \t\r\n") + 1);

        config[key] = value;
    }

    return config;
}
