#include "../tensor.hpp"
#include <fstream>
#include <iostream>
#include <string>

const uint LABEL_HEADER_SIZE = 8;
const uint MAGIC_NUMBER_LABELS = 0x801;
const uint TENSOR_SIZE = 10;

template<typename T>
class IOlabel {
private:
    std::string label_dataset_input;
    int label_index;

public:
    IOlabel(const std::string& dataset_input, int index)
            : label_dataset_input(dataset_input), label_index(index) {}

    std::vector<double> extractLabel() {
        std::ifstream input_file(label_dataset_input, std::ios::binary);

        if(!input_file.is_open()) {
            throw std::runtime_error("File open failed: " + label_dataset_input);
        }

        // Read the MNIST Header
        uint32_t magic_number, num_items;
        input_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number);
        input_file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
        num_items = __builtin_bswap32(num_items);

        if(magic_number != MAGIC_NUMBER_LABELS) {
            throw std::runtime_error("Not a MNIST label data file");
        }

        input_file.seekg(LABEL_HEADER_SIZE + label_index);

        uint8_t label;
        input_file.read(reinterpret_cast<char*>(&label), sizeof(label));

        std::vector<double> label_data(TENSOR_SIZE, 0.0);
        label_data[label] = 1.0;

        return label_data;

    }

    void saveLoadedLabelsToFile(std::vector<double> label_data, std::string label_tensor_output) {
        Tensor<T> tensor({TENSOR_SIZE});

        for (size_t i = 0; i < label_data.size(); ++i) {
            tensor({i}) = label_data[i]; // This assumes tensor indexing allows for setting values
        }

        writeTensorToFile(tensor, label_tensor_output);

    }
};
