#include "../tensor.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

const uint IMAGE_HEADER_SIZE = 16; // bytes for magic number
const uint MAGIC_NUMBER_IMAGES = 0x803;

// Function to normalize a vector of uint8 values to double values in the range 0.0 to 1.0
template<typename T>
std::vector<T> normalize(const std::vector<uint8_t>& input) {
    std::vector<T> toReturn;
    for (uint8_t value : input) {
        toReturn.push_back(static_cast<T>(value) / static_cast<T>(255));
    }
    return toReturn;
}

template<typename T>
class IOimage {
private:
    std::string image_dataset_input;
    int image_index;
    uint32_t num_rows;
    uint32_t num_cols;

public:
    // Constructor
    IOimage(std::string  dataset_input, int index)
            : image_dataset_input(std::move(dataset_input)), image_index(index) {}

    std::vector<T> extractImageAndNormaliseImage() {
        std::ifstream input_file(image_dataset_input, std::ios::binary);

        if (!input_file.is_open()) {
            throw std::runtime_error("File open failed: " + image_dataset_input);
        }

        // Read the MNIST Header
        uint32_t magic_number, num_images;

        input_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number);

        if (magic_number != MAGIC_NUMBER_IMAGES) {
            throw std::runtime_error("Not a MNIST image data file");
        }

        input_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
        num_images = __builtin_bswap32(num_images);

        input_file.read(reinterpret_cast<char*>(&this->num_rows), sizeof(this->num_rows)); // Use this-> to refer to the member variable
        this->num_rows = __builtin_bswap32(this->num_rows);

        input_file.read(reinterpret_cast<char*>(&this->num_cols), sizeof(this->num_cols));
        this->num_cols = __builtin_bswap32(this->num_cols);

        input_file.seekg((num_rows * num_cols * image_index) + IMAGE_HEADER_SIZE);
        std::vector<uint8_t> image(num_rows * num_cols);
        input_file.read(reinterpret_cast<char*>(image.data()), num_rows * num_cols);
        input_file.close(); // No longer need the file open

        return normalize<T>(image);
    }

    // Convert image to tensor and save to file
    void saveLoadedImagesToFile(std::vector<T> image_data, std::string image_tensor_output) {
        // Tensor should have the dimensions rows x cols
        Tensor<T> tensor({num_rows, num_cols});

        // Transfer vector to tensor
        size_t vectorIdx = 0; // Counter to have the correct index in the vector
        for (size_t i = 0; i < num_rows; i++) {
            for (size_t j = 0; j < num_cols; j++) {
                tensor({i, j}) = image_data[vectorIdx++];
            }
        }

        // Write to file
        writeTensorToFile(tensor, image_tensor_output);
    }
};
