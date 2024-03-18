#include "nn.hpp"
#include "data_loader/image_io.hpp"
#include "data_loader/label_io.hpp"
#include "helpers.hpp"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "expected config as one and only parameter" << std::endl;
        return -1;
    }

    std::ifstream configfile(argv[1]);
    if (!configfile.is_open()) {
        std::cerr << "could not open configfile" << std::endl;
        return -1;
    }

    std::map <std::string, std::string> config = parseConfigfile(configfile);

    // load hyperparameters and paths from config file
    int hiddenSize = std::stoi(config["hidden_size"]);
    int epochs = std::stoi(config["num_epochs"]);
    int batchSize = std::stoi(config["batch_size"]);
    double learningRate = std::stod(config["learning_rate"]);

    std::string trainingImagePath = config["rel_path_train_images"];
    std::string trainingLabelPath = config["rel_path_train_labels"];

    std::string testingImagePath = config["rel_path_test_images"];
    std::string testingLabelPath = config["rel_path_test_labels"];

    std::string predictionLogFileName = config["rel_path_log_file"];

    // open log file and create the testing log header
    std::ofstream file(predictionLogFileName);
    if (!file) {
        std::cerr << "Unable to open file for writing.\n";
        return -1;
    }
    file << "Current batch: 0\n";
    file.close();

    std::cout << "Config Loaded" << std::endl;

    size_t trainingItemCount = getItemCount(trainingLabelPath);
    size_t testingItemCount = getItemCount(testingLabelPath);

    std::vector<std::vector<double>> trainingImageData = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> trainingLabelData = std::vector<std::vector<double>>();

    std::vector<std::vector<double>> testingImageData = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> testingLabelData = std::vector<std::vector<double>>();

    for(int i = 0; i < trainingItemCount; i++) {
        IOimage<double> ioimage(trainingImagePath, i);
        IOlabel<double> iolabel(trainingLabelPath, i);
        trainingImageData.push_back(ioimage.extractImageAndNormaliseImage());
        trainingLabelData.push_back(iolabel.extractLabel());
    }

   for (int i = 0; i < testingItemCount; i++) {
        IOimage<double> ioimage(testingImagePath, i);
        IOlabel<double> iolabel(testingLabelPath, i);
        testingImageData.push_back(ioimage.extractImageAndNormaliseImage());
        testingLabelData.push_back(iolabel.extractLabel());
    }

    std::cout << "Data Loaded" << std::endl;

    // Initialize neural network with config parameters
    NeuralNetwork neuralNetwork(learningRate, trainingImageData, trainingLabelData, testingImageData, testingLabelData);

    // Setup layers based on sizes
    neuralNetwork.setupLayers(INPUT_SIZE, hiddenSize, OUTPUT_SIZE);

    // Train the network
    neuralNetwork.train(epochs);

    std::cout << "Training Complete" << std::endl;

    // Test the network
    neuralNetwork.test(predictionLogFileName);

    return 0;
}