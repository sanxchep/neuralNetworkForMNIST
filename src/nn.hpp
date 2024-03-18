#pragma once
#include "layers.hpp"
#include "loss.hpp"
#include <eigen3/Eigen/Dense>
#include <ranges>
#include <vector>
#include <memory>
#include <iostream>
#include <numeric>
#include "helpers.hpp"
#include <chrono>

class NeuralNetwork {
private:
    double learningRate;
    std::vector <Eigen::VectorXd> trainingImageData, trainingLabelData, testingImageData, testingLabelData;
    std::vector <std::shared_ptr<BaseLayer>> layers;
    CrossEntropyLoss lossLayer;
    std::vector<double> lossHistory;

public:
    NeuralNetwork(double lr, const std::vector <std::vector<double>> &trainingImages,
                  const std::vector <std::vector<double>> &trainingLabels,
                  const std::vector <std::vector<double>> &tesingImages,
                  const std::vector <std::vector<double>> &tesingLabels)
            : learningRate(lr) {

        // Convert the input data to Eigen::VectorXd because we did not do it in the data loader
        // TODO: This is not efficient, we should convert the data to Eigen::VectorXd in the data loader
        for (const auto &vec: trainingImages) {
            trainingImageData.emplace_back(Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size()));
        }
        for (const auto &vec: trainingLabels) {
            trainingLabelData.emplace_back(Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size()));
        }

        for (const auto &vec: tesingImages) {
            testingImageData.emplace_back(Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size()));
        }

        for (const auto &vec: tesingLabels) {
            testingLabelData.emplace_back(Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size()));
        }
    }

    void setupLayers(int inputSize, int hiddenSize, int outputSize) {
        layers.push_back(std::make_shared<FullyConnectedLayer>(inputSize, hiddenSize, learningRate));
        layers.push_back(std::make_shared<ReLU>());
        layers.push_back(std::make_shared<FullyConnectedLayer>(hiddenSize, outputSize, learningRate));
        layers.push_back(std::make_shared<SoftMax>());
    }

    Eigen::VectorXd forwardPass(const Eigen::VectorXd &input) {
        Eigen::VectorXd output = input;
        for (const auto &layer: layers) {
            output = layer->forward(output);
        }
        return output;
    }

    void backwardPass(const Eigen::VectorXd &gradient) {
        Eigen::VectorXd error = gradient;
        for (auto & layer : std::ranges::reverse_view(layers)) {
            error = layer->backward(error);
        }
    }

    void train(size_t epochs) {
        auto timerStart = std::chrono::high_resolution_clock::now();

        Eigen::setNbThreads(4); // Use 4 threads for Eigen operations

        std::cout << "Training with " << Eigen::nbThreads() << " threads." << std::endl;

        double loss;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            // Clear loss history for this epoch
            lossHistory.clear();

            // Run for every image in the dataset
            for (size_t datasetIndex = 0; datasetIndex < trainingImageData.size(); ++datasetIndex) {

                // Forward pass
                Eigen::VectorXd prediction_tensor = forwardPass(trainingImageData[datasetIndex]);

                // Compute loss
                loss = lossLayer.forward(prediction_tensor, trainingLabelData[datasetIndex]);

                // Backward pass
                Eigen::VectorXd error = lossLayer.backward(prediction_tensor, trainingLabelData[datasetIndex]);

                backwardPass(error);
            }

            // Store loss for this epoch
            lossHistory.push_back(loss);

            // Compute average loss for the epoch
            double avgLoss = std::accumulate(lossHistory.begin(), lossHistory.end(), 0.0) / lossHistory.size();

            std::cout << "Epoch " << epoch + 1 << ", Average Loss: " << avgLoss << std::endl;

            // early stopping
            if (avgLoss < 0.0001) {
                std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
                break;
            }
        }

        // Stop timer and calculate duration
        auto timerStop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(timerStop - timerStart).count();

        std::cout << "Training took " << duration << " seconds." << std::endl;
    }

    void test(const std::string& filename) {
        // Total of correct predictions and incorrect predictions
        int correct = 0;
        int incorrect = 0;

        for (int datasetIndex = 0; datasetIndex < testingImageData.size(); datasetIndex++) {

            // Forward pass
            Eigen::VectorXd output = forwardPass(testingImageData[datasetIndex]);

            // Get the index of the maximum element in the output vector
            int predictionLabel;
            output.maxCoeff(&predictionLabel);

            // Get the index of the maximum element in the label vector
            int actualLabel;
            testingLabelData[datasetIndex].maxCoeff(&actualLabel);

            // Log the prediction as per the format
            logPrediction(predictionLabel, actualLabel, datasetIndex, filename);

            // Update correct and incorrect counts
            if (predictionLabel == actualLabel) {
                correct++;
            } else {
                incorrect++;
            }
        }

        std::cout << "Correct: " << correct << ", Incorrect: " << incorrect << std::endl;
        std::cout << "Accuracy: " << (double) correct / (correct + incorrect) * 100 << "%" << std::endl;
    }
};