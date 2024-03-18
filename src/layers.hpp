#pragma once
#include <eigen3/Eigen/Dense>
#include <string>
#include <memory>
#include <cmath>
#include <random>

// Base class for all layer types in a neural network.
// It defines the interface for the forward and backward pass operations.
class BaseLayer {
public:
    // Forward pass takes an input vector and returns the layer's output vector.
    virtual Eigen::VectorXd forward(const Eigen::VectorXd& input) = 0;

    // Backward pass takes a gradient vector from the next layer and returns
    // the gradient vector with respect to the input of this layer.
    virtual Eigen::VectorXd backward(const Eigen::VectorXd& gradient) = 0;

    // Virtual destructor to allow derived class objects to be deleted correctly.
    virtual ~BaseLayer() = default;
};

// Fully connected (dense) layer implementation.
class FullyConnectedLayer : public BaseLayer {
    Eigen::MatrixXd weights; // Matrix of weights for the layer.
    Eigen::VectorXd biases;  // Vector of biases for the layer.
    Eigen::VectorXd inputCache; // Cached input vector for use in backward pass.
    double learningRate; // Learning rate for parameter updates.

public:
    // Constructor to initialize layer with given input and output sizes, and learning rate.
    FullyConnectedLayer(int inputSize, int outputSize, double lr) : learningRate(lr) {
        // Random number generator for initializing weights and biases.
        // std::random_device is used to obtain a seed for the random number engine.
        // std::mt19937 is a standard mersenne_twister_engine seeded with rd().
        std::random_device rd;
        std::mt19937 gen(rd());

        // He initialization for weight parameters, beneficial for layers before ReLU activations.
        double stddev = sqrt(2.0 / inputSize);
        std::normal_distribution<> d(0, stddev);

        // Initialize weights and biases using He initialization for weights and zeros for biases.
        weights = Eigen::MatrixXd(outputSize, inputSize).unaryExpr([&](double) { return d(gen); });
        biases = Eigen::VectorXd::Zero(outputSize);
    }

    // Performs the forward pass of the layer: computes the weighted sum of inputs and biases.
    Eigen::VectorXd forward(const Eigen::VectorXd& input) override {
        inputCache = input; // Cache input for use in backward pass.
        return (weights * input + biases).eval(); // Compute output.
    }

    // Performs the backward pass of the layer: computes gradients and updates parameters.
    Eigen::VectorXd backward(const Eigen::VectorXd& gradient) override {
        // Compute gradients for weights and biases using outer product of gradient and cached input.
        Eigen::MatrixXd dWeights = gradient * inputCache.transpose();
        const Eigen::VectorXd& dBiases = gradient;

        // Update weights and biases using the calculated gradients and learning rate.
        weights -= learningRate * dWeights;
        biases -= learningRate * dBiases;

        // Return gradient with respect to the input for use in previous layer's backward pass.
        return weights.transpose() * gradient;
    }
};

// Rectified Linear Unit (ReLU) activation layer.
class ReLU : public BaseLayer {
    Eigen::VectorXd inputCache; // Cached input vector for use in backward pass.

public:
    // Performs the ReLU operation on the input vector.
    Eigen::VectorXd forward(const Eigen::VectorXd& input) override {
        inputCache = input; // Cache input for use in backward pass.
        // Apply ReLU function element-wise: max(0, x).
        return input.unaryExpr([](double x) { return std::max(0.0, x); });
    }

    // Computes gradient of ReLU function during backward pass.
    Eigen::VectorXd backward(const Eigen::VectorXd& gradient) override {
        // Apply element-wise gradient of ReLU: 1 for x > 0, otherwise 0.
        Eigen::VectorXd gradInput = gradient.cwiseProduct(inputCache.unaryExpr([](double x) { return x > 0 ? 1.0 : 0.0; }));
        return gradInput;
    }
};

// Softmax activation layer for output normalization.
class SoftMax : public BaseLayer {
    Eigen::VectorXd outputCache; // Cached output vector for use in backward pass.

public:
    // Performs the SoftMax operation on the input vector.
    Eigen::VectorXd forward(const Eigen::VectorXd& input) override {
        // Subtract max coefficient for numerical stability and compute exponential.
        Eigen::VectorXd exp = (input.array() - input.maxCoeff()).exp();
        outputCache = exp / exp.sum(); // Normalize to get probabilities.
        return outputCache;
    }

    // Computes gradient of SoftMax function during backward pass.
    Eigen::VectorXd backward(const Eigen::VectorXd& gradient) override {
        long dim = gradient.size();
        Eigen::MatrixXd jacobian(dim, dim); // Jacobian matrix for SoftMax gradients.

        // Compute the jacobian matrix of the SoftMax function.
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                // Derivative formula differs depending on if indices are equal or not.
                jacobian(i, j) = i == j ? outputCache(i) * (1.0 - outputCache(j)) : -outputCache(i) * outputCache(j);
            }
        }

        // Multiply the gradient by the jacobian to get the gradient with respect to the input.
        return jacobian * gradient;
    }
};
