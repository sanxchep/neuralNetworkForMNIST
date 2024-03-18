#pragma once
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <limits>

// Class to compute CrossEntropyLoss, commonly used as a loss function for classification tasks.
class CrossEntropyLoss {
public:
    // Default constructor.
    CrossEntropyLoss() = default;

    // Calculates the forward pass of the cross-entropy loss.
    // This method computes the loss given the predictions and the target distributions.
    static double forward(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets) {

        // Ensure numerical stability by adding a small value to predictions to avoid log(0).
        // std::numeric_limits<double>::epsilon() is used to get the smallest positive value such that 1.0 + epsilon != 1.0.
        Eigen::VectorXd safe_predictions = predictions.array().max(std::numeric_limits<double>::epsilon());

        // Compute the natural logarithm of the predictions.
        // Logarithm of each element is taken to calculate the log likelihood.
        Eigen::VectorXd log_predictions = safe_predictions.array().log();

        // Calculate the negative log likelihood, which is the essence of cross-entropy loss.
        // This is achieved by element-wise multiplication of the targets with the log_predictions, followed by a sum and negation.
        double loss = -(targets.array() * log_predictions.array()).sum();

        return loss;
    }

    // Calculates the gradient of the loss function with respect to the predictions.
    // This method is crucial for the backward pass in training neural networks.
    static Eigen::VectorXd backward(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets) {

        // Compute the gradient of the cross-entropy loss with respect to the predictions.
        // This is done by dividing the negative targets by the predictions, ensuring numerical stability by avoiding division by zero.
        Eigen::VectorXd gradient = -targets.array() / predictions.array().max(std::numeric_limits<double>::epsilon());

        return gradient;
    }
};
