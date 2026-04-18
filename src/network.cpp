#include "../include/network.h"

// Explain this details?
NeuralNetwork::NeuralNetwork(std::vector<int> topology, double learning_rate) : topology(topology), learningRate(learning_rate)
{
    // Z = W * X + b
    // {768, 64, 10}  
    for (size_t i = 0; i < topology.size() - 1; ++i)
    {
        weights.push_back(Matrix::random(topology[i + 1], topology[i]));
        biases.push_back(Matrix::random(topology[i + 1], 1));
    }
}

Matrix NeuralNetwork::feedForward(const Matrix& input)
{
    // (n x 1)
    Matrix current = input;
    layerOutputs.clear();
    layerOutputs.push_back(current);

    for (size_t i = 0; i < weights.size(); ++i)
    {
        // Z = W * X + bias
        current = weights[i].multiply(current).add(biases[i]);

        if (i == weights.size() - 1)
        {
            current = Matrix::applySoftmax(current);
        }
        else 
        {
            current = current.map(sigmoid);
        }
        layerOutputs.push_back(current);
    }

    return current;
}


double NeuralNetwork::calculateLoss(const Matrix& output, const Matrix& target)
{
    double loss = 0.0;

    for (int i = 0; i < output.rows; ++i)
    {
        double error = output.data[i][0] - target.data[i][0];
        loss += (error * error);
    }

    return loss / (output.rows);
}

void NeuralNetwork::train(const Matrix& input, const Matrix& target)
{
    Matrix output = feedForward(input);

    if (output.rows != target.rows || output.cols != target.cols) 
    {
        std::cerr << "Dimension Mismatch!" << std::endl;
        std::cerr << "Output: " << output.rows << "x" << output.cols << std::endl;
        std::cerr << "Target: " << target.rows << "x" << target.cols << std::endl;
        throw std::runtime_error("The matrix's sizes don't match!");
    }
    
    // Error = Predicted - Target
    Matrix currentError = layerOutputs.back().subtract(target);

    for (int i = weights.size() - 1; i >= 0; --i)
    {
        // Gradients
        Matrix gradients = layerOutputs[i + 1].map(sigmoid_diff);
        gradients = gradients.multiplyElement(currentError);

        // Calculate the loss for prev layer
        Matrix nextError = weights[i].transpose().multiply(gradients);
        
        // Data Weights = Gradient x (layerOutputs[i]^T)
        Matrix data_weights = gradients.multiply(layerOutputs[i].transpose());

        // Update weights & biases 
        // Gradient Descent
        weights[i] = weights[i].subtract(data_weights.multiplyScalar(learningRate));
        biases[i] = biases[i].subtract(gradients.multiplyScalar(learningRate));

        // Next layer
        currentError = nextError;
    }
}

// Later
// void NeuralNetwork::predict(const Matrix& input_image);
// {

// }