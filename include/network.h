#ifndef NETWORK_H
#define NETWORK_H

#include <cmath>
#include "matrix.h"
#include "activations.h"
#include "dataloader.h"
#include <vector>

class NeuralNetwork
{
public:
    // Create NN
    NeuralNetwork(std::vector<int> topology, double learning_rate);

    // 
    Matrix feedForward(const Matrix &input);

    void train(const Matrix& input, const Matrix& target);
    
    double calculateLoss(const Matrix &output, const Matrix &target);

    void predict(const Matrix& input_image);

private:
    std::vector<int> topology;
    double learningRate;

    //Z = W . X + b
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
    std::vector<Matrix> layerOutputs;
};

#endif //NETWORK_H
