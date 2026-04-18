#include "../include/activations.h"

double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_diff(double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double relu(double x)
{
    return (x > 0 ? x : 0);
}

double relu_diff(double x)
{
    return ((x > 0) ? 1 : 0);
}