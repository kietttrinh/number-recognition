#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath>
#include "matrix.h"
#include "dataloader.h"

double sigmoid(double x);
double sigmoid_diff(double x);
double relu(double x);
double relu_diff(double x);

#endif //ACTIVATIONS_H
