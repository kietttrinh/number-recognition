#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include "matrix.h"
#include "dataloader.h"
#include <cmath>
#include <string>

DataPoint<Matrix, Matrix> mnist_parser(const std::string& filepath, const std::string& label_str);
int get_prediction(const Matrix& output);
void print_prediction_probs(const Matrix& output);

#endif //UTILS_H
