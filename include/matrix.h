#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

struct Matrix
{
    int rows, cols;
    std::vector<std::vector<double>> data;

    Matrix(int r, int c);

    void print() const;
    static Matrix random(int r, int c);

    Matrix add(const Matrix &other) const;
    Matrix subtract(const Matrix &other) const;
    Matrix multiply(const Matrix &other) const; // Dot product
    Matrix multiplyScalar(double scalar) const;
    Matrix multiplyElement(const Matrix &other) const; // Hadamard product
    Matrix transpose() const;
    Matrix map(double (*func)(double)) const;
    Matrix applyConvolution(const Matrix &input, const Matrix &kernel) const;
    static Matrix applySoftmax(const Matrix& m);
    static Matrix oneHot(int label, int num_classes);
    Matrix flatten() const;
    Matrix resize(int r, int c) const;
};

#endif //MATRIX_H
