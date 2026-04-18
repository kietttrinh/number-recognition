#include "../include/matrix.h"
#include <random>
#include <stdexcept>
#include <iomanip>

Matrix::Matrix(int r, int c) : rows(r), cols(c)
{
    data.assign(r, std::vector<double>(c, 0.0));
}

void Matrix::print() const
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i)
    {    
        for (int j = 0; j < cols; ++j)
        {
            std::cout << data[i][j];
        }
        std::cout << "\n";
    }    
    
}

Matrix Matrix::random(int r, int c) 
{
    Matrix res(r, c);
    double scale = sqrt(2.0 / (c));
    for (int i = 0; i < r; ++i) 
    {
        for (int j = 0; j < c; ++j) 
        {
            res.data[i][j] = (((double)rand() / RAND_MAX) * 2 - 1) * scale;
        }
    }
    
    return res;
}

Matrix Matrix::add(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("The matrix's sizes don't match!");
    }

    Matrix res(rows, cols);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            res.data[i][j] = data[i][j] + other.data[i][j];
        }
    }

    return res;
}

Matrix Matrix::subtract(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("The matrix's sizes don't match!");
    }

    Matrix res(rows, cols);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            res.data[i][j] = data[i][j] - other.data[i][j];
        }
    }

    return res;
}

Matrix Matrix::multiply(const Matrix &other) const
{
    // (m x n) * (n x p) = (m x p)
    if (cols != other.rows) 
    {
        throw std::invalid_argument("The matrix's sizes don't match!");
    }

    Matrix res(rows, other.cols);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < other.cols; ++j)
        {
            for (int k = 0; k < cols; ++k)
            {
                res.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }

    return res;
}

Matrix Matrix::multiplyScalar(double scalar) const 
{
    Matrix res(rows, cols);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            res.data[i][j] = this->data[i][j] * scalar;
        }
    }
    return res;
}

// Backpropagation
Matrix Matrix::multiplyElement(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("The matrix's sizes don't match!");
    }

    Matrix res(rows, cols);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            res.data[i][j] = data[i][j] * other.data[i][j];
        }
    }
    
    return res;
}

Matrix Matrix::transpose() const
{
    Matrix res(cols, rows);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            res.data[j][i] = data[i][j];
        }
    }

    return res;
}

// Function pointer `double (*func)(double)`
Matrix Matrix::map(double (*func)(double)) const
{
    Matrix res(rows, cols);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            res.data[i][j] = func(data[i][j]);
        }
    }

    return res;
}

Matrix Matrix::applyConvolution(const Matrix &input, const Matrix &kernel) const
{
    int outputRows = input.rows - kernel.rows + 1;
    int outputCols = input.cols - kernel.cols + 1;

    Matrix output(outputRows, outputCols);

    // Loop 1 & 2
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outputRows; ++i)
    {
        for (int j = 0; j < outputCols; ++j)
        {
            double sum = 0.0;
            // Loop 3 & 4
            #pragma omp parallel for collapse(2)
            for (int ki = 0; ki < kernel.rows; ++ki)
            {
                for (int kj = 0; kj < kernel.cols; ++kj)
                {
                    sum += input.data[i + ki][j + kj] * kernel.data[ki][kj];
                }
            }
            output.data[i][j] = sum;
        }
    }

    return output;
}

Matrix Matrix::applySoftmax(const Matrix& m)
{
    Matrix res(m.rows, m.cols);

    double sum = 0.0;
    double max = m.data[0][0];

    for (int i = 0; i < m.rows; ++i) if (max < m.data[i][0]) max = m.data[i][0];

    for (int i = 0; i < m.rows; ++i)
    {
        res.data[i][0] = std::exp(m.data[i][0] - max);  
        sum += res.data[i][0];
    }

    for (int i = 0; i < m.rows; ++i)
    {
        res.data[i][0] /= sum;
    }

    return res;
}

Matrix Matrix::oneHot(int label, int size) 
{
    Matrix res(size, 1);
    for (int i = 0; i < size; ++i) res.data[i][0] = (i == label ? 1.0 : 0.0);
    return res;
}

Matrix Matrix::flatten() const
{
    Matrix res(rows * cols, 1);

    int k = 0;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            res.data[k++][0] = data[i][j];
        }
    }

    return res;
}

// https://courses.cs.vt.edu/~masc1044/L17-Rotation/ScalingNN.html
Matrix Matrix::resize(int r, int c) const
{
    Matrix res(r, c);

    double rowRatio = (double)this->rows / r;
    double colRatio = (double)this->cols / c;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            int oldI = std::min((int)(i * rowRatio), this->rows - 1);
            int oldJ = std::min((int)(j * colRatio), this->cols - 1);

            res.data[i][j] = this->data[oldI][oldJ];
        }
    }

    return res;
}
