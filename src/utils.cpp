#include "../include/utils.h"

// // https://courses.cs.vt.edu/~masc1044/L17-Rotation/ScalingNN.html
// Matrix resizeMatrix(const Matrix& original, int targetSize) 
// {
//     Matrix resized(targetSize, targetSize);
    
//     for (int i = 0; i < targetSize; ++i) {
//         for (int j = 0; j < targetSize; ++j) {
//             int orig_i = (i * original.rows) / targetSize;
//             int orig_j = (j * original.cols) / targetSize;
            
//             resized.data[i][j] = original.data[orig_i][orig_j];
//         }
//     }
//     return resized;
// }

DataPoint<Matrix, Matrix> mnist_parser(const std::string& filepath, const std::string& label_str)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Can't open file: " + filepath);
    }

    int width, height; // original size
    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));

    // 
    Matrix rawImg(height, width);
    unsigned char pixel;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            file.read(reinterpret_cast<char*>(&pixel), 1);

            // Normalize [0, 1]
            rawImg.data[i][j] = static_cast<double>(pixel) / 255.0;
        }
    }
    file.close();

    Matrix standardized = rawImg.resize(28, 28);

    Matrix features = standardized.flatten();
    Matrix label = Matrix::oneHot(std::stoi(label_str), 10);

    return {features, label};
}

int get_prediction(const Matrix& output) 
{
    int predicted_digit = 0;
    double max_probability = output.data[0][0];

    for (int i = 1; i < output.rows; ++i) {
        if (output.data[i][0] > max_probability) {
            max_probability = output.data[i][0];
            predicted_digit = i;
        }
    }

    return predicted_digit;
}