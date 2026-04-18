#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
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

DataPoint<Matrix, Matrix> mnist_parser(const std::string& filepath, const std::string& label_str) {
    int width, height, channels;
    
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, 1);
    
    if (!data)
    {
        throw std::runtime_error("stb_image failure: " + std::string(stbi_failure_reason()) + " at " + filepath);
    }

    Matrix rawImg(height, width);
    for (int i = 0; i < height; ++i) 
    {
        for (int j = 0; j < width; ++j) 
        {
            rawImg.data[i][j] = static_cast<double>(data[i * width + j]) / 255.0;
        }
    }

    stbi_image_free(data);

    Matrix standardized = rawImg.resize(28, 28);

    Matrix features = standardized.flatten();
    Matrix label = Matrix::oneHot(std::stoi(label_str), 10);

    // std::cout << "Raw: " << rawImg.rows << "x" << rawImg.cols 
    //       << " | Resized: " << standardized.rows << "x" << standardized.cols 
    //       << " | Flattened: " << features.rows << "x" << features.cols << std::endl;

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

void print_prediction_probs(const Matrix& output) 
{
    std::cout << "\n--- RESULTS ---\n";
    for (int i = 0; i < output.rows; ++i) 
    {
        double percentage = output.data[i][0] * 100.0;
        std::cout << i << ": " << std::fixed << std::setprecision(2) << percentage << "%";
        if (i < output.rows - 1) std::cout << " | ";
    }
    std::cout << "\n----------------------\n";
}