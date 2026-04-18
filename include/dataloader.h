#ifndef DATALOADER_H
#define DATALOADER_H

#include "matrix.h"
#include <string>
#include <algorithm>
#include <random>
#include <functional>
#include <filesystem>

namespace fs = std::filesystem;

template <typename T, typename L>
struct DataPoint
{
    T features; // Features (Matrix, vector)
    L label; // Label 
};

template <typename T, typename L>
class DataLoader
{
public:
    // Load data
    static std::vector<DataPoint<T, L>> load_from_directory(
        const std::string& base_path,
        std::function<DataPoint<T, L>(const std::string&, const std::string&)> parser // Define parser
    );

    // Split
    static void split_data(
        std::vector<DataPoint<T, L>>& all_data,
        std::vector<DataPoint<T, L>>& train_set,
        std::vector<DataPoint<T, L>>& test_set,
        double ratio
    );

    // Create batch
    static std::vector<std::vector<DataPoint<T, L>>> create_batches(
        std::vector<DataPoint<T, L>>& data,
        int batch_size
    );

private:
 
};

template <typename T, typename L>
void DataLoader<T, L>::split_data(
    std::vector<DataPoint<T, L>>& all_data,
    std::vector<DataPoint<T, L>>& train_set,
    std::vector<DataPoint<T, L>>& test_set,
    double ratio)
{
    std::random_device rd;
    std::mt19937_64 generator(rd());
    std::shuffle(all_data.begin(), all_data.end(), generator);

    size_t train_size = static_cast<size_t>(all_data.size() * ratio);

    train_set.assign(all_data.begin(), all_data.begin() + train_size);
    test_set.assign(all_data.begin() + train_size, all_data.end());
}

template <typename T, typename L> 
// [] (n size) -> [[], [],...,[]] ((n / size) batches)
// 2D array
std::vector<std::vector<DataPoint<T, L>>> DataLoader<T, L>::create_batches(
        std::vector<DataPoint<T, L>>& data,
        int batch_size)
{
    std::vector<std::vector<DataPoint<T, L>>> all_batches;

    for (size_t i = 0; i < data.size(); i += (size_t)batch_size)
    {
        // Make sure not Buffer Overflow
        size_t end = std::min(i + (size_t)batch_size, data.size()); 
        
        std::vector<DataPoint<T, L>> batch(data.begin() + i, data.begin() + end);

        all_batches.push_back(batch);
    }

    return all_batches;
}

template <typename T, typename L>
std::vector<DataPoint<T, L>> DataLoader<T, L>::load_from_directory(
        const std::string& base_path,
        std::function<DataPoint<T, L>(const std::string&, const std::string&)> parser)
{
    std::vector<DataPoint<T, L>> all_data;

    if (!fs::exists(base_path) || !fs::is_directory(base_path))
    {
        throw std::runtime_error("Path not exists!");
    }

    // Auto get into subdirectory
    for (const auto& entry : fs::recursive_directory_iterator(base_path))
    {
        if (fs::is_regular_file(entry))
        {
            std::string filepath = entry.path().string();
            std::string label = entry.path().parent_path().filename().string();

            try
            {
                DataPoint<T, L> dp = parser(filepath, label);
                all_data.push_back(dp);
            }
            catch (const std::exception& e)
            {
                std::cerr << "Processing file " << filepath << ": " << e.what() << "\n";
            }
        }
    }

    return all_data;
}

#endif //DATALOADER_H
