#include <iostream>
#include <vector>
#include "../include/matrix.h"
#include "../include/network.h"
#include "../include/dataloader.h"
#include "../include/utils.h"
#include "../include/activations.h"

int main()
{
    std::vector<int> topology = {784, 64, 10}; // {784, 128, 64, 10}
    double learning_rate = 0.1;
    NeuralNetwork nn(topology, learning_rate);

    std::cout << "[*] OPERATING SYSTEM...\n";

    std::cout << "[*] LOADING DATA...\n\n";

    std::string data_path = "./data";

    auto all_data = DataLoader<Matrix, Matrix>::load_from_directory(data_path, mnist_parser);

    if (all_data.empty()) 
    {
        throw std::runtime_error("[X] DATA NOT FOUND: " + data_path + "\n");
    }

    std::vector<DataPoint<Matrix, Matrix>> train_set, test_set;
    DataLoader<Matrix, Matrix>::split_data(all_data, train_set, test_set, 0.8);

    std::cout << "LOADED:\n " << " + Train: " << train_set.size() << "\n + Test: " << test_set.size() << "\n";

    int epochs = 10; 
    int batch_size = 32;

    for (int e = 0; e < epochs; ++e)
    {
        double total_loss = 0.0;

        auto batches = DataLoader<Matrix, Matrix>::create_batches(train_set, batch_size);

        for (auto& batch : batches)
        {
            for (auto& point : batch)
            {
                nn.train(point.features, point.label);

                // 
                total_loss += nn.calculateLoss(point.features, point.label);
            }
        }

        std::cout << "Epoch " << e + 1 << "/" << epochs << "done.\n";
    }

    std::cout << "Evaluated\n\n";
    int correct = 0;

    for (auto& test_point : test_set)
    {
        Matrix output = nn.feedForward(test_point.features);

        int predict = get_prediction(output);
        int actual = get_prediction(test_point.label);

        if (predict == actual) ++correct;
    }

    std::cout << "[*] RATE: " << (double)correct / test_set.size() * 100 << "%\n\n";

    return 0;
}
