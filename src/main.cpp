#include <iostream>
#include <vector>
#include "../include/matrix.h"
#include "../include/network.h"
#include "../include/dataloader.h"
#include "../include/utils.h"
#include "../include/activations.h"

int main()
{
    try
    {
        std::vector<int> topology = {784, 64, 10}; // {784, 128, 64, 10}
        double learning_rate = 0.001;
        NeuralNetwork nn(topology, learning_rate);

        std::cout << "[*] OPERATING SYSTEM...\n";

        std::cout << "[*] LOADING DATA...\n\n";

        std::string data_path = "./data";

        auto all_data = DataLoader<Matrix, Matrix>::load_from_directory(data_path, mnist_parser);
        
        if (all_data.empty()) 
        {
            throw std::runtime_error("[X] DATA NOT FOUND: " + data_path + "\n");
        }
        
        std::shuffle(all_data.begin(), all_data.end(), std::mt19937(std::random_device()()));

        std::vector<DataPoint<Matrix, Matrix>> train_set, test_set;
        DataLoader<Matrix, Matrix>::split_data(all_data, train_set, test_set, 0.8);

        std::cout << "LOADED:\n " << " + Train: " << train_set.size() << "\n  + Test: " << test_set.size() << "\n\n\n";

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
                }
            }

            std::cout << "Epoch " << e + 1 << "/" << epochs << " DONE.\n";
        }

        std::cout << "\nEVALUATED:\n";
        int correct = 0;

        for (auto& test_point : test_set)
        {
            Matrix output = nn.feedForward(test_point.features);

            int predict = get_prediction(output);
            int actual = get_prediction(test_point.label);

            if (predict == actual) ++correct;
        }

        std::cout << "[*] RATE: " << (double)correct / test_set.size() * 100 << "%\n\n";

        // Input file path iykyk
        std::string input_path;
        while (true) {
            std::cout << "\nFile's path: ";
            std::cin >> input_path;

            if (input_path == "exit" || input_path == "ex") break;

            try {
                DataPoint<Matrix, Matrix> sample = mnist_parser(input_path, "0");

                Matrix output = nn.feedForward(sample.features);

                print_prediction_probs(output);

                std::cout << "=> PREDICT: " << get_prediction(output) << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "[!] Read file error: " << e.what() << std::endl;
            }
        }
        std::cout << "[*] end.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "\n[!] CRITICAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
