#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#include "src/n-net.h"

void test_XOR();
void test_ABC();
void test_save();
void test_bench();
void stock_market_test();

int main(int, char**){
    // test_XOR();
    // test_ABC();
    // test_save();
    test_bench();
    // stock_market_test();

}

void stock_market_test() {
    const int N = 12*6;  // Number of elements to read

    NNet::netnum_mat_t input;  // Vector to hold input data
    NNet::netnum_mat_t target;  // Vector to hold target data

    std::ifstream inputFile("test.txt");  // Open the file

    if (inputFile.is_open()) {
        std::string line;

        // Read the first three numbers
        std::vector<float> firstVector;
        for (int i = 0; i < N-1 && std::getline(inputFile, line); i++) {
            float number = std::stof(line);
            firstVector.push_back(number);
        }

        while (std::getline(inputFile, line)) {  // Read lines one by one
            std::vector<float> innerVector = firstVector;
            float number = std::stof(line);
            innerVector.push_back(number);
            std::vector<float> innerTarget = {number};

            // Add innerVector to input
            input.push_back(innerVector);

            // Add number to target
            target.push_back(innerTarget);

            // Move one position
            firstVector.erase(firstVector.begin());
            firstVector.push_back(number);
        }

        inputFile.close();  // Close the file

        // remove last element in input
        input.pop_back();
        input.pop_back();

        // remove first element in target
        target.erase(target.begin());
        target.pop_back();
        
        //print the size of input and target
        std::cout << "input size: " << input.size() << "x" << input[0].size() << std::endl;
        std::cout << "target size: " << target.size() << std::endl;

        // print the first element of input
        std::cout << "input last: " << input[input.size() - 1][N - 1] << std::endl;
        // traget last
        std::cout << "target last: " << target[target.size() - 1][0] << std::endl;

        //make training data
        NNet::netnum_mat_t training_input;
        NNet::netnum_mat_t training_target;

        //use 95% of the data for training
        for (int i = 0; i < input.size() * 0.95; i++) {
            training_input.push_back(input[i]);
            training_target.push_back(target[i]);
        }

        NNet::netnum_mat_t testing_input;
        NNet::netnum_mat_t testing_target;

        //use 5% of the data for testing
        for (int i = input.size() * 0.95; i < input.size(); i++) {
            testing_input.push_back(input[i]);
            testing_target.push_back(target[i]);
        }


        // create the net
        NNet::Net net({N, (int)(N*0.9), (int)(N*0.6), 1}, 4);

        // reduce the learning rate
        net.alpha = 0.01;
        net.eta = 0.1;

        // train the net
        net.train(training_input, training_target, 10000);
        net.save_model("stock-market.net");

        // test the net
        NNet::netnum_vec_t result;
        for (int i = 0; i < testing_input.size(); i++) {
            net.predict(testing_input[i], result);
            std::cout << "predicted: " << result[0] << "\tactual: " << testing_target[i][0] << std::endl;
        }

    } else {
        std::cout << "Unable to open file." << std::endl;
    }

}


void test_bench() {
    NNet::DataLoader loader = NNet::DataLoader::from_png_folder("../test-data/my-hand-written-numbers");

    unsigned input_size = (unsigned)loader.get_input_values()[0].size();
    unsigned output_size = (unsigned)loader.get_target_values()[0].size();

    std::cout << "Input size: " << input_size << std::endl;
    std::cout << "Output size: " << output_size << std::endl;

    NNet::Net net({input_size, (unsigned)(input_size*0.9), (unsigned)(input_size*0.2), output_size}, 8);

    // adjust the learning rate
    net.eta = 0.10;
    net.alpha = 0.5;

    net.train(loader.get_input_values(), loader.get_target_values(), 10000);

    std::cout << "Average error: " << net.get_recent_avg_error() << std::endl;

    net.save_model("my-hand-written-numbers.net");


    auto to_test = NNet::DataLoader::from_png("../test-data/my-hand-written-numbers/2.png", {});
    
    // print the to_test image
    // for (int i = 0; i < 20; i++) {
    //     for (int j = 0; j < 20; ++j) {
    //         std::cout << (to_test[i * 20 + j]);
    //     }
    //     std::cout << std::endl;
    // }


    std::vector<NNet::netnum_t> output;
    net.predict(to_test, output);

    for(auto &i : output) {
        std::cout << round(i) << " ";
    }
}


void test_XOR() {
    // Simulating XOR
    std::vector<std::vector<NNet::netnum_t>> input_values = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<NNet::netnum_t>> target_values = {
        {0},
        {1},
        {1},
        {0}
    };

    std::vector<NNet::netnum_t> result;

    NNet::Net net({2, 3, 1});
    net.alpha = 0.15;
    net.eta = 0.5;

    net.train(input_values, target_values, 1000);

    std::cout << "XOR:" << std::endl;
    std::cout << "Average error: " << net.get_recent_avg_error() << std::endl;

    for (unsigned i = 0; i < input_values.size(); i++) {
        net.predict(input_values[i], result);
        std::cout << "Input: " << input_values[i][0] << ", " << input_values[i][1] << " -> " << (result[0]) << std::endl;
    }
}

void test_ABC() {
    std::vector<std::vector<NNet::netnum_t>> input_values = {
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1},
    };

    std::vector<std::vector<NNet::netnum_t>> target_values = {
        {0},
        {0},
        {0},
        {0},
        {0},
        {1},
        {1},
        {1},
    };

    std::vector<NNet::netnum_t> result;

    NNet::Net net({3, 4, 1});

    net.alpha = 0.15;
    net.eta = 0.5;

    net.train(input_values, target_values, 10000);

    std::cout << "\nABC:" << std::endl;
    std::cout << "Average error: " << net.get_recent_avg_error() << std::endl;

    for (unsigned i = 0; i < input_values.size(); i++) {
        net.predict(input_values[i], result);
        std::cout << "Input: " << input_values[i][0] << ", " << input_values[i][1] << ", " << input_values[i][2] << " -> " << (result[0]) << std::endl;
    }
}

void test_save() {
    // Simulating XOR
    
    std::vector<std::vector<NNet::netnum_t>> input_values = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };


    std::vector<NNet::netnum_t> result;

    std::vector<std::vector<NNet::netnum_t>> target_values = {
        {0},
        {1},
        {1},
        {0}
    };

    {
        NNet::Net net({2, 3, 1});
        net.alpha = 0.15;
        net.eta = 0.5;

        net.train(input_values, target_values, 1000);

        std::cout << "XOR:" << std::endl;
        std::cout << "Average error: " << net.get_recent_avg_error() << std::endl;

        for (unsigned i = 0; i < input_values.size(); i++) {
            net.predict(input_values[i], result);
            std::cout << "Input: " << input_values[i][0] << ", " << input_values[i][1] << " -> " << (result[0]) << std::endl;
        }

        net.save_model("xor.net");
    }

    {
        NNet::Net net = NNet::Net::load_model("xor.net");

        std::cout << "\nXOR Read:" << std::endl;
        for (unsigned i = 0; i < input_values.size(); i++) {
            net.predict(input_values[i], result);
            std::cout << "Input: " << input_values[i][0] << ", " << input_values[i][1] << " -> " << (result[0]) << std::endl;
        }
    }
}
