#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

#include "src/n-net.h"

void test_XOR();
void test_ABC();
void test_save();
void test_bench();

int main(int, char**){
    // test_XOR();
    // test_ABC();
    // test_save();
    test_bench();

}

void test_bench() {
    NNet::DataLoader loader = NNet::DataLoader::from_png_folder("../test-data/my-hand-written-numbers");

    NNet::Net net({400, 400, 100, 10});

    auto start_time = std::chrono::high_resolution_clock::now();
    net.train(loader.get_input_values(), loader.get_target_values(), 1000);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Training time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;

    std::cout << "Average error: " << net.get_recent_avg_error() << std::endl;

    net.save_model("my-hand-written-numbers.net");



    auto to_test = NNet::DataLoader::from_png("../test-data/my-hand-written-numbers/7.png", {});
    
    // print the to_test image
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; ++j) {
            std::cout << (to_test[i * 20 + j]);
        }
        std::cout << std::endl;
    }


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
