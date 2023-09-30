#include <iostream>
#include <iterator>
#include <vector>

#include "src/n-net.h"

void test_XOR() {
    // Simulating XOR
    std::vector<std::vector<NNet::netnum_t>> input_vals = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<NNet::netnum_t>> target_vals = {
        {0},
        {1},
        {1},
        {0}
    };

    std::vector<NNet::netnum_t> result_vals;

    NNet::Net net({2, 3, 1});
    net.alpha = 0.15;
    net.eta = 0.5;

    net.train(input_vals, target_vals, 1000);

    std::cout << "XOR:" << std::endl;
    std::cout << "Average error: " << net.get_recent_avg_error() << std::endl;

    for (unsigned i = 0; i < input_vals.size(); i++) {
        net.predict(input_vals[i], result_vals);
        std::cout << "Input: " << input_vals[i][0] << ", " << input_vals[i][1] << " -> " << (result_vals[0]) << std::endl;
    }
}

void test_ABC() {
    std::vector<std::vector<NNet::netnum_t>> input_vals = {
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1},
    };

    std::vector<std::vector<NNet::netnum_t>> target_vals = {
        {0},
        {0},
        {0},
        {0},
        {0},
        {1},
        {1},
        {1},
    };

    std::vector<NNet::netnum_t> result_vals;

    NNet::Net net({3, 4, 1});

    net.alpha = 0.15;
    net.eta = 0.5;

    net.train(input_vals, target_vals, 10000);

    std::cout << "\nABC:" << std::endl;
    std::cout << "Average error: " << net.get_recent_avg_error() << std::endl;

    for (unsigned i = 0; i < input_vals.size(); i++) {
        net.predict(input_vals[i], result_vals);
        std::cout << "Input: " << input_vals[i][0] << ", " << input_vals[i][1] << ", " << input_vals[i][2] << " -> " << (result_vals[0]) << std::endl;
    }
}

void test_save() {
    // Simulating XOR
    
    std::vector<std::vector<NNet::netnum_t>> input_vals = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };


    std::vector<NNet::netnum_t> result_vals;

    std::vector<std::vector<NNet::netnum_t>> target_vals = {
        {0},
        {1},
        {1},
        {0}
    };

    {
        NNet::Net net({2, 3, 1});
        net.alpha = 0.15;
        net.eta = 0.5;

        net.train(input_vals, target_vals, 1000);

        std::cout << "XOR:" << std::endl;
        std::cout << "Average error: " << net.get_recent_avg_error() << std::endl;

        for (unsigned i = 0; i < input_vals.size(); i++) {
            net.predict(input_vals[i], result_vals);
            std::cout << "Input: " << input_vals[i][0] << ", " << input_vals[i][1] << " -> " << (result_vals[0]) << std::endl;
        }

        net.save_model("xor.net");
    }

    {
        NNet::Net net = NNet::Net::load_model("xor.net");

        std::cout << "\nXOR Read:" << std::endl;
        for (unsigned i = 0; i < input_vals.size(); i++) {
            net.predict(input_vals[i], result_vals);
            std::cout << "Input: " << input_vals[i][0] << ", " << input_vals[i][1] << " -> " << (result_vals[0]) << std::endl;
        }
    }
}


int main(int, char**){
    //test_XOR();
    //test_ABC();
    test_save();
}

