# neural-net
A simple neural network implementation in C++.

## Features
- [ ] Different activation functions
- [x] Loading / Saving models
- [ ] Multi threaded training
- [ ] Cache optimisation

## Usage
```cpp
#include <iostream>
#include <iterator>
#include <vector>

#include "n-net.h"

void main() {
    // Example: Simulating XOR
    std::vector<std::vector<NNet::netnum_t>> input_values = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<NNet::netnum_t>> expected_values = {
        {0},
        {1},
        {1},
        {0}
    };

    // Create a network with 2 input neurons, 3 hidden neurons and 1 output neuron
    NNet::Net net({2, 3, 1});

    net.train(input_values, expected_values, 1000);
    std::cout << "Average error: " << net.get_recent_avg_error() << std::endl;

    std::cout << "XOR:" << std::endl;
    std::vector<NNet::netnum_t> prediction;
    for (unsigned i = 0; i < input_values.size(); i++) {
        net.predict(input_values[i], prediction);
        std::cout << "Input: " << 
            input_values[i][0] << "," << input_values[i][1] << " -> " << (prediction[0]) << std::endl;
    }
}
