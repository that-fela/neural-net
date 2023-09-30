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


    NNet::Net net({2, 3, 1});

    net.train(input_vals, target_vals, 1000);
    std::cout << "Average error: " << net.get_recent_avg_error() << std::endl;

    std::cout << "XOR:" << std::endl;
    std::vector<NNet::netnum_t> result_vals;
    for (unsigned i = 0; i < input_vals.size(); i++) {
        net.predict(input_vals[i], result_vals);
        std::cout << "Input: " << 
            input_vals[i][0] << "," << input_vals[i][1] << " -> " << (result_vals[0]) << std::endl;
    }
}
