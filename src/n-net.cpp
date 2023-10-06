#include "n-net.h"
#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace NNet;

Net::Net(const std::vector<unsigned> &topology, unsigned num_threads){
    omp_set_num_threads(num_threads);

    unsigned num_layers = topology.size();
    assert(num_layers > 1);

    for (unsigned layer = 0; layer < num_layers; layer++) {
        m_layers.push_back(Layer());

        unsigned num_outputs = (layer == num_layers - 1) ? 0 : topology[layer + 1];

        for (unsigned neuron = 0; neuron <= topology[layer]; neuron++) {
            m_layers.back().push_back(Neuron(num_outputs, neuron, this));
        }

        m_layers.back().back().set_output(1.0);
    }
}

Net::~Net() {

}

void Net::feed_forward(const std::vector<netnum_t> &input_vals) {
    assert(input_vals.size() == m_layers[0].size() - 1);

    for (unsigned i = 0; i < input_vals.size(); i++) {
        m_layers[0][i].set_output(input_vals[i]);
    }

    for (unsigned layer = 1; layer < m_layers.size(); layer++) {
        Layer &previous_layer = m_layers[layer - 1];

        #pragma omp parallel for
        for (unsigned n = 0; n < m_layers[layer].size() - 1; n++) {
            m_layers[layer][n].feed_forward(previous_layer, n);
        }
    }
}

void Net::back_prop(const std::vector<netnum_t> &target) {
    // calc the error
    Layer &output_layer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < output_layer.size() - 1; n++) {
        netnum_t delta = target[n] - output_layer[n].get_output();
        m_error += delta * delta;
    }

    m_error /= output_layer.size() - 1;
    m_error = sqrt(m_error);

    // calc recent avg error
    m_recent_avg_error = (m_recent_avg_error * m_recent_avg_smoothing_factor + m_error) / (m_recent_avg_smoothing_factor + 1.0);
    // std::cout << "Error: " << m_recent_avg_error << std::endl;

    // calc output layer gradients
    for (unsigned n = 0; n < output_layer.size() - 1; n++) {
        output_layer[n].calc_output_gradients(target[n]);
    }

    // calc hidden layer gradients
    for (unsigned layer = m_layers.size() - 2; layer > 0; layer--) {
        Layer &hidden_layer = m_layers[layer];
        Layer &next_layer = m_layers[layer + 1];

        #pragma omp parallel for
        for (unsigned n = 0; n < hidden_layer.size(); n++) {
            hidden_layer[n].calc_hidden_gradients(next_layer);
        }
    }


    // update connection weights
    for (unsigned layer = m_layers.size() - 1; layer > 0; layer--) {
        Layer &layer_ = m_layers[layer];
        Layer &prev_layer = m_layers[layer - 1];

        #pragma omp parallel for
        for (Neuron& neuron : layer_) {
            neuron.update_input_weights(prev_layer);
        }
    }
}

void Net::get_results(std::vector<netnum_t> &result) const {
    result.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; n++) {
        result.push_back(m_layers.back()[n].get_output());
    }
}

void Net::train(
    const std::vector<std::vector<netnum_t>> &input_vals, 
    const std::vector<std::vector<netnum_t>> &target_vals,
    unsigned num_passes
) {
    assert(input_vals.size() == target_vals.size());
    
    // input the same size as input layer
    if (input_vals[0].size() != m_layers[0].size() - 1) {
        std::cerr << "Input size does not match input layer size" << std::endl;

        std::cerr << "Input size: " << input_vals[0].size() << std::endl;
        std::cerr << "Input layer size: " << m_layers[0].size() - 1 << std::endl;

        exit(1);
    }

    // target the same size as output layer
    if (target_vals[0].size() != m_layers.back().size() - 1) {
        std::cerr << "Target size does not match output layer size" << std::endl;

        std::cerr << "Target size: " << target_vals[0].size() << std::endl;
        std::cerr << "Output layer size: " << m_layers.back().size() - 1 << std::endl;

        exit(1);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto time_per_pass = start_time;

    for (unsigned pass = 0; pass < num_passes; pass++) {
        for (unsigned i = 0; i < input_vals.size(); i++) {
            feed_forward(input_vals[i]);
            back_prop(target_vals[i]);
        }

        if (pass % 100 == 0) {
            std::cout << "Pass: " << pass << "\tError: " << std::setprecision(6) << m_recent_avg_error << "\tTime per pass: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - time_per_pass).count() << "ms" << std::endl;
            time_per_pass = std::chrono::high_resolution_clock::now();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Training time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;

}

void Net::predict(const netnum_vec_t &input_vals, netnum_vec_t &result_vals) {
    feed_forward(input_vals);
    get_results(result_vals);
}

void Net::save_model(const char *filename) {
    std::ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // clear file
    file.seekp(0, std::ios::beg);

    // save topology
    unsigned num_layers = m_layers.size();
    file.write((char*)&num_layers, sizeof(unsigned));

    // save neurons per layer
    for (unsigned layer = 0; layer < num_layers; layer++) {
        unsigned num_neurons = m_layers[layer].size() - 1;
        file.write((char*)&num_neurons, sizeof(unsigned));
    }

    // save weights
    for (unsigned layer = 0; layer < num_layers; layer++) {
        for (unsigned n = 0; n < m_layers[layer].size(); n++) {
            Neuron &neuron = m_layers[layer][n];

            unsigned num_weights = neuron.m_output_weights.size();
            file.write((char*)&num_weights, sizeof(unsigned));

            for (unsigned w = 0; w < num_weights; w++) {
                file.write((char*)&neuron.m_output_weights[w].weight, sizeof(netnum_t));
                file.write((char*)&neuron.m_output_weights[w].d_weight, sizeof(netnum_t));
            }
        }
    }

    file.close();
}


Net Net::load_model(const char *filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return Net();
    }

    // load topology
    unsigned num_layers;
    file.read((char*)&num_layers, sizeof(unsigned));

    std::vector<unsigned> topology;
    topology.resize(num_layers);

    // load neurons per layer
    for (unsigned layer = 0; layer < num_layers; layer++) {
        unsigned num_neurons;
        file.read((char*)&num_neurons, sizeof(unsigned));
        topology[layer] = num_neurons;
    }

    Net net(topology);

    // load weights
    for (unsigned layer = 0; layer < num_layers; layer++) {
        for (unsigned n = 0; n < net.m_layers[layer].size(); n++) {
            Neuron &neuron = net.m_layers[layer][n];

            unsigned num_weights;
            file.read((char*)&num_weights, sizeof(unsigned));

            for (unsigned w = 0; w < num_weights; w++) {
                file.read((char*)&neuron.m_output_weights[w].weight, sizeof(netnum_t));
                file.read((char*)&neuron.m_output_weights[w].d_weight, sizeof(netnum_t));
            }
        }
    }

    file.close();

    return net;
}

