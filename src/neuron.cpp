#include "n-net.h"

using namespace NNet;

netnum_t random_weight() {
    return rand() / (netnum_t)(RAND_MAX);
}

netnum_t transfer_func(netnum_t sum) {
    // using sigmoid
    return 1 / (1 + exp(-sum));
}

netnum_t transfer_func_derv(netnum_t sum) {
    // using derivative of sigmoid
    return (exp(-sum) / pow(1 + exp(-sum), 2));
}

Neuron::Neuron(unsigned num_outputs, unsigned index) {
    m_index = index;

    for (unsigned conns = 0; conns < num_outputs; conns++) {
        m_output_weights.push_back({random_weight(), 0});
    }
}

void Neuron::feed_forward(const Layer &previous_layer, unsigned index) {
    netnum_t sum = 0.0;

    for (unsigned n = 0; n < previous_layer.size(); n++) {
        sum += previous_layer[n].m_output * previous_layer[n].m_output_weights[index].weight;
    }

    m_output = transfer_func(sum);
}