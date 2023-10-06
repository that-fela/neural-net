#include "n-net.h"

using namespace NNet;

netnum_t random_weight() {
    return rand() / (netnum_t)(RAND_MAX);
}

netnum_t transfer_func(netnum_t num) {
    // return 1.0 / (1.0 + exp(-num)); // sigmoid
    return num / (1 + abs(num)); // fast sigmoid
    // return tanh(num); // tanh
}

netnum_t transfer_func_derv(netnum_t num) {
    // using derivative of sigmoid
    return 1.0 / (1.0 + fabs(num)) * 1.0 / (1.0 + fabs(num)); // (1 - o) simplifies to o squared
    // return 1.0 - num * num;
}

Neuron::Neuron(unsigned num_outputs, unsigned index, const Net *net_ref) {
    m_index = index;
    m_net = net_ref;

    for (unsigned conns = 0; conns < num_outputs; conns++) {
        m_output_weights.push_back({random_weight(), 0});
    }
}

void Neuron::feed_forward(const Layer &previous_layer, unsigned index) {
    size_t size = previous_layer.size(); // Store the size before the loop

    netnum_t sum = 0.0;
    for (size_t n = 0; n < size; n++) {
        const Neuron& neuron = previous_layer[n];
        const Connection& weight = neuron.m_output_weights[index];

        sum += neuron.m_output * weight.weight;
    }

    m_output = transfer_func(sum);
}

void Neuron::calc_output_gradients(netnum_t target) {
    netnum_t delta = target - m_output;
    m_gradient = delta * transfer_func_derv(m_output);
}

void Neuron::calc_hidden_gradients(const Layer &next_layer) {
    netnum_t dow = sum_derv_of_weights(next_layer);
    m_gradient = dow * transfer_func_derv(m_output);
}

netnum_t Neuron::sum_derv_of_weights(const Layer &next_layer) const {
    netnum_t sum = 0.0;

    for (unsigned n = 0; n < next_layer.size() - 1; n++) {
        sum += m_output_weights[n].weight * next_layer[n].m_gradient;
    }

    return sum;
}

void Neuron::update_input_weights(Layer &previous_layer) {
    size_t size = previous_layer.size(); // Store the size before the loop

    for (size_t n = 0; n < size; n++) {
        Neuron &neuron = previous_layer[n];
        auto &output_weight = neuron.m_output_weights[m_index];

        netnum_t old_d_weight = output_weight.d_weight;
        netnum_t new_d_weight = neuron.m_net->eta * neuron.m_output * m_gradient + neuron.m_net->alpha * old_d_weight;

        output_weight.d_weight = new_d_weight;
        output_weight.weight += new_d_weight;
    }
}