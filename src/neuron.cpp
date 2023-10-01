#include "n-net.h"

using namespace NNet;

netnum_t random_weight() {
    return rand() / (netnum_t)(RAND_MAX);
}

netnum_t transfer_func(netnum_t num) {
    // using sigmoid
    return 1.0 / (1.0 + exp(-num));
    // return tanh(num);
}

netnum_t transfer_func_derv(netnum_t num) {
    // using derivative of sigmoid
    netnum_t o = 1.0 / (1.0 + exp(-num));
    return o * (1 - o);
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
    netnum_t sum = 0.0;

    for (unsigned n = 0; n < previous_layer.size(); n++) {
        sum += previous_layer[n].m_output * previous_layer[n].m_output_weights[index].weight;
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
    for (unsigned n = 0; n < previous_layer.size(); n++) {
        Neuron &neuron = previous_layer[n];

        netnum_t old_d_weight = neuron.m_output_weights[m_index].d_weight;
        netnum_t new_d_weight = neuron.m_net->eta * neuron.m_output * m_gradient + neuron.m_net->alpha * old_d_weight;

        neuron.m_output_weights[m_index].d_weight = new_d_weight;
        neuron.m_output_weights[m_index].weight += new_d_weight;
    }
}