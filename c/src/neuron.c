#include "../include/neural-net.h"
#include <assert.h>

netnum_t random_weight() {
    return rand() / (netnum_t)(RAND_MAX);
}

netnum_t transfer_function(netnum_t num) {
    // return 1.0 / (1.0 + exp(-num)); // sigmoid
    // return num / (1 + fabsf(num)); // fast sigmoid
    return tanh(num); // tanh
}

netnum_t transfer_function_derv(netnum_t num) {
    // using derivative of sigmoid
    // return 1.0 / (1.0 + fabs(num)) * 1.0 / (1.0 + fabs(num)); // (1 - o) simplifies to o squared
    return 1.0 - num * num;
}

Neuron nn_create_neuron(unsigned num_outputs, unsigned my_index, Net *net) {
    Neuron neuron;

    neuron.output = 0.0;
    neuron.index = my_index;
    neuron.gradient = 0.0;

    neuron.net = net;

    neuron.output_weights_size = num_outputs;

    for (unsigned c = 0; c < num_outputs; c++) {
        Connection con = {random_weight(), 0};
        neuron.output_weights[c] = con;
    }

    return neuron;
}

void nn_neuron_feed_forward(Neuron *neuron, Neuron *prev_layer, unsigned prev_layer_size, unsigned index) {
    netnum_t sum = 0.0;

    for (unsigned i = 0; i < prev_layer_size; i++) {

        netnum_t output = prev_layer[i].output;
        netnum_t weight = prev_layer[i].output_weights[index].weight;

        sum += weight * output;
    }

    neuron->output = transfer_function(sum);
}

void nn_neuron_calc_output_gradients(Neuron *neuron, netnum_t expected_output) {
    netnum_t delta = expected_output - neuron->output;
    neuron->gradient = delta * transfer_function_derv(neuron->output);
}

void nn_neuron_calc_hidden_gradients(Neuron *neuron, Neuron *next_layer, unsigned next_layer_size) {
    netnum_t dow = 0.0;

    for (unsigned n = 0; n < next_layer_size - 1; n++) {
        dow += neuron->output_weights[n].weight * next_layer[n].gradient;
    }

    neuron->gradient = dow * transfer_function_derv(neuron->output);
}

void nn_neuron_update_input_weights(Neuron *neuron, Neuron *prev_layer, unsigned prev_layer_size) {
    for (unsigned n = 0; n < prev_layer_size; n++) {
        Neuron *prev_neuron = &prev_layer[n];
        Connection *con = &prev_neuron->output_weights[neuron->index];

        netnum_t old_delta_weight = con->d_weight;
        netnum_t eta = (neuron->net)->eta;
        netnum_t alpha = (neuron->net)->alpha;
        netnum_t new_delta_weight = eta * prev_neuron->output * neuron->gradient + alpha * old_delta_weight;

        con->d_weight = new_delta_weight;
        con->weight += new_delta_weight;
    }
}