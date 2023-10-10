#include "../include/neural-net.h"
#include <assert.h>

Net nn_create_net(unsigned num_layers, unsigned *input_layer_sizes, unsigned thread_count) {
    omp_set_num_threads(thread_count);

    assert(num_layers > 1);
    assert(num_layers <= NN_MAX_LAYERS);

    Net net;

    net.num_layers = num_layers;
    for (unsigned i = 0; i < net.num_layers; i++) {
        if (input_layer_sizes[i] > 0) {
            net.layer_sizes[i] = input_layer_sizes[i] + 1;
        }
    }

    net.eta = NN_ETA;
    net.alpha = NN_ALPHA;

    net.error = 0.0;
    net.recent_average_error = 0.0;
    net.recent_average_smoothing_factor = NN_RECENT_AVERAGE_SMOOTHING_FACTOR;
    

    for (unsigned layer_index = 0; layer_index < num_layers; layer_index++) {
        unsigned layer_size = net.layer_sizes[layer_index];
        assert(layer_size > 0);
        assert(layer_size <= NN_MAX_LAYER_SIZE);

        unsigned num_outputs = (layer_index == num_layers - 1) ? 0 : net.layer_sizes[layer_index + 1];

        unsigned last;
        for (last = 0; last < layer_size; last++) {
            Neuron neuron = nn_create_neuron(num_outputs, last, &net);
            net.layers[layer_index][last] = neuron;
        }

        net.layers[layer_index][last-1].output = 1.0;
    }

    return net;
}

void nn_feed_forward(Net *net, netnum_t *input, unsigned input_size) {
    unsigned l1_size = net->layer_sizes[0];
    assert(input_size == l1_size - 1);

    for (unsigned i = 0; i < input_size; i++) {
        net->layers[0][i].output = input[i];
    }

    for (unsigned layer = 1; layer < net->num_layers; layer++) {
        Neuron *prev_layer = net->layers[layer - 1];

        // #pragma omp parallel for
        unsigned limit = net->layer_sizes[layer] - 1;
        for (unsigned neuron = 0; neuron < limit; neuron++) {
            Neuron *to_be_fed = &net->layers[layer][neuron];
            unsigned prev_layer_size = net->layer_sizes[layer - 1];
            nn_neuron_feed_forward(to_be_fed, prev_layer, prev_layer_size, neuron);
        }
    }
}

void nn_back_prop(Net *net, netnum_t *expected_output, unsigned expected_size) {    
    Neuron *output_layer = net->layers[net->num_layers - 1];
    unsigned output_layer_size = net->layer_sizes[net->num_layers - 1];

    assert(expected_size == output_layer_size - 1);

    net->error = 0.0;

    for (unsigned n = 0; n < output_layer_size - 1; n++) {
        netnum_t expected = expected_output[n];
        netnum_t actual = output_layer[n].output;
        netnum_t delta = expected - actual;
        net->error += delta * delta;
    }

    net->error /= output_layer_size - 1;
    net->error = sqrt(net->error);

    net->recent_average_error = (net->recent_average_error * net->recent_average_smoothing_factor + net->error) / (net->recent_average_smoothing_factor + 1.0);

    for (unsigned n = 0; n < output_layer_size - 1; n++) {
        nn_neuron_calc_output_gradients(&output_layer[n], expected_output[n]);
    }

    for (unsigned layer = net->num_layers - 2; layer > 0; layer--) {
        Neuron *hidden_layer = net->layers[layer];
        Neuron *next_layer = net->layers[layer + 1];

        // #pragma omp parallel for
        for (unsigned n = 0; n < net->layer_sizes[layer]; n++) {
            nn_neuron_calc_hidden_gradients(&hidden_layer[n], next_layer, net->layer_sizes[layer + 1]);
        }
    }

    for (unsigned l = net->num_layers - 1; l > 0; l--) {
        Neuron *layer = net->layers[l];
        Neuron *prev_layer = net->layers[l - 1];

        // #pragma omp parallel for
        for (unsigned n = 0; n < net->layer_sizes[l] - 1; n++) {
            nn_neuron_update_input_weights(&layer[n], prev_layer, net->layer_sizes[l - 1]);
        }
    }
}

void nn_get_results(Net *net, netnum_t *result, unsigned result_size) {
    assert(result_size == net->layer_sizes[net->num_layers - 1] - 1);

    for (unsigned n = 0; n < result_size; n++) {
        result[n] = net->layers[net->num_layers - 1][n].output;
    }
}