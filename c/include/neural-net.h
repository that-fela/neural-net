#ifndef _NEURAL_NET_H_
#define _NEURAL_NET_H_

#define NN_ETA      0.15
#define NN_ALPHA    0.5
#define NN_RECENT_AVERAGE_SMOOTHING_FACTOR 100.0

#define NN_MAX_LAYERS       5
#define NN_MAX_LAYER_SIZE   402
#define NN_CONNECTIONS      402

#include <setjmp.h>
#include <asm-generic/errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>

typedef float netnum_t;
typedef struct Neuron Neuron;
typedef struct Net Net;
typedef netnum_t** MatInput;

typedef struct {
    netnum_t weight;
    netnum_t d_weight;
} Connection;

struct Neuron{
    netnum_t output;
    unsigned index;
    netnum_t gradient;

    Net *net;

    unsigned output_weights_size;
    Connection output_weights[NN_CONNECTIONS];
};

struct Net{
    unsigned num_layers;
    unsigned layer_sizes[NN_MAX_LAYERS];

    netnum_t eta; // [0.0..1.0] overall net training rate
    netnum_t alpha; // [0.0..n] multiplier of last weight change (momentum)

    netnum_t error;
    netnum_t recent_average_error;
    netnum_t recent_average_smoothing_factor;

    Neuron layers[NN_MAX_LAYERS][NN_MAX_LAYER_SIZE];
};

// ==================== //
// Net functions
// ==================== //

// Public
Net         nn_create_net(unsigned num_layers, unsigned *input_layer_sizes, unsigned thread_count);
netnum_t    nn_get_error(Net *net);
void        nn_train(
                Net *net, unsigned num_passes,
                netnum_t **train_input, unsigned train_input_size, unsigned train_input_width,
                netnum_t **expected_output, unsigned expected_output_size, unsigned expected_output_width
            );
void        nn_predict(Net *net, netnum_t *input, unsigned input_size, netnum_t *results, unsigned result_size);

// Private
void nn_feed_forward(Net *net, netnum_t *input, unsigned input_size);
void nn_back_prop(Net *net, netnum_t *expected_output, unsigned expected_size);
void nn_get_results(Net *net, netnum_t *results, unsigned result_size);


// ==================== //
// Neuron functions
// ==================== //

// Public
Neuron      nn_create_neuron(unsigned num_outputs, unsigned my_index, Net *net);

void        nn_neuron_feed_forward(Neuron *neuron, Neuron *prev_layer, unsigned prev_layer_size, unsigned index);
void        nn_neuron_calc_output_gradients(Neuron *neuron, netnum_t expected_output);
void        nn_neuron_calc_hidden_gradients(Neuron *neuron, Neuron *next_layer, unsigned next_layer_size);
void        nn_neuron_update_input_weights(Neuron *neuron, Neuron *prev_layer, unsigned prev_layer_size);

#endif // _NEURAL_NET_H_