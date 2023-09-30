#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include <cstddef>
#include <math.h>
#include <cstdlib>

namespace NNet {
    class Neuron;
    typedef std::vector<Neuron> Layer;
    typedef double netnum_t;

    struct Connection {
        netnum_t weight;
        netnum_t d_weight;
    };


    // -------------
    // Net
    // -------------
    class Net {
    public:
        Net(const std::vector<unsigned> &topology);
        void feed_forward(const std::vector<netnum_t> &input);
        void back_prop(const std::vector<netnum_t> &target);
        void get_results(std::vector<netnum_t> &result) const;

    private:
        std::vector<Layer> m_layers;
        netnum_t           m_error;
        netnum_t           m_recent_avg_error;
        netnum_t           m_recent_avg_smoothing_factor;
    };


    // -------------
    // Neuron
    // -------------
    class Neuron {
    public:
        Neuron(unsigned num_outputs, unsigned index);

        void        set_output(const netnum_t output) { m_output = output; }
        netnum_t    get_output() const { return m_output; }
        void        feed_forward(const Layer &previous_layer, unsigned index);
        void        calc_output_gradients(netnum_t target);
        void        calc_hidden_gradients(const Layer &next_layer);
        void        update_input_weights(Layer &previous_layer);

    private:
        netnum_t                m_output;
        unsigned                m_index;
        std::vector<Connection> m_output_weights;
    };
}
