#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include <cstddef>
#include <math.h>
#include <cstdlib>
#include <fstream>

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
        netnum_t eta = 0.15; // [0.0..1.0] overall net training rate
        netnum_t alpha = 0.5; // [0.0..n] multiplier of last weight change (momentum)

        Net(const std::vector<unsigned> &topology);
        Net() {}
        ~Net();

        void train(
            const std::vector<std::vector<netnum_t>> &input_vals, 
            const std::vector<std::vector<netnum_t>> &target_vals,
            unsigned num_passes
        );
        void predict(const std::vector<netnum_t> &input_vals, std::vector<netnum_t> &result_vals);

        void        save_model(const char *filename);
        static Net  load_model(const char *filename);

        netnum_t get_recent_avg_error() const { return m_recent_avg_error; }

    private:
        void feed_forward(const std::vector<netnum_t> &input);
        void back_prop(const std::vector<netnum_t> &target);
        void get_results(std::vector<netnum_t> &result) const;

        std::vector<Layer> m_layers;
        netnum_t           m_error;
        netnum_t           m_recent_avg_error;
        netnum_t           m_recent_avg_smoothing_factor = 100;
    };


    // -------------
    // Neuron
    // -------------
    class Neuron {
    public:
        Neuron(unsigned num_outputs, unsigned index, const Net *net_ref);

        void                        set_output(const netnum_t output) { m_output = output; }
        netnum_t                    get_output() const { return m_output; }

        void        feed_forward(const Layer &previous_layer, unsigned index);
        void        calc_output_gradients(netnum_t target);
        void        calc_hidden_gradients(const Layer &next_layer);
        void        update_input_weights(Layer &previous_layer);
        netnum_t    sum_derv_of_weights(const Layer &next_layer) const;

        std::vector<Connection> m_output_weights;
    private:
        netnum_t                m_output;
        unsigned                m_index;
        netnum_t                m_gradient;
        const Net               *m_net;
    };
}
