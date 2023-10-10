#include <vector>
#include <iostream>
#include <cassert>
#include <cstddef>
#include <math.h>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <omp.h>

namespace NNet {
    class Neuron;
    typedef std::vector<Neuron> Layer;
    typedef float netnum_t;

    typedef std::vector<netnum_t> netnum_vec_t;
    typedef std::vector<std::vector<netnum_t>> netnum_mat_t;

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

        Net(const std::vector<unsigned> &topology, unsigned num_threads = 1);
        Net() {}
        ~Net();

        void train(
            const netnum_mat_t &input_vals, 
            const netnum_mat_t &target_vals,
            unsigned num_passes
        );
        void predict(const netnum_vec_t &input_vals, netnum_vec_t &result_vals);

        void        save_model(const char *filename);
        static Net  load_model(const char *filename);

        netnum_t get_recent_avg_error() const { return m_recent_avg_error; }

    private:
        void feed_forward(const netnum_vec_t &input);
        void back_prop(const netnum_vec_t &target);
        void get_results(netnum_vec_t &result) const;

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


    // -------------
    // DataLoader
    // -------------
    // class DataLoader {
    // public:
    //     DataLoader() {}
    //     ~DataLoader() {}

    //     static DataLoader from_png_folder(const char *filename);
    //     static netnum_vec_t from_png(const char *filename, const std::vector<netnum_t> &target);

    //     netnum_mat_t get_input_values() const { return m_input_values; }
    //     netnum_mat_t get_target_values() const { return m_target_values; }
    // private:

    //     netnum_mat_t m_input_values;
    //     netnum_mat_t m_target_values;
    // };
}
