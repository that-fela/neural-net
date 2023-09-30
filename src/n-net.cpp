#include "n-net.h"

using namespace NNet;

Net::Net(const std::vector<unsigned> &topology){
    unsigned num_layers = topology.size();
    assert(num_layers > 1);

    for (unsigned layer = 0; layer < num_layers; layer++) {
        m_layers.push_back(Layer());

        unsigned num_outputs = (layer == num_layers - 1) ? 0 : topology[layer + 1];

        for (unsigned neuron = 0; neuron <= topology[layer]; neuron++) {
            m_layers.back().push_back(Neuron(num_outputs, neuron));
        }
    }
}

void Net::feed_forward(const std::vector<netnum_t> &input_vals) {
    assert(input_vals.size() == m_layers[0].size() - 1);

    for (unsigned i = 0; i < input_vals.size(); i++) {
        m_layers[0][i].set_output(input_vals[i]);
    }

    for (unsigned layer = 1; layer < m_layers.size(); layer++) {
        Layer &previous_layer = m_layers[layer - 1];

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

    // calc output layer gradients
    for (unsigned n = 0; n < output_layer.size() - 1; n++) {
        output_layer[n].calc_output_gradients(target[n]);
    }

    // calc hidden layer gradients
    for (unsigned layer = m_layers.size() - 2; layer > 0; layer--) {
        Layer &hidden_layer = m_layers[layer];
        Layer &next_layer = m_layers[layer + 1];

        for (unsigned n = 0; n < hidden_layer.size(); n++) {
            hidden_layer[n].calc_hidden_gradients(next_layer);
        }
    }

    // update connection weights
    for (unsigned layer = m_layers.size() - 1; layer > 0; layer--) {
        Layer &layer_ = m_layers[layer];
        Layer &prev_layer = m_layers[layer - 1];

        for (unsigned n = 0; n < layer_.size() - 1; n++) {
            layer_[n].update_input_weights(prev_layer);
        }
    }
}