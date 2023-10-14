#include <gtest/gtest.h>
#include "../include/n-net.h"

using namespace NNet;

TEST(NeuronTest, FeedForward) {
    // Create a previous layer with 2 neurons
    Layer previous_layer;
    previous_layer.push_back(Neuron(2, 0, nullptr));
    previous_layer.push_back(Neuron(2, 1, nullptr));

    // Create a neuron with 2 output connections
    Neuron neuron(2, 0, nullptr);

    // Set the weights of the output connections
    neuron.m_output_weights[0].weight = 0.5;
    neuron.m_output_weights[1].weight = -0.5;

    // Set the output of the previous layer neurons
    previous_layer[0].set_output(1.0);
    previous_layer[1].set_output(-1.0);

    // Feed forward
    neuron.feed_forward(previous_layer, 0);

    // Check the output
    EXPECT_FLOAT_EQ(neuron.get_output(), 0.054005407);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}