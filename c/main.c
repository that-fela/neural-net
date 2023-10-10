#include <stdio.h>

#include "include/neural-net.h"

int main(){
    unsigned layer_sizes[3] = {2, 3, 1};

    Net net = nn_create_net(3, layer_sizes, 1);

    // simulating XOR
    netnum_t inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    netnum_t outputs[4][1] = {{0}, {1}, {1}, {0}};

    for (unsigned i = 0; i < 10000; i++) {
        for (unsigned j = 0; j < 4; j++) {
            nn_feed_forward(&net, inputs[j], 2);
            nn_back_prop(&net, outputs[j], 1);
        }

        if (i % 1000 == 0) {
            printf("Error: %f\n", net.error);
        }
    }

    for (unsigned i = 0; i < 4; i++) {
        nn_feed_forward(&net, inputs[i], 2);
        printf("%f\n", net.layers[net.num_layers - 1][0].output);
    }
}
