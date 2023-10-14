#include <stdio.h>

#include "include/neural-net.h"
#include "include/imageloader.h"

void XOR_test();
void large_test();

int main(){
    // XOR_test();
    large_test();

    return 0;
}

const unsigned data_length = 10;
netnum_t *get_large() {
    netnum_t *inputs = (netnum_t*)malloc(data_length * 400 * sizeof(netnum_t));

    for (unsigned i = 0; i < data_length; i++) {
        for (unsigned j = 0; j < 400; j++) {
            inputs[i * 400 + j] = j / 400;
        }
    }

    return inputs;
}

netnum_t *get_large_awns() {
    netnum_t *inputs = (netnum_t*)malloc(data_length * 10 * sizeof(netnum_t));

    for (unsigned i = 0; i < data_length; i++) {
        for (unsigned j = 0; j < 10; j++) {
            inputs[i * 10 + j] = j / 10;
        }
    }

    return inputs;
}

void large_test() {
    ImageData idata = get_images();

    netnum_t *inputs = idata.images;
    netnum_t *outputs = idata.labels;

    printf("input size: %dx%d\n", data_length, 400);

    unsigned layer_sizes[4] = {400, 360, 80, 4};
    Net net = nn_create_net(4, layer_sizes, 8);

    nn_train(&net, 1000, (MatInput)inputs, data_length, 400, (MatInput)outputs, data_length, 1);
}

void XOR_test() {
    unsigned layer_sizes[3] = {2, 3, 1};

    Net net = nn_create_net(3, layer_sizes, 1);

    // simulating XOR
    netnum_t inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    netnum_t outputs[4][1] = {{0}, {1}, {1}, {0}};

    nn_train(&net, 100, (MatInput)inputs, 4, 2, (MatInput)outputs, 4, 1);

    for (unsigned i = 0; i < 4; i++) {
        nn_feed_forward(&net, inputs[i], 2);
        printf("%f\n", net.layers[net.num_layers - 1][0].output);
    }
}