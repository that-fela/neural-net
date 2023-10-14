// -------------
// DataLoader
// -------------
#pragma once

#include "n-net.h"

#include <cassert>
#include <iostream>
#include <filesystem>

class DataLoader {
public:
    typedef NNet::netnum_t netnum_t;
    typedef std::vector<netnum_t> netnum_vec_t;
    typedef std::vector<std::vector<netnum_t>> netnum_mat_t;

    DataLoader() {}
    ~DataLoader() {}

    static DataLoader from_png_folder(const char *filename);
    static netnum_vec_t from_png(const char *filename, const std::vector<float> &target);

    netnum_mat_t get_input_values() const { return m_input_values; }
    netnum_mat_t get_target_values() const { return m_target_values; }
private:

    netnum_mat_t m_input_values;
    netnum_mat_t m_target_values;
};