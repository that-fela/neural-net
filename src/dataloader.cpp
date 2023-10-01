#include "n-net.h"

#include "../lib/lodepng.h"
#include <cassert>

using namespace NNet;

std::vector<NNet::netnum_t> DataLoader::from_png(const char *filename, const std::vector<netnum_t> &target) {
    std::vector<unsigned char> image; // This will hold the raw pixel data.
    unsigned width, height;
    unsigned error = lodepng::decode(image, width, height, filename);

    if (error) {
        std::cout << "Error while loading PNG: " << lodepng_error_text(error) << std::endl;
    }

    std::vector<NNet::netnum_t> loaded_image;

    for (size_t i = 0; i < image.size(); i += 4) {
        auto red = image[i] / 255.0;
        auto green = image[i + 1] / 255.0;
        auto blue = image[i + 2] / 255.0;

        auto temp = (red + green + blue) / 3.0;
        auto temp2 = temp > 0.5 ? 0 : 1;

        loaded_image.push_back(temp2);
    }

    return loaded_image;
}

DataLoader DataLoader::from_png_folder(const char *path) {
    // get all files in folder
    std::vector<std::string> files;
    std::vector<std::string> filenames;

    for(const auto& entry : std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_regular_file(entry.path()) && entry.path().extension() == ".png") {
            filenames.push_back(entry.path().filename());

            // add path to filenamex
            auto full_path = std::string(path).append("/").append(entry.path().filename());
            files.push_back(full_path);
        }
    }

    assert(files.size() > 0);

    DataLoader loader;

    // iterate through files, and load them
    for (auto &file : files) {
        auto input = loader.from_png(file.c_str(), {});

        loader.m_input_values.push_back(input);
    }

    for (auto &filename : filenames) {
        auto target = std::vector<netnum_t>(10, 0.0);

        auto number = filename.substr(0, filename.find("."));
        auto number_int = std::stoi(number);

        target[number_int] = 1.0;

        loader.m_target_values.push_back(target);
    }

    return loader;
}