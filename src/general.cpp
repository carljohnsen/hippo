#include "hippo.hpp"

// Returns the mean of the given data
float mean(const std::vector<float>& data) {
    float sum = 0.0;
    for (int64_t i = 0; i < (int64_t) data.size(); i++) {
        sum += data[i];
    }
    return sum / data.size();
}

// Returns the standard deviation of the given data
float stddev(const std::vector<float>& data) {
    float m = mean(data);
    float sum = 0.0;
    for (int64_t i = 0; i < (int64_t) data.size(); i++) {
        sum += std::pow(data[i] - m, 2.0);
    }
    return std::sqrt(sum / data.size());
}

void write_pgm(const std::string &filename, const std::vector<float> &data, const int64_t width, const int64_t height) {
    std::ofstream file(filename);
    file << "P2\n" << width << " " << height << "\n255\n";
    for (int64_t i = 0; i < height; i++) {
        for (int64_t j = 0; j < width; j++) {
            file << (int64_t) (data[i*width + j] * 255) << " ";
        }
        file << "\n";
    }
    file.close();
}