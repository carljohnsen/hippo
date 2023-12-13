#include "hippo.hpp"

// Returns the mean of the given data
float mean(const std::vector<float>& data) {
    float sum = 0.0;
    for (int64_t i = 0; i < (int64_t) data.size(); i++) {
        sum += data[i];
    }
    return sum / data.size();
}

FILE* open_file_read(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    return fdopen(fd, "rb");
}

FILE* open_file_write(const std::string &path) {
    int fd = open(path.c_str(), O_CREAT | O_RDWR | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    return fdopen(fd, "r+b");
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