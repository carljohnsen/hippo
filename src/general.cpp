#include "hippo.hpp"

// TODO handle aligned allocation
template <typename T>
std::vector<T> load_file(const std::string &path, const int64_t offset, const int64_t n_elements) {
    std::vector<T> data(n_elements);
    load_file(data.data(), path, offset, n_elements);
    return data;
}

template <typename T>
void load_file(T *dst, const std::string &path, const int64_t offset, const int64_t n_elements) {
    FILE *fp;
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    fp = fdopen(fd, "rb");
    fseek(fp, offset*sizeof(T), SEEK_SET);
    int64_t n = fread((char *) dst, sizeof(T), n_elements, fp);
    assert(n == n_elements && "Failed to read all elements");
    fclose(fp);
}

// TODO handle aligned allocation
template <typename T>
std::vector<T> load_file_strided(const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range) {
    std::vector<T> data(shape.z*shape.y*shape.x);
    load_file_strided(data.data(), path, disk_shape, shape, range);
    return data;
}

template <typename T>
void load_file_strided(T *dst, const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range) {
    const idx3d
        strides_global = {disk_shape.y*disk_shape.x, disk_shape.x, 1},
        strides_local = {shape.y*shape.x, shape.x, 1};

    if (shape.z == disk_shape.z && shape.y == disk_shape.y && shape.x == disk_shape.x) {
        load_file(dst, path, 0, disk_shape.z*disk_shape.y*disk_shape.x);
        return;
    }

    FILE *fp;
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    fp = fdopen(fd, "rb");
    fseek(fp, range.z_start*strides_global.z + range.y_start*strides_global.y + range.x_start*strides_global.x*sizeof(T), SEEK_SET);
    for (int64_t z = 0; z < shape.z; z++) {
        for (int64_t y = 0; y < shape.y; y++) {
            int64_t n = fread((char *) &dst[z*strides_local.z + y*strides_local.y], sizeof(T), shape.x, fp);
            assert(n == shape.x && "Failed to read all elements");
            fseek(fp, (strides_global.y - shape.x)*sizeof(T), SEEK_CUR);
        }
        fseek(fp, (strides_global.y - shape.y)*strides_global.y*sizeof(T), SEEK_CUR);
    }
    fclose(fp);
}

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

template <typename T>
void store_file(const std::vector<T> &data, const std::string &path, const int64_t offset) {
    std::ofstream file;
    file.open(path, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        file.clear();
        file.open(path, std::ios::binary | std::ios::out);
    }
    file.seekp(offset*sizeof(T), std::ios::beg);
    file.write(reinterpret_cast<const char*>(data.data()), data.size()*sizeof(T));
    file.flush();
    file.close();
}

template <typename T>
void store_file(const T *data, const std::string &path, const int64_t offset, const int64_t n_elements) {
    FILE *fp;
    int fd = open(path.c_str(), O_CREAT | O_RDWR | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    fp = fdopen(fd, "r+b");
    fseek(fp, offset*sizeof(T), SEEK_SET);
    fwrite((char *) data, sizeof(T), n_elements, fp);
    fclose(fp);
}

template <typename T>
void store_file_strided(const std::vector<T> &data, const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range) {
    store_file_strided(data.data(), path, disk_shape, shape, range);
}

template <typename T>
void store_file_strided(const T *data, const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range) {
    const idx3d
        strides_global = {disk_shape.y*disk_shape.x, disk_shape.x, 1},
        strides_local = {shape.y*shape.x, shape.x, 1};

    if (shape.z == disk_shape.z && shape.y == disk_shape.y && shape.x == disk_shape.x) {
        store_file(data, path, 0, disk_shape.z*disk_shape.y*disk_shape.x);
        return;
    }

    FILE *fp;
    int fd = open(path.c_str(), O_CREAT | O_RDWR | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    fp = fdopen(fd, "r+b");
    fseek(fp, range.z_start*strides_global.z + range.y_start*strides_global.y + range.x_start*strides_global.x*sizeof(T), SEEK_SET);
    for (int64_t z = 0; z < shape.z; z++) {
        for (int64_t y = 0; y < shape.y; y++) {
            fwrite((char *) &data[z*strides_local.z + y*strides_local.y], sizeof(T), shape.x, fp);
            fseek(fp, (strides_global.y - shape.x)*sizeof(T), SEEK_CUR);
        }
        fseek(fp, (strides_global.y - shape.y)*strides_global.y*sizeof(T), SEEK_CUR);
    }
    fclose(fp);
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

// Explicit instantiations
template std::vector<float> load_file<float>(const std::string &path, const int64_t offset, const int64_t n_elements);
template std::vector<uint8_t> load_file<uint8_t>(const std::string &path, const int64_t offset, const int64_t n_elements);
template std::vector<float> load_file_strided<float>(const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range);
template std::vector<uint8_t> load_file_strided<uint8_t>(const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range);
template void store_file<float>(const std::vector<float> &data, const std::string &path, const int64_t offset);
template void store_file<uint8_t>(const std::vector<uint8_t> &data, const std::string &path, const int64_t offset);
template void store_file<float>(const float *data, const std::string &path, const int64_t offset, const int64_t n_elements);
template void store_file<uint8_t>(const uint8_t *data, const std::string &path, const int64_t offset, const int64_t n_elements);
