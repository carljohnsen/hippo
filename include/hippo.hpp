#ifndef HIPPO_HPP
#define HIPPO_HPP

// Includes
//#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <omp.h>
// TODO pybind11 later on :)
//#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
//#include <pybind11/numpy.h>
#include <tuple>
#include <unordered_set>
#include <vector>

// Custom datatypes
typedef uint16_t field_type;
struct idx3d {
    int64_t z, y, x;
};
struct idx3drange {
    int64_t z_start, z_end, y_start, y_end, x_start, x_end;
};
typedef std::vector<std::unordered_set<int64_t>> mapping;
// TODO pybind11 later on :)
//template <typename T>
//using np_array = pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;
typedef uint16_t voxel_type;

// Constants
constexpr int64_t
    // On disk parameters for generated input
    Nz_total = 256,
    Ny_total = 256,
    Nx_total = 256,
    // Out-of-core main memory parameters
    Nz_global = 128,
    Ny_global = 128,
    Nx_global = 128,
    // Out-of-core GPU memory parameters
    Nz_local = 64,
    Ny_local = 64,
    Nx_local = 64,
    // Input image generation parameters
    C = 4;

// General function headers
// Returns the mean of the given data
float mean(const std::vector<float> &data);

// Returns the standard deviation of the given data
float stddev(const std::vector<float> &data);

// Writes the given 2D floating point array (between 0.0 and 1.0) to a PGM file
void write_pgm(const std::string &filename, const std::vector<float> &data, const int64_t width, const int64_t height);

// Templated functions
template<typename T>
FILE* open_file_read(const std::string &path) {
    //int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    int fd = open(path.c_str(), O_RDONLY);
    return fdopen(fd, "rb");
}

template<typename T>
FILE* open_file_write(const std::string &path) {
    //int fd = open(path.c_str(), O_CREAT | O_RDWR | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    int fd = open(path.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    return fdopen(fd, "r+b");
}

// Loads `n_elements` of a file located at `path` on disk at `offset` elements from the beginning of the file, into a vector of type `T`.
template <typename T>
void load_file(T *dst, const std::string &path, const int64_t offset, const int64_t n_elements) {
    FILE *fp = open_file_read<T>(path);
    fseek(fp, offset*sizeof(T), SEEK_SET);
    int64_t n = fread((char *) dst, sizeof(T), n_elements, fp);
    assert(n == n_elements && "Failed to read all elements");
    fclose(fp);
}

template <typename T>
std::vector<T> load_file(const std::string &path, const int64_t offset, const int64_t n_elements) {
    std::vector<T> data(n_elements);
    load_file(data.data(), path, offset, n_elements);
    return data;
}

// Reads the specified index `range` of a file located at `path` on disk which is of the given `shape`, into `dst`.
// `disk_shape` is the shape of the file on disk, and `shape` is the shape of the allocated memory.
// This version exists to avoid allocating a vector for each call, and reads directly from disk.
// The last stride is always assumed to be 1, for both src and dst.
// It is up to the caller to ensure that 1) `range` doesn't exceed `shape`, 2) `dst` is large enough to hold the data, 3) `dst` is set to 0 in case of a partial read and 0s are desired and 4) `dst` is an aligned allocation (e.g. using `aligned_alloc()`) to maximize performance.
template <typename T>
void load_file_strided(T *dst, const std::string &path, const idx3d &shape_total, const idx3d &shape_global, const idx3drange &range, const idx3d &offset_global) {
    const idx3d
        strides_total = {shape_total.y*shape_total.x, shape_total.x, 1},
        strides_global = {shape_global.y*shape_global.x, shape_global.x, 1},
        sizes = {range.z_end - range.z_start, range.y_end - range.y_start, range.x_end - range.x_start};

    if (shape_global.z == shape_total.z && shape_global.y == shape_total.y && shape_global.x == shape_total.x) {
        load_file(dst, path, 0, shape_total.z*shape_total.y*shape_total.x);
        return;
    }

    FILE *fp = open_file_read<T>(path);
    fseek(fp, (range.z_start*strides_total.z + range.y_start*strides_total.y + range.x_start*strides_total.x)*sizeof(T), SEEK_SET);
    for (int64_t z = 0; z < sizes.z; z++) {
        for (int64_t y = 0; y < sizes.y; y++) {
            int64_t n = fread((char *) &dst[(z+offset_global.z)*strides_global.z + (y+offset_global.y)*strides_global.y + offset_global.x*strides_global.x], sizeof(T), sizes.x, fp);
            assert(n == sizes.x && "Failed to read all elements");
            fseek(fp, (strides_total.y - sizes.x)*sizeof(T), SEEK_CUR);
        }
        fseek(fp, (strides_total.z - sizes.y*strides_total.y)*sizeof(T), SEEK_CUR);
    }
    fclose(fp);
}

// Loads the specified index `range` of a file located at `path` on disk which is of the given `shape`, into a vector of type `T`.
template <typename T>
std::vector<T> load_file_strided(const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range, const idx3d &offset_global) {
    std::vector<T> data(shape.z*shape.y*shape.x);
    load_file_strided(data.data(), path, disk_shape, shape, range, offset_global);
    return data;
}

template <typename T>
void load_partial(T *__restrict__ dst, FILE *fp, const int64_t offset, const int64_t n_elements) {
    fseek(fp, offset*sizeof(T), SEEK_SET);
    int64_t n = fread((char *) dst, sizeof(T), n_elements, fp);
    //assert(n == n_elements && "Failed to read all elements");
}

// Stores `data.size()` elements of `data` into a file located at `path` on disk at `offset` elements from the beginning of the file.
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
    FILE *fp = open_file_write<T>(path);
    fseek(fp, offset*sizeof(T), SEEK_SET);
    fwrite((char *) data, sizeof(T), n_elements, fp);
    fclose(fp);
}

template <typename T>
void store_file_strided(const T *data, const std::string &path, const idx3d &shape_total, const idx3d &shape_global, const idx3drange &range, const idx3d &offset_global) {
    const idx3d
        strides_total = {shape_total.y*shape_total.x, shape_total.x, 1},
        strides_global = {shape_global.y*shape_global.x, shape_global.x, 1},
        sizes = {range.z_end - range.z_start, range.y_end - range.y_start, range.x_end - range.x_start};

    if (shape_global.z == shape_total.z && shape_global.y == shape_total.y && shape_global.x == shape_total.x) {
        store_file(data, path, 0, shape_total.z*shape_total.y*shape_total.x);
        return;
    }

    FILE *fp = open_file_write<T>(path);
    fseek(fp, (range.z_start*strides_total.z + range.y_start*strides_total.y + range.x_start*strides_total.x)*sizeof(T), SEEK_SET);
    for (int64_t z = 0; z < sizes.z; z++) {
        for (int64_t y = 0; y < sizes.y; y++) {
            int64_t n = fwrite((char *) &data[(z+offset_global.z)*strides_global.z + (y+offset_global.y)*strides_global.y + (0+offset_global.x)*strides_global.x], sizeof(T), sizes.x, fp);
            fseek(fp, (strides_total.y - sizes.x)*sizeof(T), SEEK_CUR);
        }
        fseek(fp, (strides_total.z - sizes.y*strides_total.y) * sizeof(T), SEEK_CUR);
    }
    fclose(fp);
}

template <typename T>
void store_file_strided(const std::vector<T> &data, const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range, const idx3d &offset_global) {
    store_file_strided(data.data(), path, disk_shape, shape, range, offset_global);
}

template <typename T>
void store_partial(const T *__restrict__ src, FILE *fp, const int64_t offset, const int64_t n_elements) {
    fseek(fp, offset*sizeof(T), SEEK_SET);
    int64_t n = fwrite((char *) src, sizeof(T), n_elements, fp);
    //assert(n == n_elements && "Failed to write all elements");
}

#endif // HIPPO_HPP