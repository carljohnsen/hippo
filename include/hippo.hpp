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
    Nz_total = 128,
    Ny_total = 128,
    Nx_total = 128,
    // Out-of-core main memory parameters
    Nz_global = 64,
    Ny_global = 64,
    Nx_global = 64,
    // Out-of-core GPU memory parameters
    Nz_local = 32,
    Ny_local = 32,
    Nx_local = 32,
    // Input image generation parameters
    C = 4;

// General functions

// Loads `n_elements` of a file located at `path` on disk at `offset` elements from the beginning of the file, into a vector of type `T`.
template <typename T>
std::vector<T> load_file(const std::string &path, const int64_t offset, const int64_t n_elements);

template <typename T>
void load_file(T *dst, const std::string &path, const int64_t offset, const int64_t n_elements);

// Loads the specified index `range` of a file located at `path` on disk which is of the given `shape`, into a vector of type `T`.
template <typename T>
std::vector<T> load_file_strided(const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range, const idx3d &offset_global);

// Reads the specified index `range` of a file located at `path` on disk which is of the given `shape`, into `dst`.
// `disk_shape` is the shape of the file on disk, and `shape` is the shape of the allocated memory.
// This version exists to avoid allocating a vector for each call, and reads directly from disk.
// The last stride is always assumed to be 1, for both src and dst.
// It is up to the caller to ensure that 1) `range` doesn't exceed `shape`, 2) `dst` is large enough to hold the data, 3) `dst` is set to 0 in case of a partial read and 0s are desired and 4) `dst` is an aligned allocation (e.g. using `aligned_alloc()`) to maximize performance.
template <typename T>
void load_file_strided(T *dst, const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range, const idx3d &offset_global);

// Returns the mean of the given data
float mean(const std::vector<float> &data);

// Returns the standard deviation of the given data
float stddev(const std::vector<float> &data);

// Stores `data.size()` elements of `data` into a file located at `path` on disk at `offset` elements from the beginning of the file.
template <typename T>
void store_file(const std::vector<T> &data, const std::string &path, const int64_t offset);

template <typename T>
void store_file(const T *data, const std::string &path, const int64_t offset, const int64_t n_elements);

template <typename T>
void store_file_strided(const std::vector<T> &data, const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range, const idx3d &offset_global);

template <typename T>
void store_file_strided(const T *data, const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range, const idx3d &offset_global);

// Writes the given 2D floating point array (between 0.0 and 1.0) to a PGM file
void write_pgm(const std::string &filename, const std::vector<float> &data, const int64_t width, const int64_t height);

#endif // HIPPO_HPP