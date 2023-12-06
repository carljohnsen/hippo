#ifndef HIPPO_HPP
#define HIPPO_HPP

// Includes
//#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <omp.h>
#include <tuple>
#include <unordered_set>
#include <vector>

// Custom datatypes
struct idx3d {
    int64_t z, y, x;
};
struct idx3drange {
    int64_t z_start, z_end, y_start, y_end, x_start, x_end;
};
typedef std::vector<std::unordered_set<int64_t>> mapping;

// Constants
constexpr int64_t
    // On disk parameters for generated input
    Nz_total = 64,
    Ny_total = 64,
    Nx_total = 64,
    // Out-of-core main memory parameters
    Nz_global = 32,
    Ny_global = 32,
    Nx_global = 32,
    // Out-of-core GPU memory parameters
    Nz_local = 16,
    Ny_local = 16,
    Nx_local = 16,
    // Input image generation parameters
    C = 4;

// Common functions

// Loads `n_elements` of a file located at `path` on disk at `offset` elements from the beginning of the file, into a vector of type `T`.
template <typename T>
std::vector<T> load_file(std::string &path, int64_t offset, int64_t n_elements);

// Stores `data.size()` elements of `data` into a file located at `path` on disk at `offset` elements from the beginning of the file.
template <typename T>
void store_file(std::string &path, std::vector<T> &data, int64_t offset);

#endif // HIPPO_HPP