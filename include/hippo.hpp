#ifndef hippo_hpp
#define hippo_hpp

// Includes
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <list>
#include <tuple>
#include <unordered_set>
#include <vector>

// Custom datatypes
struct dim3 {
    int64_t z, y, x;
};
typedef std::vector<std::unordered_set<int64_t>> mapping;

// Constants
// Out-of-core main memory parameters
constexpr int64_t
    Nz_global = 512,
    Ny_global = 512,
    Nx_global = 512,
// Out-of-core GPU memory parameters
    Nz_local = 128,
    Ny_local = 128,
    Nx_local = 128;

// Common functions

// Loads `n_elements` of a file located at `path` on disk at `offset` elements from the beginning of the file, into a vector of type `T`.
template <typename T>
std::vector<T> load_file(std::string &path, int64_t offset, int64_t n_elements);

// Stores `data.size()` elements of `data` into a file located at `path` on disk at `offset` elements from the beginning of the file.
template <typename T>
void store_file(std::string &path, std::vector<T> &data, int64_t offset);

#endif /* hippo_hpp */