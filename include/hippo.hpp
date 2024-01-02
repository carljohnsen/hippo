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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
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
template <typename T>
using np_array = pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;
typedef uint16_t voxel_type;

// Constants
constexpr idx3d local_shape = { 64, 64, 64 };

// Number of devices to use
constexpr int64_t n_devices = 1;

// Number of streams to use per device (TODO: must be 1 for now)
constexpr int64_t n_streams = 1;

constexpr int64_t
    // Input image generation parameters
    C = 4,
    // Filesystem block size in bytes. Needed for O_DIRECT.
    disk_block_size = 4096; // TODO get from filesystem.

// General function headers
// Returns the mean of the given data
float mean(const std::vector<float> &data);

// Returns the standard deviation of the given data
float stddev(const std::vector<float> &data);

// Writes the given 2D floating point array (between 0.0 and 1.0) to a PGM file
void write_pgm(const std::string &filename, const std::vector<float> &data, const int64_t width, const int64_t height);

// Opens a file for reading. The file is opened with O_DIRECT, which means that the file must be aligned to the disk block size. It is up to the caller to close the file properly.
FILE* open_file_read(const std::string &path);

// Opens a file for writing. The file is opened with O_DIRECT, which means that the file must be aligned to the disk block size. It is up to the caller to close the file properly.
FILE* open_file_write(const std::string &path);

// Templated functions

template <typename T>
void load_file_no_alloc(T *dst, FILE *fp, const int64_t offset, const int64_t n_elements) {
    fseek(fp, offset*sizeof(T), SEEK_SET);
    int64_t n = fread((char *) dst, sizeof(T), n_elements, fp);
    assert(n == n_elements && "Failed to read all elements");
}

// Loads `n_elements` of a file located at `path` on disk at `offset` elements from the beginning of the file, into a vector of type `T`.
template <typename T>
void load_file(T *dst, const std::string &path, const int64_t total_offset, const int64_t n_elements) {
    // Open the file
    FILE *fp = open_file_write(path);

    // Calculate the aligned start and end positions
    int64_t
        disk_block_size_elements = disk_block_size / sizeof(T),
        start_pos = total_offset*sizeof(T),
        end_pos = (total_offset+n_elements)*sizeof(T),
        aligned_start = (start_pos / disk_block_size) * disk_block_size,
        aligned_end = ((end_pos + disk_block_size - 1) / disk_block_size) * disk_block_size,
        aligned_size = aligned_end - aligned_start,
        aligned_n_elements = aligned_size / sizeof(T),
        aligned_offset = aligned_start / sizeof(T);

    if (start_pos % disk_block_size == 0 && end_pos % disk_block_size == 0 && n_elements % disk_block_size_elements == 0 && (int64_t) dst % disk_block_size == 0 && total_offset % disk_block_size_elements == 0) {
        load_file_no_alloc(dst, fp, total_offset, n_elements);
    } else {
        // Allocate a buffer for the write
        T *buffer = (T *) aligned_alloc(disk_block_size, aligned_size);

        // Read the buffer from disk
        load_file_no_alloc(buffer, fp, aligned_offset, aligned_n_elements);

        // Copy the data to the destination
        memcpy((char *) dst, (char *) buffer + start_pos - aligned_start, n_elements*sizeof(T));

        // Free the buffer and close the file
        free(buffer);
    }
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
    // Calculate the strides and sizes
    const idx3d
        strides_total = {shape_total.y*shape_total.x, shape_total.x, 1},
        strides_global = {shape_global.y*shape_global.x, shape_global.x, 1},
        sizes = {range.z_end - range.z_start, range.y_end - range.y_start, range.x_end - range.x_start};

    // If the shape on disk is the same as the shape in memory, just load the entire file
    if (shape_global.y == shape_total.y && shape_global.x == shape_total.x && offset_global.y == 0 && offset_global.x == 0 && range.y_start == 0 && range.x_start == 0 && range.y_end == shape_total.y && range.x_end == shape_total.x) {
        load_file(dst + (offset_global.z*strides_global.z), path, range.z_start*strides_total.z, sizes.z*strides_total.z);
        return;
    }
    assert (false && "Not implemented yet :) - After the deadline!");

    // Open the file
    FILE *fp = open_file_read(path);
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
    assert(n == n_elements && "Failed to read all elements");
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
    // Open the file
    FILE *fp = open_file_write(path);

    // Calculate the aligned start and end positions
    int64_t
        start_pos = offset*sizeof(T),
        end_pos = (offset+n_elements)*sizeof(T),
        aligned_start = (start_pos / disk_block_size) * disk_block_size,
        aligned_end = ((end_pos + disk_block_size - 1) / disk_block_size) * disk_block_size,
        aligned_size = aligned_end - aligned_start,
        aligned_n_elements = aligned_size / sizeof(T);

    // Allocate a buffer for the write
    T *buffer = (T *) aligned_alloc(disk_block_size, aligned_size);

    // If the start is not aligned, read the first block
    if (start_pos != aligned_start) {
        // Read the first block
        fseek(fp, aligned_start, SEEK_SET);
        int64_t n = fread((char *) buffer, sizeof(T), disk_block_size, fp);
        assert (n == disk_block_size && "Failed to read all elements");
    }

    // If the end is not aligned, read the last block
    if (end_pos != aligned_end) {
        // Read the last block
        fseek(fp, aligned_end - disk_block_size, SEEK_SET);
        int64_t n = fread((char *) buffer + aligned_size - disk_block_size, sizeof(T), disk_block_size, fp);
        assert (n == disk_block_size && "Failed to read all elements");
    }

    // Copy the data to the buffer
    memcpy((char *) buffer + start_pos - aligned_start, (char *) data, n_elements*sizeof(T));

    // Write the buffer to disk
    fseek(fp, aligned_start, SEEK_SET);
    int64_t n = fwrite((char *) buffer, sizeof(T), aligned_n_elements, fp);
    assert (n == aligned_n_elements && "Failed to write all elements");

    // Free the buffer and close the file
    free(buffer);
    fclose(fp);
}

template <typename T>
void store_file_strided(const T *data, const std::string &path, const idx3d &shape_total, const idx3d &shape_global, const idx3drange &range, const idx3d &offset_global) {
    // Calculate the strides and sizes
    const idx3d
        strides_total = {shape_total.y*shape_total.x, shape_total.x, 1},
        strides_global = {shape_global.y*shape_global.x, shape_global.x, 1},
        sizes = {range.z_end - range.z_start, range.y_end - range.y_start, range.x_end - range.x_start};

    // If the shape on disk is the same as the shape in memory, just store the entire file
    if (shape_global.y == shape_total.y && shape_global.x == shape_total.x && offset_global.y == 0 && offset_global.x == 0 && range.y_start == 0 && range.x_start == 0 && range.y_end == shape_total.y && range.x_end == shape_total.x) {
        store_file(data + offset_global.z*strides_global.z, path, range.z_start*strides_total.z, sizes.z*strides_total.z);
        return;
    }

    assert (false && "Not implemented yet :) - After the deadline!");

    // Open the file
    FILE *fp = open_file_write(path);
    fseek(fp, (range.z_start*strides_total.z + range.y_start*strides_total.y + range.x_start*strides_total.x)*sizeof(T), SEEK_SET);
    for (int64_t z = 0; z < sizes.z; z++) {
        for (int64_t y = 0; y < sizes.y; y++) {
            int64_t n = fwrite((char *) &data[(z+offset_global.z)*strides_global.z + (y+offset_global.y)*strides_global.y + (0+offset_global.x)*strides_global.x], sizeof(T), sizes.x, fp);
            assert (n == sizes.x && "Failed to write all elements");
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
    assert(n == n_elements && "Failed to write all elements");
}

#endif // HIPPO_HPP