#ifndef HISTOGRAMS_HPP
#define HISTOGRAMS_HPP

#include "hippo.hpp"

// Functions
void axis_histogram_par_gpu(const np_array<voxel_type> np_voxels,
                            const tuple<uint64_t,uint64_t,uint64_t> offset,
                            const uint64_t outside_block_size,
                            np_array<uint64_t> &np_x_bins,
                            np_array<uint64_t> &np_y_bins,
                            np_array<uint64_t> &np_z_bins,
                            np_array<uint64_t> &np_r_bins,
                            const tuple<uint64_t, uint64_t> center,
                            const tuple<double, double> vrange,
                            const bool verbose);

#endif // HISTOGRAMS_HPP