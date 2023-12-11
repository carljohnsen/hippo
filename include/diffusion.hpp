#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

#include "hippo.hpp"

// Standard deviation of the Gaussian
constexpr float SIGMA = 5.0f;
// Radius of the kernel
// TODO has to be < than any of the Ns!
constexpr int64_t R = (int64_t) (4.0 * SIGMA);
// Number of times to repeat the diffusion
constexpr int64_t REPITITIONS = 1;

// The flat sizes of all the shapes
constexpr int64_t TOTAL_FLAT_SIZE = Nz_total * Ny_total * Nx_total;
constexpr int64_t GLOBAL_FLAT_SIZE = (Nz_global+2*R) * (Ny_global) * (Nx_global);
constexpr int64_t LOCAL_FLAT_SIZE = (Nz_local+2*R) * (Ny_local+2*R) * (Nx_local+2*R);
constexpr int64_t disk_global_flat_size = ((GLOBAL_FLAT_SIZE*sizeof(float) / disk_block_size) + (GLOBAL_FLAT_SIZE*sizeof(float) % disk_block_size == 0 ? 0 : 1)) * disk_block_size;
constexpr int64_t disk_local_flat_size = ((LOCAL_FLAT_SIZE*sizeof(float) / disk_block_size) + (LOCAL_FLAT_SIZE*sizeof(float) % disk_block_size == 0 ? 0 : 1)) * disk_block_size;
constexpr int64_t disk_mask_flat_size = ((LOCAL_FLAT_SIZE*sizeof(bool) / disk_block_size) + (LOCAL_FLAT_SIZE*sizeof(bool) % disk_block_size == 0 ? 0 : 1)) * disk_block_size;
constexpr int64_t disk_kernel_flat_size = (((R*2+1) / disk_block_size) + ((R*2+1) % disk_block_size == 0 ? 0 : 1)) * disk_block_size;


// Number of devices to use
constexpr int64_t N_DEVICES = 1;
// Number of streams to use per device
constexpr int64_t N_STREAMS = 1;

void convert_float_to_uint8(const std::string &src, const std::string &dst);
void convert_uint8_to_float(const std::string &src, const std::string &dst);
void diffusion(const std::string &input_file, const std::vector<float>& kernel, const std::string &output_file);
void diffusion_core(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const int64_t dim);
void illuminate(const bool *__restrict__ mask, float *__restrict__ output);
void store_mask(const float *__restrict__ input, bool *__restrict__ mask);
void stage_to_device(float *__restrict__ stage, const float *__restrict__ src, const idx3drange &range);
void stage_to_host(float *__restrict__ dst, const float *__restrict__ stage, const idx3drange &range);

#endif // DIFFUSION_HPP