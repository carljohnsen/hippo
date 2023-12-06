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
constexpr int64_t GLOBAL_FLAT_SIZE = (Nz_global+2*R) * (Ny_global+2*R) * (Nx_global+2*R);
constexpr int64_t LOCAL_FLAT_SIZE = (Nz_local+2*R) * (Ny_local+2*R) * (Nx_local+2*R);

// Number of devices to use
constexpr int64_t N_DEVICES = 1;
// Number of streams to use per device
constexpr int64_t N_STREAMS = 1;

void convert_float_to_uint8(std::string &src, std::string &dst);
void convert_uint8_to_float(std::string &src, std::string &dst);
void diffusion(std::string &input_file, std::vector<float>& kernel, std::string &output_file);
void diffusion_core(float *input, float *kernel, float *output, int64_t dim);
void illuminate(bool *mask, float *output);
void read_block(float *dst, std::string &path, idx3drange &range);
void store_mask(float *input, bool *mask);
void stage_to_device(float *stage, float *src, idx3drange &range);
void stage_to_host(float *dst, float *stage, idx3drange &range);
void write_block(float *src, std::string &path, idx3drange &range);

#endif // DIFFUSION_HPP