#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

#include "hippo.hpp"

void convert_float_to_uint8(const std::string &src, const std::string &dst, const int64_t total_flat_size);
void convert_uint8_to_float(const std::string &src, const std::string &dst, const int64_t total_flat_size);
void diffusion(const std::string &input_file, const std::vector<float>& kernel, const std::string &output_file, const idx3d &total_shape, const idx3d &global_shape, const int64_t repititions);
void diffusion_core(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const int64_t dim, const int64_t kernel_size);
void illuminate(const bool *__restrict__ mask, float *__restrict__ output, const int64_t local_flat_size);
void store_mask(const float *__restrict__ input, bool *__restrict__ mask, const int64_t local_flat_size);
void stage_to_device(float *__restrict__ stage, const float *__restrict__ src, const idx3drange &range, const idx3d &global_shape, const int64_t kernel_size);
void stage_to_host(float *__restrict__ dst, const float *__restrict__ stage, const idx3drange &range, const idx3d &global_shape, const int64_t kernel_size);

#endif // DIFFUSION_HPP