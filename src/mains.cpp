#include "diffusion.hpp"

// This file contains main functions for testing the individual parts of the library, without having to compile to Python.

void verify_diffusion(const idx3d &total_shape, const idx3d &global_shape) {
    constexpr float SIGMA = 5.0f;
    constexpr int64_t R = (int64_t) (4.0f * SIGMA);

    assert(R < total_shape.z && R < total_shape.y && R < total_shape.x && "R must be smaller than the total block size");
    assert(R < global_shape.z && R < global_shape.y && R < global_shape.x && "R must be smaller than the global block size");
    assert(R < local_shape.z && R < local_shape.y && R < local_shape.x && "R must be smaller than the local block size");

    // Define the vectors
    std::vector<float> kernel(R*2+1);
    std::string
        input_filename = "data/input_img.uint8",
        output_filename = "data/output_img.uint8";

    // Set the kernel to a normalized 1D Gaussian
    for (int64_t i = 0; i < R*2+1; i++) {
        float x = (float)(i-R);
        kernel[i] = 1.0f / (SIGMA * std::sqrt(2.0 * M_PI)) * std::exp(-(x*x) / (2.0 * SIGMA * SIGMA));
    }
    write_pgm("output/kernel.pgm", kernel, R*2+1, 1);

    diffusion(input_filename, kernel, output_filename, total_shape, global_shape, 10);
}

int main() {
    constexpr idx3d
        total_shape = { 1024, 1024, 1024 },
        global_shape = { 256, total_shape.y, total_shape.x };

    verify_diffusion(total_shape, global_shape);
    return 0;
}