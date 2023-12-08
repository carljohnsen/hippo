#include "diffusion.hpp"

// This file contains main functions for testing the individual parts of the library, without having to compile to Python.

void verify_diffusion() {
    assert(R < Nx_total && R < Ny_total && R < Nz_total && "R must be smaller than the total block size");
    assert(R < Nx_global && R < Ny_global && R < Nz_global && "R must be smaller than the global block size");
    assert(R < Nx_local && R < Ny_local && R < Nz_local && "R must be smaller than the local block size");

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

    diffusion(input_filename, kernel, output_filename);
}

int main() {
    verify_diffusion();
    return 0;
}