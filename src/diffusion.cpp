#include "diffusion.hpp"

void diffusion_core(float *input, float *kernel, float *output, int64_t dim) {
    // Måske er det de her ranges (som er nice i open acc), der goer at det ikke koerer hurtigt? Altsaa at cuda launcher flere blokke end der er data?
    // Ellers: 3d kernels er hurtigere end fladt paa dpcpp, saa maaske er det ogsaa tilfaeldet her?

    //#pragma acc parallel loop collapse(3) present(input[0:LOCAL_FLAT_SIZE], output[0:LOCAL_FLAT_SIZE], kernel[0:R*2+1]) //vector_length(32)

    //#pragma acc parallel present(input[0:LOCAL_FLAT_SIZE], output[0:LOCAL_FLAT_SIZE], kernel[0:R*2+1])
    //#pragma omp target data map(to: input[0:LOCAL_FLAT_SIZE], kernel[0:R*2+1]) map(from: output[0:LOCAL_FLAT_SIZE])
    //#pragma acc kernels present(input[0:LOCAL_FLAT_SIZE], output[0:LOCAL_FLAT_SIZE], kernel[0:R*2+1])
    {
    //#pragma omp target teams distribute parallel for
    //#pragma acc loop independent
    for (int64_t i = 0; i < Nz_local+2*R; i++) {
        //#pragma acc loop independent
        for (int64_t j = 0; j < Ny_local+2*R; j++) {
            //#pragma acc loop independent
            for (int64_t k = 0; k < Nx_local+2*R; k++) {
                const int64_t
                    X[3] = {i, j, k},
                    stride[3] = {(Ny_local+2*R)*(Nx_local+2*R), Nx_local+2*R, 1},
                    Ns[3] = {Nz_local+2*R, Ny_local+2*R, Nx_local+2*R},
                    ranges[2] = {
                        -std::min(R, X[dim]), std::min(R, Ns[dim]-X[dim]-1)
                    },
                    output_index = i*stride[0] + j*stride[1] + k*stride[2];

                float sum = 0.0f;
                //#pragma acc loop reduction(+:sum)
                //#pragma acc loop seq
                //#pragma omp parallel for simd reduction(+:sum)
                for (int64_t r = -R; r <= R; r++) {
                    const int64_t input_index = output_index + r*stride[dim];
                    float val = r >= ranges[0] && r <= ranges[1] ? input[input_index] : 0.0f;
                    sum += val * kernel[R+r];
                }

                output[output_index] = sum;
            }
        }
    }
    }
}

void illuminate(bool *mask, float *output) {
    //#pragma acc parallel loop present(mask[0:LOCAL_FLAT_SIZE], output[0:LOCAL_FLAT_SIZE])
    //#pragma omp target teams distribute parallel for map(tofrom: mask[0:LOCAL_FLAT_SIZE], output[0:LOCAL_FLAT_SIZE])
    for (int64_t i = 0; i < LOCAL_FLAT_SIZE; i++) {
        if (mask[i]) {
            output[i] = 1.0f;
        }
    }
}

void store_mask(float *input, bool *mask) {
    //#pragma acc parallel loop present(input[0:LOCAL_FLAT_SIZE], mask[0:LOCAL_FLAT_SIZE])
    //#pragma omp target teams distribute parallel for map(tofrom: input[0:LOCAL_FLAT_SIZE], mask[0:LOCAL_FLAT_SIZE])
    for (int64_t i = 0; i < LOCAL_FLAT_SIZE; i++) {
        mask[i] = input[i] == 1.0f;
    }
}

void stage_to_device(float *stage, float *src, idx3drange &range) {
    auto [start_z, end_z, start_y, end_y, start_x, end_x] = range;
    start_z = std::max(start_z-R, (int64_t) 0);
    start_y = std::max(start_y-R, (int64_t) 0);
    start_x = std::max(start_x-R, (int64_t) 0);
    end_z   = std::min(end_z+R, Nz_global+2*R);
    end_y   = std::min(end_y+R, Ny_global+2*R);
    end_x   = std::min(end_x+R, Nx_global+2*R);
    const int64_t
        size_z = end_z - start_z,
        size_y = end_y - start_y,
        size_x = end_x - start_x,
        offset_z = start_z == 0 ? R : 0,
        offset_y = start_y == 0 ? R : 0,
        offset_x = start_x == 0 ? R : 0,
        global_strides[3] = {(Ny_global+2*R)*(Nx_global+2*R), (Nx_global+2*R), 1},
        local_strides[3] = {(Ny_local+2*R)*(Nx_local+2*R), Nx_local+2*R, 1};

    memset(stage, 0, LOCAL_FLAT_SIZE*sizeof(float));

    // Fill the staging area
    for (int64_t z = 0; z < size_z; z++) {
        for (int64_t y = 0; y < size_y; y++) {
            for (int64_t x = 0; x < size_x; x++) {
                stage[(offset_z+z)*local_strides[0] + (offset_y+y)*local_strides[1] + (offset_x+x)*local_strides[2]] =
                    src[(start_z+z)*global_strides[0] + (start_y+y)*global_strides[1] + (start_x+x)*global_strides[2]];
            }
        }
    }
}

void stage_to_host(float *dst, float *stage, idx3drange &range) {
    auto [start_z, end_z, start_y, end_y, start_x, end_x] = range;
    end_z   = std::min(end_z, Nz_global+2*R);
    end_y   = std::min(end_y, Ny_global+2*R);
    end_x   = std::min(end_x, Nx_global+2*R);

    const int64_t
        offset_z = R,
        offset_y = R,
        offset_x = R,
        size_z = end_z - start_z,
        size_y = end_y - start_y,
        size_x = end_x - start_x,
        global_strides[3] = {(Ny_global+2*R)*(Nx_global+2*R), (Nx_global+2*R), 1},
        local_strides[3] = {(Ny_local+2*R)*(Nx_local+2*R), Nx_local+2*R, 1};

    for (int64_t z = 0; z < size_z; z++) {
        for (int64_t y = 0; y < size_y; y++) {
            for (int64_t x = 0; x < size_x; x++) {
                dst[(start_z+z)*global_strides[0] + (start_y+y)*global_strides[1] + (start_x+x)*global_strides[2]] =
                    stage[(z+offset_z)*local_strides[0] + (y+offset_y)*local_strides[1] + (x+offset_x)*local_strides[2]];
            }
        }
    }
}

void convert_float_to_uint8(std::string &src, std::string &dst) {
    const int64_t chunk_size = 4096;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Write the output to the uint8 file
    std::ifstream inf(src, std::ios::binary);
    std::ios::openmode mode = std::ios::binary;
    // If file exists, append in mode
    if (std::filesystem::exists(dst)) {
        mode |= std::ios::in;
    }
    std::ofstream outf(dst, mode);
    std::vector<float> buffer0(chunk_size);
    std::vector<uint8_t> buffer1(chunk_size);
    for (int64_t chunk = 0; chunk < TOTAL_FLAT_SIZE; chunk += chunk_size) {
        int64_t size = std::min((int64_t)chunk_size, TOTAL_FLAT_SIZE - chunk);
        inf.read((char *) &buffer0[0], size*sizeof(float));
        for (int64_t i = 0; i < size; i++) {
            buffer1[i] = (uint8_t) (buffer0[i] * 255.0f); // Convert to grayscale.
        }
        outf.write((char *) &buffer1[0], size);
    }

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Converting float to uint8 took " << duration.count() << " seconds at " << (TOTAL_FLAT_SIZE*sizeof(float))/duration.count()/1e9 << " GB/s" << std::endl;
}

void convert_uint8_to_float(std::string &src, std::string &dst) {
    const int64_t chunk_size = 1024;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Construct temp0 from the uint8 file
    std::vector<float> buffer1(chunk_size);
    for (int64_t chunk = 0; chunk < TOTAL_FLAT_SIZE; chunk += chunk_size) {
        std::vector<uint8_t> buffer0 = load_file<uint8_t>(src, chunk, chunk_size);
        for (int64_t i = 0; i < (int64_t) buffer0.size(); i++) {
            buffer1[i] = buffer0[i] > 0 ? 1.0f : 0.0f; // Loading a mask.
        }
        store_file(buffer1, dst, chunk);
    }

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Converting uint8 to float took " << duration.count() << " seconds at " << (TOTAL_FLAT_SIZE*sizeof(float))/duration.count()/1e9 << " GB/s" << std::endl;
}

void diffusion(std::string &input_file, std::vector<float>& kernel, std::string &output_file) {
    auto start = std::chrono::high_resolution_clock::now();

    std::string
        temp0 = "data/temp0.float32",
        temp1 = "data/temp1.float32";

    // Compute the number of global blocks
    const int64_t
        disk_block_size = 4096,
        global_blocks_z = std::ceil(Nz_total / (float)Nz_global),
        global_blocks_y = std::ceil(Ny_total / (float)Ny_global),
        global_blocks_x = std::ceil(Nx_total / (float)Nx_global),
        local_blocks_z = std::ceil((Nz_global+2*R) / (float)Nz_local),
        local_blocks_y = std::ceil((Ny_global+2*R) / (float)Ny_local),
        local_blocks_x = std::ceil((Nx_global+2*R) / (float)Nx_local);

    const idx3d
        total_shape = {Nz_total, Ny_total, Nx_total},
        global_shape = {Nz_global+2*R, Ny_global+2*R, Nx_global+2*R};

    std::cout << "Local flat size allocation is " << LOCAL_FLAT_SIZE*sizeof(float) << " bytes" << std::endl;

    // Print the number of blocks
    std::cout << "Global blocks: " << global_blocks_z << "x" << global_blocks_y << "x" << global_blocks_x << std::endl;
    std::cout << "Local blocks: " << local_blocks_z << "x" << local_blocks_y << "x" << local_blocks_x << std::endl;

    // Allocate memory
    float
        *input = (float *) aligned_alloc(disk_block_size, GLOBAL_FLAT_SIZE*sizeof(float)),
        *output = (float *) aligned_alloc(disk_block_size, GLOBAL_FLAT_SIZE*sizeof(float)),
        *d_input = (float *) aligned_alloc(disk_block_size, LOCAL_FLAT_SIZE*sizeof(float)),
        *d_output = (float *) aligned_alloc(disk_block_size, LOCAL_FLAT_SIZE*sizeof(float)),
        *d_kernel = (float *) aligned_alloc(disk_block_size, (R*2+1)*sizeof(float));
    bool *d_mask = (bool *) aligned_alloc(disk_block_size, LOCAL_FLAT_SIZE*sizeof(int));

    memcpy(d_kernel, &kernel[0], (R*2+1)*sizeof(float));
    convert_uint8_to_float(input_file, temp0);

    //omp_set_num_threads(N_DEVICES*N_STREAMS);

    //#pragma acc enter data copyin(d_kernel[0:R*2+1])
    //#pragma omp target enter data map(to: d_kernel[0:R*2+1])
    for (int64_t reps = 0; reps < REPITITIONS; reps++) {
        std::string
            iter_input  = reps % 2 == 0 ? temp0 : temp1,
            iter_output = reps % 2 == 0 ? temp1 : temp0;
        for (int64_t global_block_z = 0; global_block_z < global_blocks_z; global_block_z++) {
            for (int64_t global_block_y = 0; global_block_y < global_blocks_y; global_block_y++) {
                for (int64_t global_block_x = 0; global_block_x < global_blocks_x; global_block_x++) {
                    // Read the block
                    const idx3drange global_range = {
                        std::max((global_block_z*Nz_global)-R, (int64_t) 0), std::min((global_block_z+1)*Nz_global+R, Nz_total),
                        std::max((global_block_y*Ny_global)-R, (int64_t) 0), std::min((global_block_y+1)*Ny_global+R, Ny_total),
                        std::max((global_block_x*Nx_global)-R, (int64_t) 0), std::min((global_block_x+1)*Nx_global+R, Nx_total)
                    };
                    const idx3d global_offset = {
                        global_range.z_start == 0 ? R : 0,
                        global_range.y_start == 0 ? R : 0,
                        global_range.x_start == 0 ? R : 0
                    };

                    // Ensure padding
                    memset(input, 0, GLOBAL_FLAT_SIZE*sizeof(float));

                    load_file_strided(input, iter_input, total_shape, global_shape, global_range, global_offset);

                    //#pragma omp parallel for schedule(static) collapse(3)
                    for (int64_t local_block_z = 0; local_block_z < local_blocks_z; local_block_z++) {
                        for (int64_t local_block_y = 0; local_block_y < local_blocks_y; local_block_y++) {
                            for (int64_t local_block_x = 0; local_block_x < local_blocks_x; local_block_x++) {
                                idx3drange local_range = {
                                    local_block_z*Nz_local, (local_block_z+1)*Nz_local,
                                    local_block_y*Ny_local, (local_block_y+1)*Ny_local,
                                    local_block_x*Nx_local, (local_block_x+1)*Nx_local
                                };

                                // Copy data to device
                                stage_to_device(d_input, input, local_range);

                                //#pragma acc data copyin(d_input[0:LOCAL_FLAT_SIZE]) create(d_mask[0:LOCAL_FLAT_SIZE], d_output[0:LOCAL_FLAT_SIZE]) copyout(d_output[0:LOCAL_FLAT_SIZE])
                                //#pragma omp target data map(to: d_input[0:LOCAL_FLAT_SIZE]) map(alloc: d_mask[0:LOCAL_FLAT_SIZE]) map(from: d_output[0:LOCAL_FLAT_SIZE])
                                {
                                    store_mask(d_input, d_mask);
                                    diffusion_core(d_input,  d_kernel, d_output, 0);
                                    diffusion_core(d_output, d_kernel, d_input,  1);
                                    diffusion_core(d_input,  d_kernel, d_output, 2);
                                    illuminate(d_mask, d_output);
                                }

                                // Copy data back to host
                                stage_to_host(output, d_output, local_range);
                            }
                        }
                    }

                    store_file_strided(output, iter_output, total_shape, global_shape, global_range, global_offset);
                }
            }
        }
    }

    convert_float_to_uint8(REPITITIONS % 2 == 0 ? temp0 : temp1, output_file);

    free(d_mask);
    free(d_kernel);
    free(d_output);
    free(d_input);
    free(output);
    free(input);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end-start;

    std::cout << "Diffusion took " << duration.count() << " seconds at " << (TOTAL_FLAT_SIZE*sizeof(float)*REPITITIONS)/duration.count()/1e9 << " GB/s" << std::endl;
}