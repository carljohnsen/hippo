#include "diffusion.hpp"

void diffusion_core(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const int64_t dim) {
    #pragma omp target teams distribute parallel for collapse(3)
    for (int64_t i = 0; i < Nz_local+2*R; i++) {
        for (int64_t j = 0; j < Ny_local+2*R; j++) {
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

                //#pragma omp simd reduction(+:sum)
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

void illuminate(const bool *__restrict__ mask, float *__restrict__ output) {
    #pragma omp target teams distribute parallel for
    for (int64_t i = 0; i < LOCAL_FLAT_SIZE; i++) {
        if (mask[i]) {
            output[i] = 1.0f;
        }
    }
}

void store_mask(const float *__restrict__ input, bool *__restrict__ mask) {
    #pragma omp target teams distribute parallel for
    for (int64_t i = 0; i < LOCAL_FLAT_SIZE; i++) {
        mask[i] = input[i] == 1.0f;
    }
}

void stage_to_device(float *__restrict__ stage, const float *__restrict__ src, const idx3drange &range) {
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

    memset(stage, 0, disk_local_flat_size);

    // Fill the staging area
    #pragma omp parallel for schedule(static) collapse(3)
    for (int64_t z = 0; z < size_z; z++) {
        for (int64_t y = 0; y < size_y; y++) {
            for (int64_t x = 0; x < size_x; x++) {
                int64_t dst_idx = (z+offset_z)*local_strides[0] + (y+offset_y)*local_strides[1] + (x+offset_x)*local_strides[2];
                int64_t src_idx = (z+start_z)*global_strides[0] + (y+start_y)*global_strides[1] + (x+start_x)*global_strides[2];
                stage[dst_idx] = src[src_idx];
            }
        }
    }
}

void stage_to_host(float *__restrict__ dst, const float *__restrict__ stage, const idx3drange &range) {
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

    #pragma omp parallel for schedule(static) collapse(3)
    for (int64_t z = 0; z < size_z; z++) {
        for (int64_t y = 0; y < size_y; y++) {
            for (int64_t x = 0; x < size_x; x++) {
                int64_t dst_idx = (z+start_z)*global_strides[0] + (y+start_y)*global_strides[1] + (x+start_x)*global_strides[2];
                int64_t src_idx = (z+offset_z)*local_strides[0] + (y+offset_y)*local_strides[1] + (x+offset_x)*local_strides[2];
                dst[dst_idx] = stage[src_idx];
            }
        }
    }
}

void convert_float_to_uint8(const std::string &src, const std::string &dst) {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    FILE *file_src = open_file_read<float>(src);
    FILE *file_dst = open_file_write<uint8_t>(dst);
    float *buffer_src = (float *) aligned_alloc(disk_block_size, 1024*disk_block_size);
    uint8_t *buffer_dst = (uint8_t *) aligned_alloc(disk_block_size, 1024*disk_block_size);

    for (int64_t chunk = 0; chunk < TOTAL_FLAT_SIZE; chunk += disk_block_size/sizeof(float)) {
        int64_t size = std::min((uint64_t)(disk_block_size/sizeof(float)), (uint64_t) (TOTAL_FLAT_SIZE - chunk));
        load_partial(buffer_src, file_src, chunk, size);
        //#pragma omp parallel for schedule(static) num_threads(2)
        for (int64_t i = 0; i < size; i++) {
            buffer_dst[i] = (uint8_t) (buffer_src[i] * 255.0f); // Convert to grayscale.
        }
        store_partial(buffer_dst, file_dst, chunk, size);
    }

    free(buffer_dst);
    free(buffer_src);
    fclose(file_dst);
    fclose(file_src);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Converting float to uint8 took " << duration.count() << " seconds at " << (TOTAL_FLAT_SIZE*sizeof(float))/duration.count()/1e9 << " GB/s" << std::endl;
}

void convert_uint8_to_float(const std::string &src, const std::string &dst) {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    FILE *file_src = open_file_read<uint8_t>(src);
    FILE *file_dst = open_file_write<float>(dst);
    uint8_t *buffer_src = (uint8_t *) aligned_alloc(disk_block_size, 1024*disk_block_size);
    float *buffer_dst = (float *) aligned_alloc(disk_block_size, 1024*disk_block_size);

    for (int64_t chunk = 0; chunk < TOTAL_FLAT_SIZE; chunk += disk_block_size/sizeof(float)) {
        int64_t size = std::min((uint64_t)(disk_block_size/sizeof(float)), (uint64_t) (TOTAL_FLAT_SIZE - chunk));
        load_partial(buffer_src, file_src, chunk, size);
        //#pragma omp parallel for schedule(static) num_threads(2)
        for (int64_t i = 0; i < size; i++) {
            buffer_dst[i] = buffer_src[i] > 0 ? 1.0f : 0.0f; // Loading a mask.
        }
        store_partial(buffer_dst, file_dst, chunk, size);
    }

    free(buffer_dst);
    free(buffer_src);
    fclose(file_dst);
    fclose(file_src);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Converting uint8 to float took " << duration.count() << " seconds at " << (TOTAL_FLAT_SIZE*sizeof(float))/duration.count()/1e9 << " GB/s" << std::endl;
}

void diffusion(const std::string &input_file, const std::vector<float>& kernel, const std::string &output_file) {
    auto start = std::chrono::high_resolution_clock::now();

    std::string
        temp0 = "data/temp0.float32",
        temp1 = "data/temp1.float32";

    // Compute the number of global blocks
    const int64_t
        global_blocks_z = std::ceil(Nz_total / (float)Nz_global),
        global_blocks_y = std::ceil(Ny_total / (float)Ny_global),
        global_blocks_x = std::ceil(Nx_total / (float)Nx_global),
        local_blocks_z = std::ceil((Nz_global+2*R) / (float)Nz_local),
        local_blocks_y = std::ceil((Ny_global+2*R) / (float)Ny_local),
        local_blocks_x = std::ceil((Nx_global+2*R) / (float)Nx_local);

    const idx3d
        total_shape = {Nz_total, Ny_total, Nx_total},
        global_shape = {Nz_global+2*R, Ny_global+2*R, Nx_global+2*R};

    // Print the number of blocks
    std::cout << "Global blocks: " << global_blocks_z << "x" << global_blocks_y << "x" << global_blocks_x << std::endl;
    std::cout << "Local blocks: " << local_blocks_z << "x" << local_blocks_y << "x" << local_blocks_x << std::endl;

    // Allocate memory. Aligned to block_size, and should overallocate to ensure alignment.
    float
        *input = (float *) aligned_alloc(disk_block_size, disk_global_flat_size),
        *output = (float *) aligned_alloc(disk_block_size, disk_global_flat_size),
        *d_input = (float *) aligned_alloc(disk_block_size, disk_local_flat_size),
        *d_output = (float *) aligned_alloc(disk_block_size, disk_local_flat_size),
        *d_kernel = (float *) aligned_alloc(disk_block_size, disk_kernel_flat_size);
    bool *d_mask = (bool *) aligned_alloc(disk_block_size, disk_mask_flat_size);

    memcpy(d_kernel, &kernel[0], (R*2+1)*sizeof(float));
    convert_uint8_to_float(input_file, temp0);

    //omp_set_num_threads(N_DEVICES*N_STREAMS);

    #pragma omp target enter data map(to: d_kernel[0:R*2+1])
    for (int64_t reps = 0; reps < REPITITIONS; reps++) {
        std::string
            iter_input  = reps % 2 == 0 ? temp0 : temp1,
            iter_output = reps % 2 == 0 ? temp1 : temp0;
        for (int64_t global_block_z = 0; global_block_z < global_blocks_z; global_block_z++) {
            for (int64_t global_block_y = 0; global_block_y < global_blocks_y; global_block_y++) {
                for (int64_t global_block_x = 0; global_block_x < global_blocks_x; global_block_x++) {
                    const idx3drange global_range = {
                        global_block_z*Nz_global, (global_block_z+1)*Nz_global,
                        global_block_y*Ny_global, (global_block_y+1)*Ny_global,
                        global_block_x*Nx_global, (global_block_x+1)*Nx_global
                    };

                    // Read the block
                    const idx3drange global_range_in = {
                        std::max(global_range.z_start-R, (int64_t) 0), std::min(global_range.z_end+R, Nz_total),
                        std::max(global_range.y_start-R, (int64_t) 0), std::min(global_range.y_end+R, Ny_total),
                        std::max(global_range.x_start-R, (int64_t) 0), std::min(global_range.x_end+R, Nx_total)
                    };
                    const idx3d global_offset_in = {
                        global_range_in.z_start == 0 ? R : 0,
                        global_range_in.y_start == 0 ? R : 0,
                        global_range_in.x_start == 0 ? R : 0
                    };

                    // Ensure padding
                    memset(input, 0, disk_global_flat_size);

                    load_file_strided(input, iter_input, total_shape, global_shape, global_range_in, global_offset_in);

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

                                #pragma omp target data map(to: d_input[0:LOCAL_FLAT_SIZE]) map(alloc: d_mask[0:LOCAL_FLAT_SIZE]) map(from: d_output[0:LOCAL_FLAT_SIZE])
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

                    const idx3d global_offset_out = { R, R, R };

                    store_file_strided(output, iter_output, total_shape, global_shape, global_range, global_offset_out);
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