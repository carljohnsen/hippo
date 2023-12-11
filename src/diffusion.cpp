#include "diffusion.hpp"

void diffusion_core(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const int64_t dim, const int64_t kernel_size) {
    const int64_t
        radius = kernel_size / 2,
        padding = kernel_size - 1;
    #pragma omp target teams distribute parallel for collapse(3)
    for (int64_t i = 0; i < local_shape.z+padding; i++) {
        for (int64_t j = 0; j < local_shape.y+padding; j++) {
            for (int64_t k = 0; k < local_shape.x+padding; k++) {
                const int64_t
                    X[3] = {i, j, k},
                    stride[3] = {(local_shape.y+padding)*(local_shape.x+padding), local_shape.x+padding, 1},
                    Ns[3] = {local_shape.z+padding, local_shape.y+padding, local_shape.x+padding},
                    ranges[2] = {
                        -std::min(radius, X[dim]), std::min(radius, Ns[dim]-X[dim]-1)
                    },
                    output_index = i*stride[0] + j*stride[1] + k*stride[2];

                float sum = 0.0f;

                //#pragma omp simd reduction(+:sum)
                for (int64_t r = -radius; r <= radius; r++) {
                    const int64_t input_index = output_index + r*stride[dim];
                    float val = r >= ranges[0] && r <= ranges[1] ? input[input_index] : 0.0f;
                    sum += val * kernel[radius+r];
                }

                output[output_index] = sum;
            }
        }
    }
}

void illuminate(const bool *__restrict__ mask, float *__restrict__ output, const int64_t local_flat_size) {
    #pragma omp target teams distribute parallel for
    for (int64_t i = 0; i < local_flat_size; i++) {
        if (mask[i]) {
            output[i] = 1.0f;
        }
    }
}

void store_mask(const float *__restrict__ input, bool *__restrict__ mask, const int64_t local_flat_size) {
    #pragma omp target teams distribute parallel for
    for (int64_t i = 0; i < local_flat_size; i++) {
        mask[i] = input[i] == 1.0f;
    }
}

void stage_to_device(float *__restrict__ stage, const float *__restrict__ src, const idx3drange &range, const idx3d &global_shape, const int64_t kernel_size) {
    const int64_t
        radius = kernel_size / 2,
        padding = kernel_size - 1,
        local_flat_size = (local_shape.z+padding) * local_shape.y * local_shape.x,
        disk_local_flat_size = ((local_flat_size*sizeof(float) / disk_block_size) + (local_flat_size*sizeof(float) % disk_block_size == 0 ? 0 : 1)) * disk_block_size;
    const idx3d
        start = {
            std::max(range.z_start-radius, (int64_t) 0),
            std::max(range.y_start-radius, (int64_t) 0),
            std::max(range.x_start-radius, (int64_t) 0)
        },
        end = {
            std::min(range.z_end+radius, global_shape.z+padding),
            std::min(range.y_end+radius, global_shape.y),
            std::min(range.x_end+radius, global_shape.x)
        },
        size = {
            end.z - start.z,
            end.y - start.y,
            end.x - start.x
        },
        offset = {
            start.z == 0 ? radius : 0,
            start.y == 0 ? radius : 0,
            start.x == 0 ? radius : 0
        },
        global_strides = { global_shape.y*global_shape.x, global_shape.x, 1 },
        local_strides =  { (local_shape.y+padding)*(local_shape.x+padding), local_shape.x+padding,   1 };

    memset(stage, 0, disk_local_flat_size);

    // Fill the staging area
    #pragma omp parallel for schedule(static) collapse(3)
    for (int64_t z = 0; z < size.z; z++) {
        for (int64_t y = 0; y < size.y; y++) {
            for (int64_t x = 0; x < size.x; x++) {
                int64_t dst_idx = (z+offset.z)*local_strides.z + (y+offset.y)*local_strides.y + (x+offset.x)*local_strides.x;
                int64_t src_idx = (z+start.z)*global_strides.z + (y+start.y)*global_strides.y + (x+start.x)*global_strides.x;
                stage[dst_idx] = src[src_idx];
            }
        }
    }
}

void stage_to_host(float *__restrict__ dst, const float *__restrict__ stage, const idx3drange &range, const idx3d &global_shape, const int64_t kernel_size) {
    const int64_t
        radius = kernel_size / 2,
        padding = kernel_size - 1;
    const idx3d
        start = { range.z_start, range.y_start, range.x_start },
        end = {
            std::min(range.z_end, global_shape.z+padding),
            std::min(range.y_end, global_shape.y),
            std::min(range.x_end, global_shape.x)
        },
        offset = { radius, radius, radius },
        size = {
            end.z - start.z,
            end.y - start.y,
            end.x - start.x
        },
        global_strides = { global_shape.y*global_shape.x, global_shape.x, 1 },
        local_strides =  { (local_shape.y+padding)*(local_shape.x+padding), local_shape.x+padding,   1 };

    #pragma omp parallel for schedule(static) collapse(3)
    for (int64_t z = 0; z < size.z; z++) {
        for (int64_t y = 0; y < size.y; y++) {
            for (int64_t x = 0; x < size.x; x++) {
                int64_t dst_idx = (z+start.z)*global_strides.z + (y+start.y)*global_strides.y + (x+start.x)*global_strides.x;
                int64_t src_idx = (z+offset.z)*local_strides.z + (y+offset.y)*local_strides.y + (x+offset.x)*local_strides.x;
                dst[dst_idx] = stage[src_idx];
            }
        }
    }
}

void convert_float_to_uint8(const std::string &src, const std::string &dst, const int64_t total_flat_size) {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    int64_t chunk_size = 2048*disk_block_size;
    FILE *file_src = open_file_read<float>(src);
    FILE *file_dst = open_file_write<uint8_t>(dst);
    float *buffer_src = (float *) aligned_alloc(disk_block_size, chunk_size*sizeof(float));
    uint8_t *buffer_dst = (uint8_t *) aligned_alloc(disk_block_size, chunk_size*sizeof(uint8_t));

    for (int64_t chunk = 0; chunk < total_flat_size; chunk += chunk_size) {
        int64_t size = std::min(chunk_size, total_flat_size - chunk);
        load_partial(buffer_src, file_src, chunk, size);
        #pragma omp parallel for schedule(static)
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
    std::cout << "Converting float to uint8 took " << duration.count() << " seconds at " << (total_flat_size*(sizeof(float) + sizeof(uint8_t)))/duration.count()/1e9 << " GB/s" << std::endl;
}

void convert_uint8_to_float(const std::string &src, const std::string &dst, const int64_t total_flat_size) {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    int64_t chunk_size = 2048*disk_block_size;
    FILE *file_src = open_file_read<uint8_t>(src);
    FILE *file_dst = open_file_write<float>(dst);
    uint8_t *buffer_src = (uint8_t *) aligned_alloc(disk_block_size, chunk_size*sizeof(uint8_t));
    float *buffer_dst = (float *) aligned_alloc(disk_block_size, chunk_size*sizeof(float));

    for (int64_t chunk = 0; chunk < total_flat_size; chunk += chunk_size) {
        int64_t size = std::min(chunk_size, total_flat_size - chunk);
        load_partial(buffer_src, file_src, chunk, size);
        #pragma omp parallel for schedule(static)
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
    std::cout << "Converting uint8 to float took " << duration.count() << " seconds at " << (total_flat_size*(sizeof(float) + sizeof(uint8_t)))/duration.count()/1e9 << " GB/s" << std::endl;
}

void diffusion(const std::string &input_file, const std::vector<float>& kernel, const std::string &output_file, const idx3d &total_shape, const idx3d &global_shape, const int64_t repititions) {
    auto start = std::chrono::high_resolution_clock::now();

    std::string
        temp0 = "data/temp0.float32",
        temp1 = "data/temp1.float32";

    // Compute the number of global blocks
    const int64_t
        kernel_size = kernel.size(),
        radius = kernel_size / 2,
        padding = kernel_size - 1, // kernel is assumed to be odd
        global_blocks = std::ceil(total_shape.z / (float)global_shape.z),
        local_blocks_z = std::ceil((global_shape.z+padding) / (float)local_shape.z),
        local_blocks_y = std::ceil((global_shape.y) / (float)local_shape.y),
        local_blocks_x = std::ceil((global_shape.x) / (float)local_shape.x),
        global_flat_size = (global_shape.z+padding) * global_shape.y * global_shape.x,
        local_flat_size = (local_shape.z+padding) * (local_shape.y+padding) * (local_shape.x+padding),
        total_flat_size = total_shape.z * total_shape.y * total_shape.x,
        disk_global_flat_size = ((global_flat_size*sizeof(float) / disk_block_size) + (global_flat_size*sizeof(float) % disk_block_size == 0 ? 0 : 1)) * disk_block_size,
        disk_local_flat_size = ((local_flat_size*sizeof(float) / disk_block_size) + (local_flat_size*sizeof(float) % disk_block_size == 0 ? 0 : 1)) * disk_block_size,
        disk_mask_flat_size = ((local_flat_size*sizeof(bool) / disk_block_size) + (local_flat_size*sizeof(bool) % disk_block_size == 0 ? 0 : 1)) * disk_block_size,
        disk_kernel_flat_size = ((kernel_size / disk_block_size) + (kernel_size % disk_block_size == 0 ? 0 : 1)) * disk_block_size;

    // Print the number of blocks
    std::cout << "Global blocks: " << global_blocks << std::endl;
    std::cout << "Local blocks: " << local_blocks_z << "x" << local_blocks_y << "x" << local_blocks_x << std::endl;

    // Allocate memory. Aligned to block_size, and should overallocate to ensure alignment.
    // TODO since the I/O functions handle alignment with buffers, the allocations doesn't need to be aligned.
    float
        *input = (float *) aligned_alloc(disk_block_size, disk_global_flat_size),
        *output = (float *) aligned_alloc(disk_block_size, disk_global_flat_size),
        *d_input = (float *) aligned_alloc(disk_block_size, disk_local_flat_size),
        *d_output = (float *) aligned_alloc(disk_block_size, disk_local_flat_size),
        *d_kernel = (float *) aligned_alloc(disk_block_size, disk_kernel_flat_size);
    bool *d_mask = (bool *) aligned_alloc(disk_block_size, disk_mask_flat_size);

    memcpy(d_kernel, &kernel[0], kernel.size()*sizeof(float));
    convert_uint8_to_float(input_file, temp0, total_flat_size);

    //omp_set_num_threads(N_DEVICES*N_STREAMS);

    #pragma omp target enter data map(to: d_kernel[0:kernel_size])
    for (int64_t reps = 0; reps < repititions; reps++) {
        std::string
            iter_input  = reps % 2 == 0 ? temp0 : temp1,
            iter_output = reps % 2 == 0 ? temp1 : temp0;
        for (int64_t global_block_i = 0; global_block_i < global_blocks; global_block_i++) {
            const idx3drange global_range = {
                global_block_i*global_shape.z, std::min((global_block_i+1)*global_shape.z, total_shape.z),
                0, total_shape.y,
                0, total_shape.x
            };

            // Read the block
            const idx3drange global_range_in = { // TODO These numbers are currently redundant, but will be needed once true 3d is implemented
                std::max(global_range.z_start-radius, (int64_t) 0), std::min(global_range.z_end+radius, total_shape.z),
                std::max(global_range.y_start-radius, (int64_t) 0), std::min(global_range.y_end+radius, total_shape.y),
                std::max(global_range.x_start-radius, (int64_t) 0), std::min(global_range.x_end+radius, total_shape.x)
            };
            const idx3d global_offset_in = {
                global_range_in.z_start == 0 ? radius : 0,
                0,
                0
            };

            // Ensure padding
            memset(input, 0, disk_global_flat_size);

            auto load_start = std::chrono::high_resolution_clock::now();
            load_file_strided(input, iter_input, total_shape, global_shape, global_range_in, global_offset_in);
            auto load_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> load_duration = load_end - load_start;
            std::cout << "Loading took " << load_duration.count() << " seconds at " << disk_global_flat_size/load_duration.count()/1e9 << " GB/s" << std::endl;

            //#pragma omp parallel for schedule(static) collapse(3)
            for (int64_t local_block_z = 0; local_block_z < local_blocks_z; local_block_z++) {
                for (int64_t local_block_y = 0; local_block_y < local_blocks_y; local_block_y++) {
                    for (int64_t local_block_x = 0; local_block_x < local_blocks_x; local_block_x++) {
                        idx3drange local_range = {
                            local_block_z*local_shape.z, (local_block_z+1)*local_shape.z,
                            local_block_y*local_shape.y, (local_block_y+1)*local_shape.y,
                            local_block_x*local_shape.x, (local_block_x+1)*local_shape.x
                        };

                        // Copy data to device
                        stage_to_device(d_input, input, local_range, global_shape, kernel_size);

                        #pragma omp target data map(to: d_input[0:local_flat_size]) map(alloc: d_mask[0:local_flat_size]) map(from: d_output[0:local_flat_size])
                        {
                            store_mask(d_input, d_mask, local_flat_size);
                            diffusion_core(d_input,  d_kernel, d_output, 0, kernel_size);
                            diffusion_core(d_output, d_kernel, d_input,  1, kernel_size);
                            diffusion_core(d_input,  d_kernel, d_output, 2, kernel_size);
                            illuminate(d_mask, d_output, local_flat_size);
                        }

                        // Copy data back to host
                        stage_to_host(output, d_output, local_range, global_shape, kernel_size);
                    }
                }
            }

            const idx3d global_offset_out = { radius, 0, 0 };

            store_file_strided(output, iter_output, total_shape, global_shape, global_range, global_offset_out);
        }
    }

    convert_float_to_uint8(repititions % 2 == 0 ? temp0 : temp1, output_file, total_flat_size);

    free(d_mask);
    free(d_kernel);
    free(d_output);
    free(d_input);
    free(output);
    free(input);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end-start;

    std::cout << "Diffusion took " << duration.count() << " seconds at " << (total_flat_size*sizeof(float)*repititions)/duration.count()/1e9 << " GB/s" << std::endl;
}