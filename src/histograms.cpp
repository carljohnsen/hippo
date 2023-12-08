#include "histograms.hpp"

// TODO for consistency, convert to OMP offload

void axis_histogram_par_gpu(const np_array<voxel_type> np_voxels,
                            const std::tuple<uint64_t,uint64_t,uint64_t> offset,
                            //const uint64_t outside_block_size,
                            np_array<uint64_t> &np_x_bins,
                            np_array<uint64_t> &np_y_bins,
                            np_array<uint64_t> &np_z_bins,
                            np_array<uint64_t> &np_r_bins,
                            const std::tuple<uint64_t, uint64_t> center,
                            const std::tuple<double, double> vrange,
                            const bool verbose) {
    if (verbose) {
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        tm local_tm = *localtime(&now);
        printf("Entered function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
    }

    pybind11::buffer_info
        voxels_info = np_voxels.request(),
        x_info = np_x_bins.request(),
        y_info = np_y_bins.request(),
        z_info = np_z_bins.request(),
        r_info = np_r_bins.request();

    const uint64_t
        image_length = voxels_info.size,
        voxel_bins   = x_info.shape[1],
        Nx = x_info.shape[0],
        Ny = y_info.shape[0],
        Nz = z_info.shape[0],
        Nr = r_info.shape[0];

    voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);

    uint64_t memory_needed = ((Nx*voxel_bins)+(Ny*voxel_bins)+(Nz*voxel_bins)+(Nr*voxel_bins))*sizeof(uint64_t);
    uint64_t
        *x_bins = (uint64_t*)x_info.ptr,
        *y_bins = (uint64_t*)y_info.ptr,
        *z_bins = (uint64_t*)z_info.ptr,
        *r_bins = (uint64_t*)r_info.ptr;

    auto [z_start, y_start, x_start] = offset;
    //uint64_t z_end   = std::min(z_start+outside_block_size, Nz);

    auto [vmin, vmax] = vrange;
    auto [cy, cx] = center;

    constexpr uint64_t
        GB = 1024 * 1024 * 1024,
        block_size = 1 * GB;

    uint64_t n_iterations = image_length / block_size;
    if (n_iterations * block_size < image_length)
        n_iterations++;

    //uint64_t initial_block = std::min(image_length, block_size);

    if (verbose) {
        printf("\nStarting %p: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%ld,%ld,%ld,%ld)\n", (void*) voxels,vmin, vmax, Nx,Ny,Nz,Nr);
        printf("Allocating result memory (%ld bytes (%.02f Mbytes))\n", memory_needed, memory_needed/1024./1024.);
        printf("Starting calculation\n");
        printf("Size of voxels is %ld bytes (%.02f Mbytes)\n", image_length * sizeof(voxel_type), (image_length * sizeof(voxel_type))/1024./1024.);
        printf("Blocksize is %ld bytes (%.02f Mbytes)\n", block_size * sizeof(voxel_type), (block_size * sizeof(voxel_type))/1024./1024.);
        printf("Doing %ld blocks\n", n_iterations);
        fflush(stdout);
    }

    auto start = std::chrono::steady_clock::now();

    // Copy the buffers to the GPU on entry and back to host on exit
    //#pragma acc data copy(x_bins[:Nx*voxel_bins], y_bins[:Ny*voxel_bins], z_bins[:Nz*voxel_bins], r_bins[:Nr*voxel_bins])
    #pragma omp target data map(tofrom: x_bins[:Nx*voxel_bins], y_bins[Ny*voxel_bins], z_bins[:Nz*voxel_bins], r_bins[:Nr*voxel_bins])
    {
        // For each block
        for (uint64_t i = 0; i < n_iterations; i++) {
            // Compute the block indices
            uint64_t this_block_start = i*block_size;
            uint64_t this_block_end = std::min(image_length, this_block_start + block_size);
            uint64_t this_block_size = this_block_end-this_block_start;
            voxel_type *buffer = voxels + this_block_start;

            // Copy the block to the GPU
            //#pragma acc data copyin(buffer[:this_block_size])
            #pragma omp target data map(to: buffer[:this_block_size])
            {
                // Compute the block
                //#pragma acc parallel loop
                #pragma omp target teams distribute parallel for
                for (uint64_t j = 0; j < this_block_size; j++) {
                    uint64_t flat_idx = i*block_size + j;
                    voxel_type voxel = buffer[j];
                    voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range

                    if (voxel != 0) { // Voxel not masked, and within vmin,vmax range
                        uint64_t x = flat_idx % Nx;
                        uint64_t y = (flat_idx / Nx) % Ny;
                        uint64_t z = (flat_idx / (Nx*Ny)) + z_start;
                        uint64_t r = floor(sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)));

                        int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                        //#pragma acc atomic
                        #pragma omp atomic
                        ++x_bins[x*voxel_bins + voxel_index];
                        //#pragma acc atomic
                        #pragma omp atomic
                        ++y_bins[y*voxel_bins + voxel_index];
                        //#pragma acc atomic
                        #pragma omp atomic
                        ++z_bins[z*voxel_bins + voxel_index];
                        //#pragma acc atomic
                        #pragma omp atomic
                        ++r_bins[r*voxel_bins + voxel_index];
                    }
                }
            }
        }
    }

    auto end = std::chrono::steady_clock::now();

    if (verbose) {
        std::chrono::duration<double> diff = end - start;
        printf("Compute took %.04f seconds\n", diff.count());
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        tm local_tm = *localtime(&now);
        printf("Exited function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
        fflush(stdout);
    }
}

void field_histogram_par_cpu(const np_array<voxel_type> np_voxels,
                             const np_array<field_type> np_field,
                             const std::tuple<uint64_t,uint64_t,uint64_t> offset,
                             const std::tuple<uint64_t,uint64_t,uint64_t> voxels_shape,
                             const std::tuple<uint64_t,uint64_t,uint64_t> field_shape,
                             const uint64_t block_size,
                             np_array<uint64_t> &np_bins,
                             const std::tuple<double, double> vrange,
                             const std::tuple<double, double> frange) {
    pybind11::buffer_info
        voxels_info = np_voxels.request(),
        field_info = np_field.request(),
        bins_info = np_bins.request();

    const uint64_t
        bins_length  = bins_info.size,
        field_bins   = bins_info.shape[0],
        voxel_bins   = bins_info.shape[1];

    auto [nZ, nY, nX] = voxels_shape;
    auto [nz, ny, nx] = field_shape;

    double dz = nz/((double)nZ), dy = ny/((double)nY), dx = nx/((double)nX);

    const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
    const field_type *field  = static_cast<field_type*>(field_info.ptr);
    uint64_t *bins = static_cast<uint64_t*>(bins_info.ptr);

    auto [f_min, f_max] = frange;
    auto [v_min, v_max] = vrange;
    auto [z_start, y_start, x_start] = offset;
    uint64_t
        z_end = std::min(z_start+block_size, nZ),
        y_end = nY,
        x_end = nX;

    #pragma omp parallel
    {
        uint64_t *tmp_bins = (uint64_t*) malloc(sizeof(uint64_t) * bins_length);
        #pragma omp for nowait
        for (uint64_t Z = 0; Z < z_end-z_start; Z++) {
            for (uint64_t Y = y_start; Y < y_end; Y++) {
                for (uint64_t X = x_start; X < x_end; X++) {
                    uint64_t flat_index = (Z*nY*nX) + (Y*nX) + X;
                    auto voxel = voxels[flat_index];
                    voxel = (voxel >= v_min && voxel <= v_max) ? voxel: 0; // Mask away voxels that are not in specified range
                    int64_t voxel_index = std::floor(static_cast<double>(voxel_bins-1) * ((voxel - v_min)/(v_max - v_min)) );

                    // And what are the corresponding x,y,z coordinates into the field array, and field basearray index i?
                    // TODO: Sample 2x2x2 volume?
                    uint64_t x = std::floor(X*dx), y = std::floor(Y*dy), z = std::floor(Z*dz);
                    uint64_t i = z*ny*nx + y*nx + x;

                    // TODO the last row of the histogram does not work, when the mask is "bright". Should be discarded.
                    if((voxel != 0) && (field[i] > 0)) { // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
                        int64_t field_index = std::floor(static_cast<double>(field_bins-1) * ((field[i] - f_min)/(f_max - f_min)) );

                        tmp_bins[field_index*voxel_bins + voxel_index]++;
                    }
                }
            }
        }
        #pragma omp critical
        {
            for (uint64_t i = 0; i < bins_length; i++)
                bins[i] += tmp_bins[i];
        }
        free(tmp_bins);
    }
}