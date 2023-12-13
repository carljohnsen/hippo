# Append the relative path '../bin' of this file to the system path
import pathlib
file_path = pathlib.Path(__file__).parent.absolute()
import sys
sys.path.append(f'{file_path}')
sys.path.append(f'{file_path}/../bin')

import datetime
import helpers
import hippo
import generate_data
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as ndi

def verify_diffusion(): # TODO better output names to correspond to diffusion
    # Constants
    sigma = 5.0
    r = 4.0 * sigma
    padding = int(2*r)
    total_shape = (256, 256, 256)
    planes_per_gb = int(1024**3 / (total_shape[1] * total_shape[2] * np.dtype(np.float32).itemsize))
    global_shape = (planes_per_gb, total_shape[1], total_shape[2])
    #global_shape = (total_shape[0]+padding, total_shape[1], total_shape[2])
    repititions = 1

    # 1 GPU for 1024**3 and planes_per_gb * 1024**2
    #ndi.gaussian_filter took 264.315 seconds
    #hippo.diffusion took 20.742 seconds
    #hippo.diffusion was 12.74 times faster than ndi.gaussian_filter
    #Average absolute difference: 9.313225746154785e-10
    #Maximum absolute difference: 1
    #Minimum absolute difference: 0
    #Standard deviation of absolute difference: 3.0517578110789142e-05

    # TODO investigate why multiple GPUs are slower.
    # 2 GPU for 1024**3 and planes_per_gb * 1024**2
    #ndi.gaussian_filter took 199.082 seconds
    #hippo.diffusion took 30.037 seconds
    #hippo.diffusion was 6.63 times faster than ndi.gaussian_filter
    #Average absolute difference: 9.313225746154785e-10
    #Maximum absolute difference: 1
    #Minimum absolute difference: 0
    #Standard deviation of absolute difference: 3.0517578110789142e-05

    # File paths
    prefix = 'diffusion'
    data_folder = 'data' # Input and output data generated during the run.
    output_folder = 'output' # Plots generated during the run.
    input_data_path = f'{data_folder}/{prefix}_input.uint8'
    hippo_data_path = f'{data_folder}/{prefix}_hippo.uint8'
    ndied_data_path = f'{data_folder}/{prefix}_ndied.uint8'
    diff_data_path = f'{data_folder}/{prefix}_diff.uint8'
    input_img_path = f'{output_folder}/{prefix}_input.png'
    hippo_img_path = f'{output_folder}/{prefix}_hippo.png'
    ndied_img_path = f'{output_folder}/{prefix}_ndied.png'
    diff_img_path = f'{output_folder}/{prefix}_diff.png'

    # Create a 1D Gaussian
    x = np.arange(-r, r + 1)
    kernel = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))

    # Generate and plot the input image
    generate_data.generate_input_img(input_data_path, total_shape)
    helpers.plot_middle_planes(input_data_path, np.uint8, input_img_path)

    # Load the input image
    ndied_start = datetime.datetime.now()
    mask = np.fromfile(input_data_path, dtype=np.uint8).reshape(total_shape)

    # Apply the Gaussian kernel to the input image in 3D
    ndied = np.zeros(total_shape, dtype=np.float32)
    ndied[mask == 255] = 1.0
    for _ in range(repititions):
        for i in range(3):
            ndi.convolve1d(ndied, kernel, axis=i, output=ndied, mode='constant', cval=0)
        ndied[mask == 255] = 1.0
    ndied*=255
    ndied = ndied.astype(np.uint8)

    # Save the image as a raw uint8 file
    ndied.tofile(ndied_data_path)
    ndied_end = datetime.datetime.now()
    ndied_duration = (ndied_end - ndied_start).total_seconds()
    print (f'ndi.gaussian_filter took {ndied_duration:.03f} seconds')

    # Plot the middle planes of the output image
    helpers.plot_middle_planes(ndied_data_path, np.uint8, ndied_img_path)

    # Run the hippo diffusion
    hippo_start = datetime.datetime.now()
    hippo.diffusion(input_data_path, kernel, hippo_data_path, total_shape, global_shape, repititions)
    hippo_end = datetime.datetime.now()
    hippo_duration = (hippo_end - hippo_start).total_seconds()
    print (f'hippo.diffusion took {hippo_duration:.03f} seconds')
    print (f'hippo.diffusion was {ndied_duration / hippo_duration:.02f} times faster than ndi.gaussian_filter')

    # Plot the middle planes of the output image
    helpers.plot_middle_planes(hippo_data_path, np.uint8, hippo_img_path)

    # Compare the output images
    diff = np.abs(ndied.astype(np.int32) - np.fromfile(hippo_data_path, dtype=np.uint8).reshape(total_shape).astype(np.int32))
    print (f'Average absolute difference: {np.mean(diff)}')
    print (f'Maximum absolute difference: {np.max(diff)}')
    print (f'Minimum absolute difference: {np.min(diff)}')
    print (f'Standard deviation of absolute difference: {np.std(diff)}')

    # Save and plot the difference image
    diff.astype(np.uint8).tofile(diff_data_path)
    helpers.plot_middle_planes(diff_data_path, np.uint8, diff_img_path)

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('output'):
        os.makedirs('output')

    verify_diffusion()