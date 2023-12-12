import helpers
import numpy as np
import sys

def generate_input_img(outpath, N = (1024, 1024, 1024)):
    if not outpath.endswith('.uint8'):
        print ('Warning: file will be saved as a raw uint8 file, but the extension is not .uint8')

    consts = helpers.parse_header('include/hippo.hpp')

    Nz, Ny, Nx = N

    img = np.zeros((Nz, Ny, Nx), dtype=np.uint8)

    # Set the 1 pixel borders to 255
    img[ 0,  :,  :] = 255
    img[-1,  :,  :] = 255
    img[ :,  0,  :] = 255
    img[ :, -1,  :] = 255
    img[ :,  :,  0] = 255
    img[ :,  :, -1] = 255

    # Set the square center of radius C to 255
    C = int(consts['C'])
    img[Nz//2-C:Nz//2+C, Ny//2-C:Ny//2+C, Nx//2-C:Nx//2+C] = 255

    # Create a 'crossing noodle' around the center
    img[Nz//2-3*C:Nz//2-2*C, Ny//2, Nx//2-3*C:Nx//2+3*C] = 255
    img[Nz//2-3*C:Nz//2+2*C, Ny//2, Nx//2-3*C:Nx//2-2*C] = 255
    img[Nz//2-3*C:Nz//2+2*C, Ny//2, Nx//2+2*C:Nx//2+3*C] = 255

    # Create an even 'crossier noodle' around the center
    img[Nz//2-6*C:Nz//2+6*C, Ny//2, Nx//2-5*C:Nx//2-4*C] = 255
    img[Nz//2-6*C:Nz//2+6*C, Ny//2, Nx//2+4*C:Nx//2+5*C] = 255
    img[Nz//2+5*C:Nz//2+6*C, Ny//2, Nx//2-5*C:Nx//2-2*C] = 255
    img[Nz//2+5*C:Nz//2+6*C, Ny//2, Nx//2+2*C:Nx//2+5*C] = 255
    img[Nz//2+3*C:Nz//2+6*C, Ny//2, Nx//2-3*C:Nx//2-2*C] = 255
    img[Nz//2+3*C:Nz//2+6*C, Ny//2, Nx//2+2*C:Nx//2+3*C] = 255
    img[Nz//2+3*C:Nz//2+4*C, Ny//2, Nx//2-3*C:Nx//2+3*C] = 255

    # A lone bar
    img[Nz//2-6*C:Nz//2-5*C, Ny//2, Nx//2-3*C:Nx//2+3*C] = 255

    # Save the image as a raw uint8 file
    img.tofile(outpath)

if __name__ == '__main__':
    # TODO argparse?
    outpath = sys.argv[1] if len(sys.argv) > 1 else 'data/input_img.uint8'