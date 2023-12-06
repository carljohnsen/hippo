import helpers
import numpy as np

if __name__ == '__main__':
    consts = helpers.parse_header('include/hippo.hpp')

    Nz = int(consts['Nz_total'])
    Ny = int(consts['Ny_total'])
    Nx = int(consts['Nx_total'])

    img = np.zeros((Nz, Ny, Nx), dtype=np.uint8)

    # Set the 1 pixel borders to 255
    img[0, :, :] = 255
    img[-1, :, :] = 255
    img[:, 0, :] = 255
    img[:, -1, :] = 255
    img[:, :, 0] = 255
    img[:, :, -1] = 255

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
    img.tofile('data/input_img.uint8')
