import cv2
import numpy as np
import matplotlib.pyplot as plt

def parse_header(path):
    # Read constants from header file. They won't all be used, but it's easier to
    # just read them all and then pick the ones we want.
    consts = dict()
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

        defines = [line for line in lines if line.startswith('#define')]
        for line in defines:
            line = line.replace('#define ', '')
            tokens = line.split(' ')
            consts[tokens[0]] = tokens[-1]

        assigns = [line for line in lines if '=' in line.split(' ')]
        for line in assigns:
            tokens = line.split(' ')
            eq_sign = tokens.index('=')
            consts[tokens[eq_sign-1]] = tokens[eq_sign+1].strip(',;')

    print (consts)
    return consts

def plot_middle_planes(inpath, shape, dtype, outpath):
    with open(inpath) as f:
        Nz, Ny, Nx = shape
        img = np.fromfile(f, dtype=dtype)
        print (f'Read {img.shape} from {inpath}')
        print (f'Max value: {img.max()}')
        print (f'All zeros: {(img == 0).all()}')
        img = img.reshape((Nz, Ny, Nx))
        img = np.hstack((img[Nz//2, :, :], img[:, Ny//2, :], img[:, :, Nx//2]))
        if dtype == np.float32:
            img *= 255 # Convert 0-1 to 0-255
        cv2.imwrite(outpath, img)
        print (f'Wrote image to {outpath}')