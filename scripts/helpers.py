import cv2
import numpy as np
import matplotlib.pyplot as plt

def parse_header(path, verbose=False):
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

    if verbose:
        print (consts)
    return consts

# Assumes that the input is a cube
def plot_middle_planes(inpath, dtype, outpath, total_shape=None, verbose=False):
    with open(inpath) as f:
        img = np.fromfile(f, dtype=dtype)
        if verbose:
            print (f'Read {img.shape} from {inpath}')
            print (f'Max value: {img.max()}')
            print (f'All zeros: {(img == 0).all()}')
        if total_shape is None:
            N = int(np.ceil(img.shape[0] ** (1/3)))
            if verbose:
                print (f'Trying to reshape to {N}^3')
            total_shape = (N, N, N)
        img = img.reshape(total_shape)
        if dtype == np.float32:
            img *= 255 # Convert 0-1 to 0-255
        plane_yx = img[total_shape[0]//2, :, :]
        plane_zx = img[:, total_shape[1]//2, :]
        plane_zy = img[:, :, total_shape[2]//2]
        if total_shape[0] == total_shape[1] == total_shape[2]:
            img = np.hstack((plane_yx, plane_zx, plane_zy))
            cv2.imwrite(outpath, img)
            if verbose:
                print (f'Wrote image to {outpath}')
        else:
            cv2.imwrite(outpath.replace('.png', '_yx.png'), plane_yx)
            cv2.imwrite(outpath.replace('.png', '_zx.png'), plane_zx)
            cv2.imwrite(outpath.replace('.png', '_zy.png'), plane_zy)
            if verbose:
                print (f'Wrote images to {outpath.replace(".png", "_yx.png")}, {outpath.replace(".png", "_zx.png")}, and {outpath.replace(".png", "_zy.png")}')