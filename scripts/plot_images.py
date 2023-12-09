import helpers
import numpy as np
import sys

if __name__ == '__main__':
    dtype_lut = {
        'float32': np.float32,
        'int64': np.int64,
        'uint8': np.uint8
    }
    inpath = sys.argv[1] if len(sys.argv) > 1 else 'data/output_img.uint8'
    outpath = sys.argv[2] if len(sys.argv) > 2 else 'output_img.png'
    consts = helpers.parse_header('include/hippo.hpp')
    shape = (int(consts['Nz_total']), int(consts['Ny_total']), int(consts['Nx_total']))
    helpers.plot_middle_planes(inpath, shape, dtype_lut[inpath.split('.')[-1]], outpath)