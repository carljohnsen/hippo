# hippo
Artefact repository for HIPPO paper submission

TODO define hierachy shapes: total, global and local.

# Quickstart
```bash
make -j benchmarks
bin/benchmarks
```

## Dependencies
These were built and tested on Ubuntu 22.04 LTS and LUMI with the following libraries
### Ubuntu 22.04 LTS
* `g++` 11.4.0 with `std=c++20`
* CUDA 11.8 and 12.2
* Python 3.10.12
  * `numpy` 1.23.5
  * `matplotlib` 3.6.2
  * `scipy` 1.9.3
  * `cv2` 4.6.0

TODO make a `requirements.txt`

## Configuration
Constants are defined in `include/hippo.hpp` and are used by both the C++ and Python implementations.

## Building
```bash
make -j
```

## Running
```bash
bin/hippo
```

# Implementations
## Connected components
## Diffusion approximation
## 2D histograms
