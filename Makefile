CXXFLAGS=-Wall -Wextra -Werror -pedantic -std=c++20 -march=native -mtune=native -fopenmp -Iinclude/ -fPIC -g -O3
#CXXFLAGS+=-foffload=nvptx-none -fcf-protection=none -fno-stack-protector -no-pie -foffload="-O3 -g" -fsanitize=address
CXXFLAGS+=-mp=gpu -Minfo=mp
#CXXFLAGS+=-fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75 --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.0 -v
PYTHON=python3.10
PYBIND_FLAGS=$(shell $(PYTHON) -m pybind11 --includes)
#CXXFLAGS+=$(PYBIND_FLAGS)
CXXFLAGS+=$(subst -I,-isystem ,$(PYBIND_FLAGS)) # We don't care about warnings from the python headers
PYBIND_SUFFIX=$(shell $(PYTHON)-config --extension-suffix)

all:
	$(CC) --version

#bin/mains: obj/mains.o obj/general.o obj/connected_components.o obj/diffusion.o obj/histograms.o
bin/mains: obj/mains.o obj/general.o obj/diffusion.o
	@mkdir -p bin
	nvc++ $(CXXFLAGS) -o $@ $^

obj/%.o: src/%.cpp
	@mkdir -p obj
	nvc++ $(CXXFLAGS) -c -o $@ $<

generate_data:
	@mkdir -p output
	python3 scripts/generate_data.py data/input_img.uint8

clean:
	rm -rf obj/ bin/ output/