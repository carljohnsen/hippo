CXXFLAGS=-O3 -Wall -Wextra -Werror -pedantic -std=c++20 -march=native -mtune=native -fopenmp -Iinclude/ -fPIC
PYTHON=python3.10
PYBIND_FLAGS=$(shell $(PYTHON) -m pybind11 --includes)
CXXFLAGS+=$(PYBIND_FLAGS)
PYBIND_SUFFIX=$(shell $(PYTHON)-config --extension-suffix)

all:
	${CC} --version

obj/%.o: src/%.cpp
	@mkdir -p obj
	${CC} ${CXXFLAGS} -c -o $@ $<

generate_data:
	python3 scripts/generate_data.py output/input_img.uint8

clean:
	rm -rf obj/ bin/ output/