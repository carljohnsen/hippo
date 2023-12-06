CLFLAGS=-O3 -Wall -Wextra -Werror -pedantic -std=c++20 -O3 -march=native -mtune=native -fopenmp -Iinclude/

all:
	${CC} --version

generate_data:
	python3 scripts/generate_data.py output/input_img.uint8

clean:
	rm -rf obj/ bin/ output/