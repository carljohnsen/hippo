#include "hippo.hpp"

template <typename T>
std::vector<T> load_file(std::string &path, int64_t offset, int64_t n_elements) {
    std::vector<T> data(n_elements);
    std::ifstream infile(path, std::ios::binary);
    infile.seekg(offset*sizeof(T), std::ios::beg);
    infile.read(reinterpret_cast<char*>(data.data()), n_elements*sizeof(T));
    infile.close();
    return data;
}

template <typename T>
void store_file(std::string &path, std::vector<T> &data, int64_t offset) {
    std::ofstream file;
    file.open(path, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        file.clear();
        file.open(path, std::ios::binary | std::ios::out);
    }
    file.seekp(offset*sizeof(T), std::ios::beg);
    file.write(reinterpret_cast<char*>(data.data()), data.size()*sizeof(T));
    file.flush();
    file.close();
}