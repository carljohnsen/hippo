#include "hippo.hpp"

// TODO handle aligned allocation
template <typename T>
std::vector<T> load_file(const std::string &path, const int64_t offset, const int64_t n_elements) {
    std::vector<T> data(n_elements);
    load_file(data.data(), path, offset, n_elements);
    return data;
}

template <typename T>
void load_file(T *dst, const std::string &path, const int64_t offset, const int64_t n_elements) {
    FILE *fp;
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    fp = fdopen(fd, "rb");
    fseek(fp, offset*sizeof(T), SEEK_SET);
    fread((char *) dst, sizeof(T), n_elements, fp);
    fclose(fp);
}

// TODO handle aligned allocation
template <typename T>
std::vector<T> load_file_strided(const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range) {
    std::vector<T> data(shape.z*shape.y*shape.x);
    load_file_strided(data.data(), path, disk_shape, shape, range);
    return data;
}

template <typename T>
void load_file_strided(T *dst, const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range) {
    const idx3d
        strides_global = {disk_shape.y*disk_shape.x, disk_shape.x, 1},
        strides_local = {shape.y*shape.x, shape.x, 1};

    if (shape.z == disk_shape.z && shape.y == disk_shape.y && shape.x == disk_shape.x) {
        load_file(dst, path, 0, disk_shape.z*disk_shape.y*disk_shape.x);
        return;
    }

    FILE *fp;
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    fp = fdopen(fd, "rb");
    fseek(fp, range.z_start*strides_global.z + range.y_start*strides_global.y + range.x_start*strides_global.x*sizeof(T), SEEK_SET);
    for (int64_t z = 0; z < shape.z; z++) {
        for (int64_t y = 0; y < shape.y; y++) {
            fread((char *) &dst[z*strides_local.z + y*strides_local.y], sizeof(T), shape.x, fp);
            fseek(fp, (strides_global.y - shape.x)*sizeof(T), SEEK_CUR);
        }
        fseek(fp, (strides_global.y - shape.y)*strides_global.y*sizeof(T), SEEK_CUR);
    }
    fclose(fp);
}

template <typename T>
void store_file(std::vector<T> &data, std::string &path, int64_t offset) {
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

template <typename T>
void store_file(T *data, std::string &path, int64_t offset, int64_t n_elements) {
    FILE *fp;
    int fd = open(path.c_str(), O_CREAT | O_RDWR | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    fp = fdopen(fd, "r+b");
    fseek(fp, offset*sizeof(T), SEEK_SET);
    fwrite((char *) data, sizeof(T), n_elements, fp);
    fclose(fp);
}

template <typename T>
void store_file_strided(const std::vector<T> &data, const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range) {
    store_file_strided(data.data(), path, disk_shape, shape, range);
}

template <typename T>
void store_file_strided(const T *data, const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range) {
    const idx3d
        strides_global = {disk_shape.y*disk_shape.x, disk_shape.x, 1},
        strides_local = {shape.y*shape.x, shape.x, 1};

    if (shape.z == disk_shape.z && shape.y == disk_shape.y && shape.x == disk_shape.x) {
        store_file(data, path, 0, disk_shape.z*disk_shape.y*disk_shape.x);
        return;
    }

    FILE *fp;
    int fd = open(path.c_str(), O_CREAT | O_RDWR | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    fp = fdopen(fd, "r+b");
    fseek(fp, range.z_start*strides_global.z + range.y_start*strides_global.y + range.x_start*strides_global.x*sizeof(T), SEEK_SET);
    for (int64_t z = 0; z < shape.z; z++) {
        for (int64_t y = 0; y < shape.y; y++) {
            fwrite((char *) &data[z*strides_local.z + y*strides_local.y], sizeof(T), shape.x, fp);
            fseek(fp, (strides_global.y - shape.x)*sizeof(T), SEEK_CUR);
        }
        fseek(fp, (strides_global.y - shape.y)*strides_global.y*sizeof(T), SEEK_CUR);
    }
    fclose(fp);
}