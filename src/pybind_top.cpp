#include "connected_components.hpp"
#include "diffusion.hpp"

namespace python_api {

    int64_t connected_components_wrapper(const std::string &base_path, const pybind11::array_t<int64_t> &n_labels, const std::tuple<int64_t, int64_t, int64_t> &global_shape, const bool verbose = false) {
        // Convert pybind11::array_t<int64_t> to std::vector<int64_t>
        auto n_labels_info = n_labels.request();
        std::vector<int64_t> n_labels_vec(n_labels_info.size);
        std::memcpy(n_labels_vec.data(), n_labels_info.ptr, n_labels_info.size * sizeof(int64_t));

        // Convert std::tuple<int64_t, int64_t, int64_t> to idx3d
        idx3d global_shape_ = { std::get<0>(global_shape), std::get<1>(global_shape), std::get<2>(global_shape) };

        // Call connected_components function in the global namespace
        return connected_components(base_path, n_labels_vec, global_shape_, verbose);
    }

    void connected_components_canonical_names_and_sizes(const std::string &path, pybind11::array_t<int64_t> &out, const std::tuple<int64_t, int64_t, int64_t> &total_shape, const std::tuple<int64_t, int64_t, int64_t> &global_shape) {
        // Extract the pointer and shape from the output array
        auto out_info = out.request();
        int64_t *out_ptr = (int64_t *) out_info.ptr;
        int64_t n_labels = out_info.shape[0];
        assert (out_info.ndim == 2 && out_info.shape[1] == 4 && "The output array must have shape (n_labels, 4)");
        idx3d total_shape_ = { std::get<0>(total_shape), std::get<1>(total_shape), std::get<2>(total_shape) };
        idx3d global_shape_ = { std::get<0>(global_shape), std::get<1>(global_shape), std::get<2>(global_shape) };

        // Ensure that the sizes (every fourth element) is initialized to 0
        for (int64_t i = 0; i < n_labels; i++) {
            out_ptr[(i*4)+3] = 0;
        }

        // Call connected_components_canonical_names function in the global namespace
        canonical_names_and_size(path, out_ptr, n_labels, total_shape_, global_shape_);
    }

    void diffusion_wrapper(const std::string &input_file, const pybind11::array_t<float> &kernel, const std::string &output_file, const std::tuple<int64_t, int64_t, int64_t> &total_shape, const std::tuple<int64_t, int64_t, int64_t> &global_shape, const int64_t repititions, const bool verbose) {
        // Convert pybind11::array_t<float> to std::vector<float>
        auto kernel_info = kernel.request();
        std::vector<float> kernel_vec(kernel_info.size);
        std::memcpy(kernel_vec.data(), kernel_info.ptr, kernel_info.size * sizeof(float));

        // Convert std::tuple<int64_t, int64_t, int64_t> to idx3d
        idx3d
            total_shape_  = { std::get<0>(total_shape),  std::get<1>(total_shape),  std::get<2>(total_shape)  },
            global_shape_ = { std::get<0>(global_shape), std::get<1>(global_shape), std::get<2>(global_shape) };

        // Call diffusion function in the global namespace
        diffusion(input_file, kernel_vec, output_file, total_shape_, global_shape_, repititions, verbose);
    }

}

PYBIND11_MODULE(hippo, m) {
    m.doc() = "Diffusion module written in C++"; // optional module docstring

    m.def("connected_components", &python_api::connected_components_wrapper, pybind11::arg("base_path"), pybind11::arg("n_labels"), pybind11::arg("global_shape"), pybind11::arg("verbose") = false, "Find the connected components in the given image");
    m.def("connected_components_canonical_names_and_sizes", &python_api::connected_components_canonical_names_and_sizes, pybind11::arg("path"), pybind11::arg("out").noconvert(), pybind11::arg("total_shape"), pybind11::arg("global_shape"), "Find the connected components in the given image and return the canonical names and sizes");
    m.def("diffusion", &python_api::diffusion_wrapper, pybind11::arg("input_file"), pybind11::arg("kernel"), pybind11::arg("output_file"), pybind11::arg("total_shape"), pybind11::arg("global_shape"), pybind11::arg("repititions"), pybind11::arg("verbose") = false, "Diffuse the input image using the given kernel");
}