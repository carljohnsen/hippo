#ifndef CC_HPP
#define CC_HPP

#include "hippo.hpp"

// Functions
void apply_renaming(std::vector<int64_t> &img, const std::vector<int64_t> &to_rename);
void apply_renaming(int64_t *__restrict__ img, const int64_t n, const std::vector<int64_t> &to_rename);
void canonical_names_and_size(const std::string &path, int64_t *__restrict__ out, const int64_t n_labels, const idx3d &total_shape, const idx3d &global_shape);
int64_t connected_components(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &global_shape, const bool verbose = false);
std::tuple<mapping, mapping> get_mappings(const std::vector<int64_t> &a, const int64_t n_labels_a, const std::vector<int64_t> &b, const int64_t n_labels_b, const idx3d &global_shape);
std::vector<int64_t> get_sizes(const std::vector<int64_t> &img, const int64_t n_labels);
std::vector<std::vector<std::tuple<int64_t, int64_t>>> generate_adjacency_tree(const int64_t chunks);
std::vector<idx3d> merge_canonical_names(const std::vector<idx3d> &names_a, const std::vector<idx3d> &names_b);
std::vector<int64_t> merge_labels(mapping &mapping_a, const mapping &mapping_b, const std::vector<int64_t> &to_rename_b);
int64_t recount_labels(const mapping &mapping_a, mapping &mapping_b, std::vector<int64_t> &to_rename_a, std::vector<int64_t> &to_rename_b);
std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t> relabel(const std::vector<int64_t> &a, const int64_t n_labels_a, const std::vector<int64_t> &b, const int64_t n_labels_b, const idx3d &global_shape, const bool verbose);
void rename_mapping(mapping &mapping_a, const std::vector<int64_t> &to_rename_other);

// Debugging functions
void print_canonical_names(const std::vector<idx3d> &names_a);
void print_mapping(const mapping &mapping_);
void print_rename(const std::vector<int64_t> &to_rename);

#endif // CC_HPP