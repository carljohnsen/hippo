#ifndef cc_hpp
#define cc_hpp

#include "hippo.hpp"

// Functions
void apply_renaming(std::vector<int64_t> &img, std::vector<int64_t> &to_rename);
std::vector<dim3> canonical_name(std::vector<int64_t> &img, int64_t n_labels);
std::tuple<mapping, mapping> get_mappings(std::vector<int64_t> &a, int64_t n_labels_a, std::vector<int64_t> &b, int64_t n_labels_b);
std::vector<int64_t> get_sizes(std::vector<int64_t> &img, int64_t n_labels);
template <typename T>
std::vector<dim3> merge_canonical_names(std::vector<dim3> &names_a, std::vector<dim3> &names_b);
std::vector<int64_t> merge_labels(mapping &mapping_a, mapping &mapping_b, std::vector<int64_t> &to_rename_b);
int64_t recount_labels(mapping &mapping_a, mapping &mapping_b, std::vector<int64_t> &to_rename_a, std::vector<int64_t> &to_rename_b);
void rename_mapping(mapping &mapping_a, std::vector<int64_t> &to_rename_other);
std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t> relabel(std::vector<int64_t> &a, int64_t n_labels_a, std::vector<int64_t> &b, int64_t n_labels_b);

// Debugging functions
void print_canonical_names(std::vector<dim3> &names_a);
void print_mapping(mapping &mapping_);
void print_rename(std::vector<int64_t> &to_rename);

#endif /* cc_hpp */