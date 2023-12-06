#include "connected_components.hpp"

void apply_renaming(std::vector<int64_t> &img, std::vector<int64_t> &to_rename) {
    #pragma omp parallel for
    for (int64_t i = 0; i < img.size(); i++) {
        if (img[i] < to_rename.size()) {
            img[i] = to_rename[img[i]];
        }
    }
}

std::vector<dim3> canonical_name(std::vector<int64_t> &img, int64_t n_labels) {
    std::unordered_set<int64_t> labels;
    std::vector<bool> found(n_labels+1, false);
    std::vector<dim3> names(n_labels+1, {-1, -1, -1});
    for (int64_t i = 0; i < img.size(); i++) {
        labels.insert(img[i]);
        if (img[i] != 0 && !found[img[i]]) {
            int64_t
                z = i / (Ny_global * Nx_global),
                y = (i % (Ny_global * Nx_global)) / Nx_global,
                x = i % Nx_global;
            found[img[i]] = true;
            names[img[i]] = {z, y, x};
        }
    }

    return names;
}

std::tuple<mapping, mapping> get_mappings(std::vector<int64_t> &a, int64_t n_labels_a, std::vector<int64_t> &b, int64_t n_labels_b) {
    mapping mapping_a(n_labels_a+1);
    mapping mapping_b(n_labels_b+1);

    for (int64_t y = 0; y < Ny_global; y++) {
        for (int64_t x = 0; x < Nx_global; x++) {
            int64_t i = y * Nx_global + x;
            if (a[i] != 0 && b[i] != 0) {
                mapping_a[a[i]].insert(b[i]);
                mapping_b[b[i]].insert(a[i]);
            }
        }
    }

    return { mapping_a, mapping_b };
}

std::vector<int64_t> get_sizes(std::vector<int64_t> &img, int64_t n_labels) {
    std::vector<int64_t> sizes(n_labels, 0);
    for (int64_t i = 0; i < img.size(); i++) {
        sizes[img[i]]++;
    }

    return sizes;
}

std::vector<dim3> merge_canonical_names(std::vector<dim3> &names_a, std::vector<dim3> &names_b) {
    std::vector<dim3> names(names_a.size());
    for (int64_t i = 1; i < names_a.size(); i++) {
        if (names_a[i].z == -1) {
            names[i] = names_b[i];
        } else {
            names[i] = names_a[i];
        }
    }

    return names;
}

std::vector<int64_t> merge_labels(mapping &mapping_a, mapping &mapping_b, std::vector<int64_t> &to_rename_b) {
    std::list<int64_t> to_check;
    std::vector<int64_t> to_rename_a(mapping_a.size());
    to_rename_a[0] = 0;
    for (int64_t i = 1; i < mapping_a.size(); i++) {
        to_check.push_back(i);
        to_rename_a[i] = i;
    }
    bool updated;
    while (!to_check.empty()) {
        updated = false;
        int64_t label_a = to_check.front();
        std::unordered_set<int64_t> others_a = mapping_a[label_a];
        for (int64_t label_b : others_a) {
            if (label_b < to_rename_b.size()) { // Initially, the mapping will be empty
                label_b = to_rename_b[label_b];
            }
            std::unordered_set<int64_t> others_b = mapping_b[label_b];
            for (int64_t label_a2 : others_b) {
                label_a2 = to_rename_a[label_a2]; // Renames to self in the beginning
                if (label_a != label_a2) {
                    updated = true;
                    mapping_a[label_a].insert(mapping_a[label_a2].begin(), mapping_a[label_a2].end());
                    mapping_a[label_a2].clear();
                    mapping_a[label_a2].insert(-1);
                    to_rename_a[label_a2] = label_a;
                    to_check.remove(label_a2);
                }
            }
        }
        if (!updated) {
            to_check.pop_front();
        }
    }

    return to_rename_a;
}

void print_canonical_names(std::vector<dim3> &names_a) {
    std::cout << "Canonical names:" << std::endl;
    for (int64_t i = 1; i < names_a.size(); i++) {
        std::cout << i << ": " << names_a[i].z << " " << names_a[i].y << " " << names_a[i].x << std::endl;
    }
    std::cout << "----------------" << std::endl;
}

void print_mapping(mapping &mapping_) {
    std::cout << "Mapping:" << std::endl;
    for (int64_t i = 1; i < mapping_.size(); i++) {
        std::cout << i << ": { ";
        for (int64_t entry : mapping_[i]) {
            std::cout << entry << " ";
        }
        std::cout << "}" << std::endl;
    }
    std::cout << "----------------" << std::endl;
}

void print_rename(std::vector<int64_t> &to_rename) {
    std::cout << "Rename:" << std::endl;
    for (int64_t i = 1; i < to_rename.size(); i++) {
        std::cout << i << ": " << to_rename[i] << std::endl;
    }
    std::cout << "----------------" << std::endl;
}

// Ensures that the labels in the renaming LUTs are consecutive
int64_t recount_labels(mapping &mapping_a, mapping &mapping_b, std::vector<int64_t> &to_rename_a, std::vector<int64_t> &to_rename_b) {
    // We assume that mapping includes 0
    std::vector<int64_t> mapped_a, unmapped_a, unmapped_b;
    int64_t popped_a = 0, popped_b = 0;
    for (int64_t i = 1; i < mapping_a.size(); i++) {
        if (mapping_a[i].size() == 0) {
            unmapped_a.push_back(i);
        } else if (!mapping_a[i].contains(-1)) {
            mapped_a.push_back(i);
        } else {
            popped_a++;
        }
    }
    for (int64_t i = 1; i < mapping_b.size(); i++) {
        if (mapping_b[i].size() == 0) {
            unmapped_b.push_back(i);
        } else if (mapping_b[i].contains(-1)) {
            popped_b++;
        }
    }
    // Sanity check
    assert (mapped_a.size() + unmapped_a.size() == mapping_a.size()-popped_a-1);
    assert (mapped_a.size() + unmapped_b.size() == mapping_b.size()-popped_b-1);

    // Assign the first mapped_a labels to start from 1
    std::vector<int64_t> new_rename_a(mapping_a.size());
    for (int64_t i = 0; i < mapped_a.size(); i++) {
        new_rename_a[mapped_a[i]] = i+1;
    }
    // Assign the unmapped_a labels to start from mapped_a.size()+1
    for (int64_t i = 0; i < unmapped_a.size(); i++) {
        new_rename_a[unmapped_a[i]] = i+1+mapped_a.size();
    }

    // Apply the new renaming to the renaming LUT
    for (int64_t i = 0; i < to_rename_a.size(); i++) {
        to_rename_a[i] = new_rename_a[to_rename_a[i]];
    }

    // TODO is this actually necessary? We'll see.
    // Update mapping b to use the new a labels
    for (int64_t i = 1; i < mapping_b.size(); i++) {
        auto entries = mapping_b[i];
        std::unordered_set<int64_t> new_entries;
        for (int64_t entry : entries) {
            if (entry != -1) {
                new_entries.insert(new_rename_a[entry]);
            }
        }
        mapping_b[i] = new_entries;
    }

    // Assign the first mapped_b labels to match the mapped_a labels
    std::vector<int64_t> new_rename_b(mapping_b.size());
    for (int64_t i = 0; i < mapped_a.size(); i++) {
        auto label = mapped_a[i];
        auto new_label = to_rename_a[label];
        auto entries = mapping_a[label];
        for (int64_t entry : entries) {
            if (entry != -1) {
                new_rename_b[entry] = new_label;
            }
        }
    }
    // Assign the unmapped_b labels to start from 1+mapped_a.size()+unmapped_a.size()
    for (int64_t i = 0; i < unmapped_b.size(); i++) {
        new_rename_b[unmapped_b[i]] = i+1+mapped_a.size()+unmapped_a.size();
    }
    // Apply the new renaming to the renaming LUT
    for (int64_t i = 0; i < to_rename_b.size(); i++) {
        to_rename_b[i] = new_rename_b[to_rename_b[i]];
    }

    return mapped_a.size() + unmapped_a.size() + unmapped_b.size();
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t> relabel(std::vector<int64_t> &a, int64_t n_labels_a, std::vector<int64_t> &b, int64_t n_labels_b) {
    auto start = std::chrono::high_resolution_clock::now();
    auto [mapping_a, mapping_b] = get_mappings(a, n_labels_a, b, n_labels_b);
    auto mappings_end = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> empty_vec;
    auto to_rename_a = merge_labels(mapping_a, mapping_b, empty_vec);
    auto merge_a_end = std::chrono::high_resolution_clock::now();
    auto to_rename_b = merge_labels(mapping_b, mapping_a, to_rename_a);
    auto merge_b_end = std::chrono::high_resolution_clock::now();
    rename_mapping(mapping_a, to_rename_b);
    auto rename_a_end = std::chrono::high_resolution_clock::now();
    rename_mapping(mapping_b, to_rename_a);
    auto rename_b_end = std::chrono::high_resolution_clock::now();
    int64_t n_new_labels = recount_labels(mapping_a, mapping_b, to_rename_a, to_rename_b);
    auto recount_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double>
        elapsed_get_mappings = mappings_end - start,
        elapsed_merge_a = merge_a_end - mappings_end,
        elapsed_merge_b = merge_b_end - merge_a_end,
        elapsed_rename_a = rename_a_end - merge_b_end,
        elapsed_rename_b = rename_b_end - rename_a_end,
        elapsed_recount = recount_end - rename_b_end;

    std::cout << "get_mappings: " << elapsed_get_mappings.count() << " s" << std::endl;
    std::cout << "merge_a: " << elapsed_merge_a.count() << " s" << std::endl;
    std::cout << "merge_b: " << elapsed_merge_b.count() << " s" << std::endl;
    std::cout << "rename_a: " << elapsed_rename_a.count() << " s" << std::endl;
    std::cout << "rename_b: " << elapsed_rename_b.count() << " s" << std::endl;
    std::cout << "recount: " << elapsed_recount.count() << " s" << std::endl;

    return { to_rename_a, to_rename_b, n_new_labels };
}

void rename_mapping(mapping &mapping_a, std::vector<int64_t> &to_rename_other) {
    for (int64_t i = 1; i < mapping_a.size(); i++) {
        auto entries = mapping_a[i];
        std::unordered_set<int64_t> new_entries;
        for (int64_t entry : entries) {
            if (entry != -1) {
                new_entries.insert(to_rename_other[entry]);
            } else {
                new_entries.insert(-1);
            }
        }
        mapping_a[i] = new_entries;
    }
}