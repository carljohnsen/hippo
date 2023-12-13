#include "connected_components.hpp"

void apply_renaming(std::vector<int64_t> &img, std::vector<int64_t> &to_rename) {
    apply_renaming(img.data(), img.size(), to_rename);
}

void apply_renaming(int64_t *__restrict__ img, const int64_t n, const std::vector<int64_t> &to_rename) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        if (img[i] < (int64_t) to_rename.size()) {
            img[i] = to_rename[img[i]];
        }
    }
}

void canonical_names_and_size(const std::string &path, int64_t *__restrict__ out, const int64_t n_labels, const idx3d &total_shape, const idx3d &global_shape) {
    std::vector<bool> found(n_labels+1, false);
    const idx3d strides = { global_shape.y * global_shape.x, global_shape.x, 1 };
    int64_t n_chunks = total_shape.z / global_shape.z; // Assuming that they are divisible
    FILE *file = open_file_read(path);
    int64_t chunk_size = global_shape.z * global_shape.y * global_shape.x;
    int64_t *img = (int64_t *) aligned_alloc(disk_block_size, chunk_size * sizeof(int64_t));
    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        std::cout << "Chunk " << chunk << " / " << n_chunks << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        load_partial(img, file, chunk*chunk_size, chunk_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "load_partial: " << (chunk_size*sizeof(int64_t)) / elapsed.count() / 1e9 << " GB/s" << std::endl;
        for (int64_t i = 0; i < chunk_size; i++) {
            int64_t label = img[i];
            if (label > n_labels || label < 0) {
                std::cout << "Label " << label << " in chunk " << chunk << " at index " << i <<" is outside of bounds 0:" << n_labels << std::endl;
                continue;
            }
            if (!found[label]) {
                int64_t
                    z = (i / strides.z) + (chunk * global_shape.z),
                    y = (i % strides.z) / strides.y,
                    x = (i % strides.y) / strides.x;
                found[label] = true;
                out[(4*label)+0] = z;
                out[(4*label)+1] = y;
                out[(4*label)+2] = x;
            }
            out[(4*label)+3] += 1;
        }
    }
    free(img);
    fclose(file);
}

int64_t connected_components(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &global_shape, const bool verbose) {
    auto cc_start = std::chrono::high_resolution_clock::now();
    // Check if the call is well-formed
    int64_t chunks = n_labels.size();
    assert ((chunks & (chunks - 1)) == 0 && "Chunks must be a power of 2");

    // Constants
    const idx3d
        global_strides = { global_shape.y * global_shape.x, global_shape.x, 1 };

    // Generate the paths to the different chunks
    std::vector<std::string> paths(chunks);
    for (int64_t i = 0; i < chunks; i++) {
        paths[i] = base_path + std::to_string(i) + ".int64";
    }

    // Generate the adjacency tree
    std::vector<std::vector<std::tuple<int64_t, int64_t>>> index_tree = generate_adjacency_tree(chunks);

    std::vector<std::vector<int64_t>> renames(chunks); // Rename LUTs, one for each chunk
    for (int64_t i = 0; i < (int64_t) index_tree.size(); i++) {
        //#pragma omp parallel for
        for (int64_t j = 0; j < (int64_t) index_tree[i].size(); j++) {
            auto [l, r] = index_tree[i][j];
            // TODO Handle when all chunks doesn't have the same shape.
            int64_t last_layer = (global_shape.z-1) * global_strides.z;
            std::vector<int64_t> a = load_file<int64_t>(paths[l], last_layer, global_strides.z);
            std::vector<int64_t> b = load_file<int64_t>(paths[r], 0, global_strides.z);

            if (i > 0) {
                // Apply the renamings obtained from the previous layer
                apply_renaming(a, renames[l]);
                apply_renaming(b, renames[r]);
            }
            auto [rename_l, rename_r, n_new_labels] = relabel(a, n_labels[l], b, n_labels[r], global_shape, verbose);
            n_labels[l] = n_new_labels;
            n_labels[r] = n_new_labels;

            if (i > 0) {
                // Run through the left subtree
                int64_t subtrees = i << 1;
                for (int64_t k = j*2*subtrees; k < (j*2*subtrees)+subtrees; k++) {
                    apply_renaming(renames[k], rename_l);
                    n_labels[k] = n_new_labels;
                }

                // Run through the right subtree
                for (int64_t k = (j*2*subtrees)+subtrees; k < (j*2*subtrees)+(2*subtrees); k++) {
                    apply_renaming(renames[k], rename_r);
                    n_labels[k] = n_new_labels;
                }
            } else {
                renames[l] = rename_l;
                renames[r] = rename_r;
            }
        }
    }

    auto cc_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cc = cc_end - cc_start;
    if (verbose) {
        std::cout << "connected_components lut building: " << elapsed_cc.count() << " s" << std::endl;
    }

    auto cc_app_start = std::chrono::high_resolution_clock::now();

    // Apply the renaming to a new global file
    std::string all_path = base_path + "all.int64";
    int64_t chunk_size = global_shape.z * global_shape.y * global_shape.x;
    FILE *all_file = open_file_write(all_path);
    // TODO handle chunks % disk_block_size != 0
    int64_t *chunk = (int64_t *) aligned_alloc(disk_block_size, chunk_size * sizeof(int64_t));
    for (int64_t i = 0; i < chunks; i++) {
        auto load_start = std::chrono::high_resolution_clock::now();
        load_file(chunk, paths[i], 0, chunk_size);
        auto load_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_load = load_end - load_start;
        if (verbose) {
            std::cout << "load_file: " << (chunk_size*sizeof(int64_t)) / elapsed_load.count() / 1e9 << " GB/s" << std::endl;
        }

        auto apply_start = std::chrono::high_resolution_clock::now();
        apply_renaming(chunk, chunk_size, renames[i]);
        auto apply_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_apply = apply_end - apply_start;
        if (verbose) {
            std::cout << "apply_renaming: " << (chunk_size*sizeof(int64_t)) / elapsed_apply.count() / 1e9 << " GB/s" << std::endl;
        }

        auto store_start = std::chrono::high_resolution_clock::now();
        store_partial(chunk, all_file, i*chunk_size, chunk_size);
        auto store_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_store = store_end - store_start;
        if (verbose) {
            std::cout << "store_partial: " << (chunk_size*sizeof(int64_t)) / elapsed_store.count() / 1e9 << " GB/s" << std::endl;
        }
    }
    free(chunk);
    fclose(all_file);

    auto cc_app_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cc_app = cc_app_end - cc_app_start;
    if (verbose) {
        std::cout << "connected_components lut application: " << elapsed_cc_app.count() << " s" << std::endl;
    }

    return n_labels[0];
}

std::tuple<mapping, mapping> get_mappings(const std::vector<int64_t> &a, const int64_t n_labels_a, const std::vector<int64_t> &b, const int64_t n_labels_b, const idx3d &global_shape) {
    std::vector<mapping> mappings_a;
    std::vector<mapping> mappings_b;

    mapping mapping_a(n_labels_a+1);
    mapping mapping_b(n_labels_b+1);

    #pragma omp parallel num_threads(8)
    {
        int64_t n_threads = omp_get_num_threads();

        #pragma omp single
        {
            mappings_a.resize(n_threads, mapping(n_labels_a+1));
            mappings_b.resize(n_threads, mapping(n_labels_b+1));
        }

        #pragma omp for schedule(static) collapse(2)
        for (int64_t y = 0; y < global_shape.y; y++) {
            for (int64_t x = 0; x < global_shape.x; x++) {
                int64_t i = (y * global_shape.x) + x;
                if (a[i] != 0 && b[i] != 0) {
                    mappings_a[omp_get_thread_num()][a[i]].insert(b[i]);
                    mappings_b[omp_get_thread_num()][b[i]].insert(a[i]);
                }
            }
        }

        for (int64_t i = 0; i < n_threads; i++) {
            #pragma omp for schedule(static)
            for (int64_t j = 1; j < n_labels_a+1; j++) {
                mapping_a[j].insert(mappings_a[i][j].begin(), mappings_a[i][j].end());
            }
            #pragma omp for schedule(static)
            for (int64_t j = 1; j < n_labels_b+1; j++) {
                mapping_b[j].insert(mappings_b[i][j].begin(), mappings_b[i][j].end());
            }
        }
    }

    return { mapping_a, mapping_b };
}

std::vector<int64_t> get_sizes(std::vector<int64_t> &img, int64_t n_labels) {
    std::vector<int64_t> sizes(n_labels, 0);
    for (int64_t i = 0; i < (int64_t) img.size(); i++) {
        sizes[img[i]]++;
    }

    return sizes;
}

std::vector<std::vector<std::tuple<int64_t, int64_t>>> generate_adjacency_tree(const int64_t chunks) {
    int64_t log_chunks = std::ceil(std::log2(chunks));
    std::vector<std::vector<std::tuple<int64_t, int64_t>>> tree(log_chunks);
    for (int64_t layer = 0; layer < log_chunks; layer++) {
        int64_t n_elements = chunks >> (layer+1); // chunks / 2^layer
        int64_t i = 1 << layer; // 1 * 2^layer
        std::vector<std::tuple<int64_t, int64_t>> indices;
        for (int64_t j = i-1; j < i*n_elements*2; j += i*2) {
            indices.push_back({j, j+1});

        }
        tree[layer] = indices;
    }
    return tree;
}

std::vector<idx3d> merge_canonical_names(std::vector<idx3d> &names_a, std::vector<idx3d> &names_b) {
    std::vector<idx3d> names(names_a.size());
    for (int64_t i = 1; i < (int64_t) names_a.size(); i++) {
        if (names_a[i].z == -1) {
            names[i] = names_b[i];
        } else {
            names[i] = names_a[i];
        }
    }

    return names;
}

std::vector<int64_t> merge_labels(mapping &mapping_a, const mapping &mapping_b, const std::vector<int64_t> &to_rename_b) {
    std::list<int64_t> to_check;
    std::vector<int64_t> to_rename_a(mapping_a.size());
    to_rename_a[0] = 0;
    for (int64_t i = 1; i < (int64_t) mapping_a.size(); i++) {
        to_check.push_back(i);
        to_rename_a[i] = i;
    }
    bool updated;
    while (!to_check.empty()) {
        updated = false;
        int64_t label_a = to_check.front();
        std::unordered_set<int64_t> others_a = mapping_a[label_a];
        for (int64_t label_b : others_a) {
            if (label_b < (int64_t) to_rename_b.size()) { // Initially, the mapping will be empty
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

void print_canonical_names(const std::vector<idx3d> &names_a) {
    std::cout << "Canonical names:" << std::endl;
    for (int64_t i = 1; i < (int64_t) names_a.size(); i++) {
        std::cout << i << ": " << names_a[i].z << " " << names_a[i].y << " " << names_a[i].x << std::endl;
    }
    std::cout << "----------------" << std::endl;
}

void print_mapping(const mapping &mapping_) {
    std::cout << "Mapping:" << std::endl;
    for (int64_t i = 1; i < (int64_t) mapping_.size(); i++) {
        std::cout << i << ": { ";
        for (int64_t entry : mapping_[i]) {
            std::cout << entry << " ";
        }
        std::cout << "}" << std::endl;
    }
    std::cout << "----------------" << std::endl;
}

void print_rename(const std::vector<int64_t> &to_rename) {
    std::cout << "Rename:" << std::endl;
    for (int64_t i = 1; i < (int64_t) to_rename.size(); i++) {
        std::cout << i << ": " << to_rename[i] << std::endl;
    }
    std::cout << "----------------" << std::endl;
}

// Ensures that the labels in the renaming LUTs are consecutive
int64_t recount_labels(const mapping &mapping_a, mapping &mapping_b, std::vector<int64_t> &to_rename_a, std::vector<int64_t> &to_rename_b) {
    // We assume that mapping includes 0
    std::vector<int64_t> mapped_a, unmapped_a, unmapped_b;
    int64_t popped_a = 0, popped_b = 0;
    for (int64_t i = 1; i < (int64_t) mapping_a.size(); i++) {
        if (mapping_a[i].size() == 0) {
            unmapped_a.push_back(i);
        } else if (!mapping_a[i].contains(-1)) {
            mapped_a.push_back(i);
        } else {
            popped_a++;
        }
    }
    for (int64_t i = 1; i < (int64_t) mapping_b.size(); i++) {
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
    for (int64_t i = 0; i < (int64_t) mapped_a.size(); i++) {
        new_rename_a[mapped_a[i]] = i+1;
    }
    // Assign the unmapped_a labels to start from mapped_a.size()+1
    for (int64_t i = 0; i < (int64_t) unmapped_a.size(); i++) {
        new_rename_a[unmapped_a[i]] = i+1+mapped_a.size();
    }

    // Apply the new renaming to the renaming LUT
    for (int64_t i = 0; i < (int64_t) to_rename_a.size(); i++) {
        to_rename_a[i] = new_rename_a[to_rename_a[i]];
    }

    // TODO is this actually necessary? We'll see.
    // Update mapping b to use the new a labels
    for (int64_t i = 1; i < (int64_t) mapping_b.size(); i++) {
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
    for (int64_t i = 0; i < (int64_t) mapped_a.size(); i++) {
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
    for (int64_t i = 0; i < (int64_t) unmapped_b.size(); i++) {
        new_rename_b[unmapped_b[i]] = i+1+mapped_a.size()+unmapped_a.size();
    }
    // Apply the new renaming to the renaming LUT
    for (int64_t i = 0; i < (int64_t) to_rename_b.size(); i++) {
        to_rename_b[i] = new_rename_b[to_rename_b[i]];
    }

    return mapped_a.size() + unmapped_a.size() + unmapped_b.size();
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t> relabel(const std::vector<int64_t> &a, const int64_t n_labels_a, const std::vector<int64_t> &b, const int64_t n_labels_b, const idx3d &global_shape, const bool verbose) {
    auto start = std::chrono::high_resolution_clock::now();
    auto [mapping_a, mapping_b] = get_mappings(a, n_labels_a, b, n_labels_b, global_shape);
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

    if (verbose) {
        std::cout << "get_mappings: " << elapsed_get_mappings.count() << " s" << std::endl;
        std::cout << "merge_a: " << elapsed_merge_a.count() << " s" << std::endl;
        std::cout << "merge_b: " << elapsed_merge_b.count() << " s" << std::endl;
        std::cout << "rename_a: " << elapsed_rename_a.count() << " s" << std::endl;
        std::cout << "rename_b: " << elapsed_rename_b.count() << " s" << std::endl;
        std::cout << "recount: " << elapsed_recount.count() << " s" << std::endl;
    }

    return { to_rename_a, to_rename_b, n_new_labels };
}

void rename_mapping(mapping &mapping_a, const std::vector<int64_t> &to_rename_other) {
    for (int64_t i = 1; i < (int64_t) mapping_a.size(); i++) {
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