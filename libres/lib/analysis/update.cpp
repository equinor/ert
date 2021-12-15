#include "ert/analysis/update.hpp"
#include <ert/util/type_vector_functions.h>
#include <ert/res_util/thread_pool.hpp>
#include <ert/res_util/matrix.hpp>
#include <ert/util/int_vector.h>
#include <ert/util/bool_vector.h>
#include <ert/util/hash.hpp>
#include <ert/enkf/local_ministep.hpp>
#include <ert/enkf/local_dataset.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_analysis.hpp>
#include <ert/res_util/res_log.hpp>
#include <ert/util/vector.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/meas_data.hpp>

namespace analysis {

/*
   Helper structs used to pass information to the multithreaded serialize and
   deserialize routines. A serialize_info structure is used at three levels, and
   this also reflects on the degree of change of the various members.

     One ministep: This corresponds to one full execution of the function
         enkf_main_analysis_update().

     One dataset: Each ministep can consist of several local dataset.

     One node: Each local dataset will consist of several nodes.

   The members explicitly marked with a mutable: comment will vary in the
   lifetime of the serialization_info, the other members will be constant. 
*/

typedef struct {
    int row_offset;
    const active_list_type *active_list;
    const char *key;
} serialize_node_info_type;

typedef struct {
    enkf_fs_type *src_fs;
    enkf_fs_type *target_fs;
    const ensemble_config_type *ensemble_config;
    int iens1; /* Inclusive lower limit. */
    int iens2; /* NOT inclusive upper limit. */
    int report_step;
    int target_step;
    matrix_type *A;
    const int_vector_type *iens_active_index;

    std::vector<int>
        active_size; /* mutable: For the serialization of one dataset - many nodes */
    std::vector<int>
        row_offset; /* mutable: For the serialization of one dataset - many nodes */
    serialize_node_info_type
        *node_info; /* mutable: For the serialization of one node */
} serialize_info_type;

int get_active_size(const ensemble_config_type *ensemble_config,
                    enkf_fs_type *fs, const char *key, int report_step,
                    const active_list_type *active_list) {
    const enkf_config_node_type *config_node =
        ensemble_config_get_node(ensemble_config, key);
    /*
     This is very awkward; the problem is that for the GEN_DATA
     type the config object does not really own the size. Instead
     the size is pushed (on load time) from gen_data instances to
     the gen_data_config instance. Therefor we have to assert
     that at least one gen_data instance has been loaded (and
     consequently updated the gen_data_config instance) before we
     query for the size.
  */
    if (enkf_config_node_get_impl_type(config_node) == GEN_DATA) {
        enkf_node_type *node = enkf_node_alloc(config_node);
        node_id_type node_id = {.report_step = report_step, .iens = 0};

        enkf_node_load(node, fs, node_id);
        enkf_node_free(node);
    }
    active_mode_type active_mode = active_list_get_mode(active_list);
    int active_size = 0;
    if (active_mode == INACTIVE)
        active_size = 0;
    else if (active_mode == ALL_ACTIVE)
        active_size = enkf_config_node_get_data_size(config_node, report_step);
    else if (active_mode == PARTLY_ACTIVE)
        active_size = active_list_get_active_size(active_list, -1);
    else
        util_abort("%s: internal error .. \n", __func__);

    return active_size;
}

void serialize_node(enkf_fs_type *fs, const enkf_config_node_type *config_node,
                    int iens, int report_step, int row_offset, int column,
                    const active_list_type *active_list, matrix_type *A) {

    enkf_node_type *node = enkf_node_alloc(config_node);
    node_id_type node_id = {.report_step = report_step, .iens = iens};
    enkf_node_serialize(node, fs, node_id, active_list, A, row_offset, column);
    enkf_node_free(node);
}

void *serialize_nodes_mt(void *arg) {
    serialize_info_type *info = (serialize_info_type *)arg;
    const auto *node_info = info->node_info;
    const enkf_config_node_type *config_node =
        ensemble_config_get_node(info->ensemble_config, node_info->key);
    for (int iens = info->iens1; iens < info->iens2; iens++) {
        int column = int_vector_iget(info->iens_active_index, iens);
        if (column >= 0) {
            serialize_node(info->src_fs, config_node, iens, info->report_step,
                           node_info->row_offset, column,
                           node_info->active_list, info->A);
        }
    }
    return NULL;
}

void serialize_node(const char *node_key, const active_list_type *active_list,
                    int row_offset, thread_pool_type *work_pool,
                    serialize_info_type *serialize_info) {

    /* Multithreaded serializing*/
    const int num_cpu_threads = thread_pool_get_max_running(work_pool);
    serialize_node_info_type node_info[num_cpu_threads];
    thread_pool_restart(work_pool);
    for (int icpu = 0; icpu < num_cpu_threads; icpu++) {
        node_info[icpu].key = node_key;
        node_info[icpu].active_list = active_list;
        node_info[icpu].row_offset = row_offset;
        serialize_info[icpu].node_info = &node_info[icpu];

        thread_pool_add_job(work_pool, serialize_nodes_mt,
                            &serialize_info[icpu]);
    }
    thread_pool_join(work_pool);

    for (int icpu = 0; icpu < num_cpu_threads; icpu++)
        serialize_info[icpu].node_info = nullptr;
}

/*
   The return value is the number of rows in the serialized
   A matrix.
*/
int serialize_dataset(const ensemble_config_type *ens_config,
                      const local_dataset_type *dataset, int report_step,
                      hash_type *use_count, thread_pool_type *work_pool,
                      serialize_info_type *serialize_info) {

    matrix_type *A = serialize_info->A;
    int ens_size = matrix_get_columns(A);
    int current_row = 0;

    const auto &unscaled_keys = local_dataset_unscaled_keys(dataset);
    serialize_info->active_size.resize(unscaled_keys.size());
    serialize_info->row_offset.resize(unscaled_keys.size());
    for (int ikw = 0; ikw < unscaled_keys.size(); ikw++) {
        const auto &key = unscaled_keys[ikw];
        const active_list_type *active_list =
            local_dataset_get_node_active_list(dataset, key.c_str());
        enkf_fs_type *src_fs = serialize_info->src_fs;
        serialize_info->active_size[ikw] = get_active_size(
            ens_config, src_fs, key.c_str(), report_step, active_list);
        serialize_info->row_offset[ikw] = current_row;

        int matrix_rows = matrix_get_rows(A);
        if ((serialize_info->active_size[ikw] + current_row) > matrix_rows)
            matrix_resize(A, matrix_rows + 2 * serialize_info->active_size[ikw],
                          ens_size, true);

        if (serialize_info->active_size[ikw] > 0) {
            serialize_node(key.c_str(), active_list,
                           serialize_info->row_offset[ikw], work_pool,
                           serialize_info);
            current_row += serialize_info->active_size[ikw];
        }
    }
    matrix_shrink_header(A, current_row, ens_size);
    return matrix_get_rows(A);
}

void deserialize_node(enkf_fs_type *target_fs, enkf_fs_type *src_fs,
                      const enkf_config_node_type *config_node, int iens,
                      int target_step, int row_offset, int column,
                      const active_list_type *active_list, matrix_type *A) {

    node_id_type node_id = {.report_step = target_step, .iens = iens};
    enkf_node_type *node = enkf_node_alloc(config_node);

    // If partly active, init node from source fs (deserialize will fill it only in part)
    enkf_node_load(node, src_fs, node_id);

    // deserialize the matrix into the node (and writes it to the target fs)
    enkf_node_deserialize(node, target_fs, node_id, active_list, A, row_offset,
                          column);
    state_map_update_undefined(enkf_fs_get_state_map(target_fs), iens,
                               STATE_INITIALIZED);
    enkf_node_free(node);
}

void *deserialize_nodes_mt(void *arg) {
    serialize_info_type *info = (serialize_info_type *)arg;
    const auto *node_info = info->node_info;
    const enkf_config_node_type *config_node =
        ensemble_config_get_node(info->ensemble_config, node_info->key);
    for (int iens = info->iens1; iens < info->iens2; iens++) {
        int column = int_vector_iget(info->iens_active_index, iens);
        if (column >= 0)
            deserialize_node(info->target_fs, info->src_fs, config_node, iens,
                             info->target_step, node_info->row_offset, column,
                             node_info->active_list, info->A);
    }
    return NULL;
}

void assert_matrix_size(const matrix_type *m, const char *name, int rows,
                        int columns) {
    if (m) {
        if (!matrix_check_dims(m, rows, columns))
            util_abort("%s: matrix mismatch %s:[%d,%d]   - expected:[%d, %d]",
                       __func__, name, matrix_get_rows(m),
                       matrix_get_columns(m), rows, columns);
    } else
        util_abort("%s: matrix:%s is NULL \n", __func__, name);
}

void deserialize_dataset(ensemble_config_type *ensemble_config,
                         const local_dataset_type *dataset, int report_step,
                         serialize_info_type *serialize_info,
                         thread_pool_type *work_pool) {

    const int num_cpu_threads = thread_pool_get_max_running(work_pool);
    const auto &unscaled_keys = local_dataset_unscaled_keys(dataset);
    serialize_info->active_size.resize(unscaled_keys.size());
    serialize_info->row_offset.resize(unscaled_keys.size());
    int current_row = 0;
    for (int ikw = 0; ikw < unscaled_keys.size(); ikw++) {
        const auto &key = unscaled_keys[ikw];
        const active_list_type *active_list =
            local_dataset_get_node_active_list(dataset, key.c_str());
        serialize_info->active_size[ikw] =
            get_active_size(ensemble_config, serialize_info->src_fs,
                            key.c_str(), 0, active_list);
        if (serialize_info->active_size[ikw] > 0) {
            serialize_info->row_offset[ikw] = current_row;
            current_row += serialize_info->active_size[ikw];
            /* Multithreaded */
            serialize_node_info_type node_info[num_cpu_threads];
            thread_pool_restart(work_pool);
            for (int icpu = 0; icpu < num_cpu_threads; icpu++) {
                node_info[icpu].key = key.c_str();
                node_info[icpu].active_list = active_list;
                node_info[icpu].row_offset = serialize_info->row_offset[ikw];
                serialize_info[icpu].node_info = &node_info[icpu];

                thread_pool_add_job(work_pool, deserialize_nodes_mt,
                                    &serialize_info[icpu]);
            }
            thread_pool_join(work_pool);
        }
    }
}

serialize_info_type *
serialize_info_alloc(enkf_fs_type *src_fs, enkf_fs_type *target_fs,
                     const ensemble_config_type *ensemble_config,
                     const int_vector_type *iens_active_index, int target_step,
                     enkf_state_type **ensemble, int report_step,
                     matrix_type *A, int num_cpu_threads) {

    serialize_info_type *serialize_info =
        new serialize_info_type[num_cpu_threads];
    int ens_size = int_vector_size(iens_active_index);
    int iens_offset = 0;
    for (int icpu = 0; icpu < num_cpu_threads; icpu++) {
        int iens_increment =
            (ens_size - iens_offset) / (num_cpu_threads - icpu);
        serialize_info[icpu] = (serialize_info_type){
            .src_fs = src_fs,
            .target_fs = target_fs,
            .ensemble_config = ensemble_config,
            .iens1 = iens_offset,
            .iens2 = iens_offset + iens_increment,
            .report_step = report_step,
            .target_step = target_step,
            .A = A,
            .iens_active_index = iens_active_index,
            .node_info = nullptr,
        };
        iens_offset = serialize_info[icpu].iens2;
    }
    serialize_info[num_cpu_threads - 1].iens2 = ens_size;
    return serialize_info;
}

/*
load a set of parameters from a enkf_fs_type storage into a set of
matrices.
*/
std::unordered_map<std::string, matrix_type *>
load_parameters(enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
                int_vector_type *iens_active_index, int last_step,
                meas_data_type *forecast, enkf_state_type **ensemble,
                hash_type *use_count, obs_data_type *obs_data,
                const local_ministep_type *ministep) {

    int cpu_threads = 4;
    thread_pool_type *tp = thread_pool_alloc(cpu_threads, false);
    int matrix_start_size = 250000;
    int active_ens_size = meas_data_get_active_ens_size(forecast);
    matrix_type *A = matrix_alloc(matrix_start_size, active_ens_size);

    serialize_info_type *serialize_info = serialize_info_alloc(
        target_fs, //src_fs - we have already copied the parameters from the src_fs to the target_fs
        target_fs, ensemble_config, iens_active_index, 0, ensemble, last_step,
        A, cpu_threads);

    std::unordered_map<std::string, matrix_type *> parameters;
    for (auto &[dataset_name, dataset] :
         local_ministep_get_datasets(ministep)) {

        if (!(local_dataset_get_size(dataset)))
            continue;

        const auto &unscaled_keys = local_dataset_unscaled_keys(dataset);

        if (unscaled_keys.size() == 0)
            continue;

        serialize_dataset(ensemble_config, dataset, last_step, use_count, tp,
                          serialize_info);

        parameters[std::string(dataset_name)] =
            matrix_alloc_copy(serialize_info->A);
    }
    delete[] serialize_info;
    matrix_free(A);
    thread_pool_free(tp);
    return parameters;
}

/*
Store a set of parameters into a enkf_fs_type storage
*/
void save_parameters(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    int_vector_type *iens_active_index, int last_step,
    enkf_state_type **ensemble, hash_type *use_count,
    const local_ministep_type *ministep,
    std::unordered_map<std::string, matrix_type *> parameters) {

    int cpu_threads = 4;
    thread_pool_type *tp = thread_pool_alloc(cpu_threads, false);

    for (auto &[dataset_name, dataset] :
         local_ministep_get_datasets(ministep)) {

        if (!(local_dataset_get_size(dataset)))
            continue;

        const auto &unscaled_keys = local_dataset_unscaled_keys(dataset);

        if (unscaled_keys.size() == 0)
            continue;

        matrix_type *A = parameters[std::string(dataset_name)];
        serialize_info_type *serialize_info = serialize_info_alloc(
            target_fs, //src_fs - we have already copied the parameters from the src_fs to the target_fs
            target_fs, ensemble_config, iens_active_index, 0, ensemble,
            last_step, A, cpu_threads);

        deserialize_dataset(ensemble_config, dataset, last_step, serialize_info,
                            tp);
        delete[] serialize_info;
    }
    thread_pool_free(tp);
}

/*
load a set of parameters from a enkf_fs_type storage into a set of
matrices with the corresponding row-scaling object.
*/
std::unordered_map<
    std::string,
    std::vector<std::pair<matrix_type *, const row_scaling_type *>>>
load_row_scaling_parameters(enkf_fs_type *target_fs,
                            ensemble_config_type *ensemble_config,
                            int_vector_type *iens_active_index, int last_step,
                            meas_data_type *forecast,
                            enkf_state_type **ensemble, hash_type *use_count,
                            obs_data_type *obs_data,
                            const local_ministep_type *ministep) {

    int matrix_start_size = 250000;
    int active_ens_size = meas_data_get_active_ens_size(forecast);
    matrix_type *A = matrix_alloc(matrix_start_size, active_ens_size);

    std::unordered_map<
        std::string,
        std::vector<std::pair<matrix_type *, const row_scaling_type *>>>
        parameters;
    for (auto &[dataset_name, dataset] :
         local_ministep_get_datasets(ministep)) {

        if (!(local_dataset_get_size(dataset)))
            continue;

        const auto &scaled_keys = local_dataset_scaled_keys(dataset);

        if (scaled_keys.size() == 0)
            continue;

        std::vector<std::pair<matrix_type *, const row_scaling_type *>>
            row_scaling_list;
        for (int ikw = 0; ikw < scaled_keys.size(); ikw++) {
            const auto &key = scaled_keys[ikw];
            const active_list_type *active_list =
                local_dataset_get_node_active_list(dataset, key.c_str());
            const auto *config_node =
                ensemble_config_get_node(ensemble_config, key.c_str());
            const int node_size =
                enkf_config_node_get_data_size(config_node, last_step);
            if (matrix_get_rows(A) < node_size)
                matrix_resize(A, node_size, active_ens_size, false);

            for (int iens = 0; iens < int_vector_size(iens_active_index);
                 iens++) {
                int column = int_vector_iget(iens_active_index, iens);
                if (column >= 0) {
                    serialize_node(target_fs, config_node, iens, last_step, 0,
                                   column, active_list, A);
                }
            }
            const row_scaling_type *row_scaling =
                local_dataset_get_row_scaling(dataset, key.c_str());
            matrix_shrink_header(A, row_scaling->size(), matrix_get_columns(A));
            row_scaling_list.push_back(
                std::pair<matrix_type *, const row_scaling_type *>(
                    matrix_alloc_copy(A), row_scaling));
        }
        parameters[std::string(dataset_name)] = row_scaling_list;
    }
    matrix_free(A);
    return parameters;
}

/*
Store a set of row-scaled parameters into a enkf_fs_type storage
*/
void save_row_scaling_parameters(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    int_vector_type *iens_active_index, int last_step,
    const local_ministep_type *ministep,
    std::unordered_map<
        std::string,
        std::vector<std::pair<matrix_type *, const row_scaling_type *>>>
        parameters) {

    for (auto &[dataset_name, dataset] :
         local_ministep_get_datasets(ministep)) {

        if (!(local_dataset_get_size(dataset)))
            continue;

        const auto &scaled_keys = local_dataset_scaled_keys(dataset);

        if (scaled_keys.size() == 0)
            continue;

        std::vector<std::pair<matrix_type *, const row_scaling_type *>>
            row_scaling_list = parameters[dataset_name];
        for (int ikw = 0; ikw < scaled_keys.size(); ikw++) {
            const auto &key = scaled_keys[ikw];
            const active_list_type *active_list =
                local_dataset_get_node_active_list(dataset, key.c_str());
            matrix_type *A = row_scaling_list[ikw].first;
            for (int iens = 0; iens < int_vector_size(iens_active_index);
                 iens++) {
                int column = int_vector_iget(iens_active_index, iens);
                if (column >= 0) {
                    deserialize_node(
                        target_fs, target_fs,
                        ensemble_config_get_node(ensemble_config, key.c_str()),
                        iens, 0, 0, column, active_list, A);
                }
            }
        }
    }
}

/*
Run the update algorithm on a set of A matrices without row-scaling
*/
void run_analysis_update(
    analysis_module_type *module, const bool_vector_type *ens_mask,
    const meas_data_type *forecast, obs_data_type *obs_data,
    rng_type *shared_rng, matrix_type *E,
    std::unordered_map<std::string, matrix_type *> parameters) {
    if (parameters.size() == 0)
        return;
    const int cpu_threads = 4;
    thread_pool_type *tp = thread_pool_alloc(cpu_threads, false);
    int active_ens_size = meas_data_get_active_ens_size(forecast);
    int active_obs_size = obs_data_get_active_size(obs_data);
    matrix_type *X = matrix_alloc(active_ens_size, active_ens_size);
    matrix_type *S = meas_data_allocS(forecast);
    assert_matrix_size(S, "S", active_obs_size, active_ens_size);
    matrix_type *R = obs_data_allocR(obs_data);
    assert_matrix_size(R, "R", active_obs_size, active_obs_size);
    matrix_type *dObs = obs_data_allocdObs(obs_data);

    matrix_type *D = NULL;
    const bool_vector_type *obs_mask = obs_data_get_active_mask(obs_data);

    if (analysis_module_check_option(module, ANALYSIS_NEED_ED)) {

        D = obs_data_allocD(obs_data, E, S);

        assert_matrix_size(E, "E", active_obs_size, active_ens_size);
        assert_matrix_size(D, "D", active_obs_size, active_ens_size);
    }

    if (analysis_module_check_option(module, ANALYSIS_SCALE_DATA))
        obs_data_scale(obs_data, S, E, D, R, dObs);

    if (!(analysis_module_check_option(module, ANALYSIS_USE_A) ||
          analysis_module_check_option(module, ANALYSIS_UPDATE_A)))
        analysis_module_initX(module, X, NULL, S, R, dObs, E, D, shared_rng);

    analysis_module_init_update(module, ens_mask, obs_mask, S, R, dObs, E, D,
                                shared_rng);

    for (const auto &[dataset_name, A] : parameters) {
        if (analysis_module_check_option(module, ANALYSIS_UPDATE_A)) {
            analysis_module_updateA(module, A, S, R, dObs, E, D, shared_rng);
        } else {
            if (analysis_module_check_option(module, ANALYSIS_USE_A)) {
                analysis_module_initX(module, X, A, S, R, dObs, E, D,
                                      shared_rng);
            }
            matrix_inplace_matmul_mt2(A, X, tp);
        }
    }

    matrix_safe_free(D);
    matrix_free(S);
    matrix_free(R);
    matrix_free(dObs);
    matrix_free(X);
    thread_pool_free(tp);
}

/*
Run the row-scaling enabled update algorithm on a set of A matrices.
*/
void run_analysis_update_with_rowscaling(
    analysis_module_type *module, const bool_vector_type *ens_mask,
    const meas_data_type *forecast, obs_data_type *obs_data,
    rng_type *shared_rng, matrix_type *E,
    std::unordered_map<
        std::string,
        std::vector<std::pair<matrix_type *, const row_scaling_type *>>>
        parameters) {

    if (parameters.size() == 0)
        return;
    int active_ens_size = meas_data_get_active_ens_size(forecast);
    int active_obs_size = obs_data_get_active_size(obs_data);
    matrix_type *X = matrix_alloc(active_ens_size, active_ens_size);
    matrix_type *S = meas_data_allocS(forecast);
    assert_matrix_size(S, "S", active_obs_size, active_ens_size);

    matrix_type *R = obs_data_allocR(obs_data);
    assert_matrix_size(R, "R", active_obs_size, active_obs_size);

    matrix_type *dObs = obs_data_allocdObs(obs_data);

    matrix_type *D = NULL;
    const bool_vector_type *obs_mask = obs_data_get_active_mask(obs_data);

    if (analysis_module_check_option(module, ANALYSIS_NEED_ED)) {

        D = obs_data_allocD(obs_data, E, S);

        assert_matrix_size(E, "E", active_obs_size, active_ens_size);
        assert_matrix_size(D, "D", active_obs_size, active_ens_size);
    }

    if (analysis_module_check_option(module, ANALYSIS_SCALE_DATA))
        obs_data_scale(obs_data, S, E, D, R, dObs);

    if (!analysis_module_check_option(module, ANALYSIS_USE_A))
        analysis_module_initX(module, X, NULL, S, R, dObs, E, D, shared_rng);

    if (analysis_module_check_option(module, ANALYSIS_UPDATE_A))
        throw std::logic_error("Sorry - row scaling for distance based "
                               "localization can not be combined with "
                               "analysis modules which update the A matrix");

    analysis_module_init_update(module, ens_mask, obs_mask, S, R, dObs, E, D,
                                shared_rng);

    for (const auto &[dataset_name, row_scaling_list] : parameters) {
        for (const auto &row_scaling_A_pair : row_scaling_list) {
            matrix_type *A = row_scaling_A_pair.first;
            const row_scaling_type *row_scaling = row_scaling_A_pair.second;

            if (analysis_module_check_option(module, ANALYSIS_USE_A))
                analysis_module_initX(module, X, A, S, R, dObs, E, D,
                                      shared_rng);

            row_scaling_multiply(row_scaling, A, X);
        }
    }

    matrix_safe_free(D);
    matrix_free(S);
    matrix_free(R);
    matrix_free(dObs);
    matrix_free(X);
}

/*
Check whether the current state and config allows the update algorithm
to be executed
*/
bool assert_update_viable(const analysis_config_type *analysis_config,
                          const enkf_fs_type *source_fs,
                          const int total_ens_size,
                          const local_updatestep_type *updatestep) {
    state_map_type *source_state_map = enkf_fs_get_state_map(source_fs);
    const int active_ens_size =
        state_map_count_matching(source_state_map, STATE_HAS_DATA);

    if (!analysis_config_have_enough_realisations(
            analysis_config, active_ens_size, total_ens_size)) {
        fprintf(stderr,
                "** ERROR ** There are %d active realisations left, which is "
                "less than the minimum specified - stopping assimilation.\n",
                active_ens_size);
        return false;
    }

    // exit if multi step update
    if ((local_updatestep_get_num_ministep(updatestep) > 1) &&
        (analysis_config_get_module_option(analysis_config,
                                           ANALYSIS_ITERABLE))) {
        util_exit("** ERROR: Can not combine iterable modules with multi step "
                  "updates - sorry\n");
    }
    return true;
}

/*
Copy all parameters from source_fs to target_fs
*/
void copy_parameters(enkf_fs_type *source_fs, enkf_fs_type *target_fs,
                     const ensemble_config_type *ensemble_config,
                     const int total_ens_size,
                     const int_vector_type *ens_active_list) {

    /*
      Copy all the parameter nodes from source case to target case;
      nodes which are updated will be fetched from the new target
      case, and nodes which are not updated will be manually copied
      over there.
    */

    if (target_fs != source_fs) {
        stringlist_type *param_keys =
            ensemble_config_alloc_keylist_from_var_type(ensemble_config,
                                                        PARAMETER);
        for (int i = 0; i < stringlist_get_size(param_keys); i++) {
            const char *key = stringlist_iget(param_keys, i);
            enkf_config_node_type *config_node =
                ensemble_config_get_node(ensemble_config, key);
            enkf_node_type *data_node = enkf_node_alloc(config_node);
            for (int j = 0; j < int_vector_size(ens_active_list); j++) {
                node_id_type node_id;
                node_id.iens = int_vector_iget(ens_active_list, j);
                node_id.report_step = 0;

                enkf_node_load(data_node, source_fs, node_id);
                enkf_node_store(data_node, target_fs, node_id);
            }
            enkf_node_free(data_node);
        }
        stringlist_free(param_keys);
    }
}
/*
   This function returns a (enkf_node_type ** ) pointer, which points
   to all the instances with the same keyword, i.e.

   alloc_node_ensemble(enkf_main , "PRESSURE");

   Will return an ensemble of pressure nodes. The enkf_node instances
   must be free'd with enkf_node_free( ) afterwards.

   Example:

   vector_type * pressure_nodes = alloc_node_ensemble(enkf_main , "PRESSURE");

   // Do something with the pressure nodes ...

   free(pressure_nodes);

*/

vector_type *alloc_node_ensemble(const ensemble_config_type *ens_config,
                                 const int ens_size, enkf_fs_type *src_fs,
                                 const char *key, int report_step) {
    vector_type *node_ensemble = vector_alloc_new();
    const enkf_config_node_type *config_node =
        ensemble_config_get_node(ens_config, key);
    node_id_type node_id = {.report_step = report_step, .iens = -1};

    for (int iens = 0; iens < ens_size; iens++) {
        enkf_node_type *node = enkf_node_alloc(config_node);
        node_id.iens = iens;
        enkf_node_load(node, src_fs, node_id);
        vector_append_owned_ref(node_ensemble, node, enkf_node_free__);
    }

    return node_ensemble;
}

/*
   This function calculates the node standard deviation from the
   ensemble. The mean can be NULL, in which case it is assumed that
   the mean has already been shifted away from the ensemble.
*/
void node_std(const vector_type *ensemble, const enkf_node_type *mean,
              enkf_node_type *std) {
    if (vector_get_size(ensemble) == 0)
        util_abort("%s: internal error - calculation std of empty list\n",
                   __func__);
    {
        int iens;
        enkf_node_clear(std);
        for (iens = 0; iens < vector_get_size(ensemble); iens++)
            enkf_node_iaddsqr(
                std, (const enkf_node_type *)vector_iget_const(ensemble, iens));
        enkf_node_scale(std, 1.0 / vector_get_size(ensemble));

        if (mean != NULL) {
            enkf_node_scale(std, -1);
            enkf_node_iaddsqr(std, mean);
            enkf_node_scale(std, -1);
        }

        enkf_node_sqrt(std);
    }
}

void node_mean(const vector_type *ensemble, enkf_node_type *mean) {
    if (vector_get_size(ensemble) == 0)
        util_abort("%s: internal error - calculation average of empty list\n",
                   __func__);
    enkf_node_clear(mean);
    for (int iens = 0; iens < vector_get_size(ensemble); iens++)
        enkf_node_iadd(
            mean, (const enkf_node_type *)vector_iget_const(ensemble, iens));

    enkf_node_scale(mean, 1.0 / vector_get_size(ensemble));
}

void inflate_node(const ensemble_config_type *ens_config, enkf_fs_type *src_fs,
                  enkf_fs_type *target_fs, const int ens_size, int report_step,
                  const char *key, const enkf_node_type *min_std) {
    vector_type *ensemble = alloc_node_ensemble(
        ens_config, ens_size, src_fs, key, report_step); // Was ANALYZED
    enkf_node_type *mean =
        enkf_node_copyc((const enkf_node_type *)vector_iget_const(ensemble, 0));
    enkf_node_type *std = enkf_node_copyc(mean);

    /* Shifting away the mean */
    node_mean(ensemble, mean);
    node_std(ensemble, mean, std);
    enkf_node_scale(mean, -1);
    for (int iens = 0; iens < ens_size; iens++)
        enkf_node_iadd((enkf_node_type *)vector_iget(ensemble, iens), mean);

    enkf_node_scale(mean, -1);

    /*
     * Now we have the ensemble represented as a mean and an ensemble of
     * deviations from the mean. This is the form suitable for actually
     * doing the inflation.
     */
    enkf_node_type *inflation = enkf_node_copyc(mean);
    enkf_node_set_inflation(inflation, std, min_std);

    for (int iens = 0; iens < vector_get_size(ensemble); iens++)
        enkf_node_imul((enkf_node_type *)vector_iget(ensemble, iens),
                       inflation);

    enkf_node_free(inflation);

    /* Add the mean back in - and store the updated node to disk.*/
    for (int iens = 0; iens < ens_size; iens++) {
        node_id_type node_id = {.report_step = report_step, .iens = iens};
        enkf_node_iadd((enkf_node_type *)vector_iget(ensemble, iens), mean);
        enkf_node_store((enkf_node_type *)vector_iget(ensemble, iens),
                        target_fs, node_id);
    }

    enkf_node_free(mean);
    enkf_node_free(std);
    vector_free(ensemble);
}

void inflate(const ensemble_config_type *ens_config, enkf_fs_type *src_fs,
             enkf_fs_type *target_fs, const int ens_size, int report_step,
             hash_type *use_count) {

    stringlist_type *keys =
        ensemble_config_alloc_keylist_from_var_type(ens_config, PARAMETER);

    for (int ikey = 0; ikey < stringlist_get_size(keys); ikey++) {
        const char *key = stringlist_iget(keys, ikey);
        if (hash_get_counter(use_count, key) > 0) {
            const enkf_config_node_type *config_node =
                ensemble_config_get_node(ens_config, key);
            const enkf_node_type *min_std =
                enkf_config_node_get_min_std(config_node);

            if (min_std != NULL)
                inflate_node(ens_config, src_fs, target_fs, ens_size,
                             report_step, key, min_std);
        }
    }
    stringlist_free(keys);
}

void assert_size_equal(int ens_size, const bool_vector_type *ens_mask) {
    if (bool_vector_size(ens_mask) != ens_size)
        util_abort("%s: fundamental inconsistency detected. Total ens_size:%d  "
                   "mask_size:%d \n",
                   __func__, ens_size, bool_vector_size(ens_mask));
}

bool smoother_update(std::vector<int> step_list,
                     const local_updatestep_type *updatestep,
                     int total_ens_size, enkf_obs_type *obs,
                     rng_type *shared_rng,
                     const analysis_config_type *analysis_config,
                     ensemble_config_type *ensemble_config,
                     enkf_state_type **ensemble, enkf_fs_type *source_fs,
                     enkf_fs_type *target_fs, FILE *log_stream, bool verbose) {
    if (!assert_update_viable(analysis_config, source_fs, total_ens_size,
                              updatestep))
        return false;

    /*
    Observations and measurements are collected in these temporary
    structures. obs_data is a precursor for the 'd' vector, and
    meas_data is a precursor for the 'S' matrix'.

    The reason for going via these temporary structures is to support
    deactivating observations which should not be used in the update
    process.
    */
    bool_vector_type *ens_mask = bool_vector_alloc(total_ens_size, false);
    state_map_type *source_state_map = enkf_fs_get_state_map(source_fs);

    state_map_select_matching(source_state_map, ens_mask, STATE_HAS_DATA, true);
    double global_std_scaling =
        analysis_config_get_global_std_scaling(analysis_config);
    meas_data_type *meas_data = meas_data_alloc(ens_mask);
    int_vector_type *ens_active_list = bool_vector_alloc_active_list(ens_mask);

    copy_parameters(source_fs, target_fs, ensemble_config, total_ens_size,
                    ens_active_list);

    hash_type *use_count = hash_alloc();
    int current_step = step_list.back();

    /* Looping over local analysis ministep */
    for (int ministep_nr = 0;
         ministep_nr < local_updatestep_get_num_ministep(updatestep);
         ministep_nr++) {
        local_ministep_type *ministep =
            local_updatestep_iget_ministep(updatestep, ministep_nr);
        local_obsdata_type *obsdata = local_ministep_get_obsdata(ministep);

        obs_data_type *obs_data = obs_data_alloc(global_std_scaling);
        obs_data_reset(obs_data);
        meas_data_reset(meas_data);

        /*
        Temporarily we will just force the timestep from the input
        argument onto the obsdata instance; in the future the
        obsdata should hold it's own here.
        */
        local_obsdata_reset_tstep_list(obsdata, step_list);

        double alpha = analysis_config_get_alpha(analysis_config);
        double std_cutoff = analysis_config_get_std_cutoff(analysis_config);
        enkf_obs_get_obs_and_measure_data(obs, source_fs, obsdata,
                                          ens_active_list, meas_data, obs_data);

        enkf_analysis_deactivate_outliers(obs_data, meas_data, std_cutoff,
                                          alpha, verbose);
        local_ministep_add_obs_data(ministep, obs_data);

        enkf_analysis_fprintf_obs_summary(obs_data, meas_data, step_list,
                                          local_ministep_get_name(ministep),
                                          log_stream);

        if ((obs_data_get_active_size(obs_data) > 0) &&
            (meas_data_get_active_obs_size(meas_data) > 0)) {
            int_vector_type *iens_active_index =
                bool_vector_alloc_active_index_list(ens_mask, -1);

            /*
            The update for one local_dataset instance consists of two main chunks:

            1. The first chunk updates all the parameters which don't have row
                scaling attached. These parameters are serialized together to the A
                matrix and all the parameters are updated in one go.

            2. The second chunk is loop over all the parameters which have row
                scaling attached. These parameters are updated one at a time.
            */

            analysis_module_type *module =
                analysis_config_get_active_module(analysis_config);
            if (local_ministep_has_analysis_module(ministep))
                module = local_ministep_get_analysis_module(ministep);
            assert_size_equal(total_ens_size, ens_mask);
            // E matrix is generated with shared rng, thus only creating it once for identical results
            int active_ens_size = meas_data_get_active_ens_size(meas_data);
            matrix_type *E =
                obs_data_allocE(obs_data, shared_rng, active_ens_size);

            // Part 1: Parameters which do not have row scaling attached.
            auto parameters = load_parameters(
                target_fs, ensemble_config, iens_active_index, current_step,
                meas_data, ensemble, use_count, obs_data, ministep);
            run_analysis_update(module, ens_mask, meas_data, obs_data,
                                shared_rng, E, parameters);
            save_parameters(target_fs, ensemble_config, iens_active_index,
                            current_step, ensemble, use_count, ministep,
                            parameters);
            for (auto &[_, A] : parameters)
                matrix_free(A);

            // Part 2: Parameters which do have row scaling attached.
            auto row_scaling_parameters = load_row_scaling_parameters(
                target_fs, ensemble_config, iens_active_index, current_step,
                meas_data, ensemble, use_count, obs_data, ministep);
            run_analysis_update_with_rowscaling(module, ens_mask, meas_data,
                                                obs_data, shared_rng, E,
                                                row_scaling_parameters);

            save_row_scaling_parameters(target_fs, ensemble_config,
                                        iens_active_index, current_step,
                                        ministep, row_scaling_parameters);
            for (auto &[_, row_scale_A_list] : row_scaling_parameters)
                for (auto [A, _] : row_scale_A_list)
                    matrix_free(A);

            matrix_safe_free(E);

            analysis_module_complete_update(module);

        } else if (target_fs != source_fs)
            res_log_ferror(
                "No active observations/parameters for MINISTEP: %s.",
                local_ministep_get_name(ministep));
    }
    inflate(ensemble_config, source_fs, target_fs, total_ens_size, current_step,
            use_count);
    hash_free(use_count);

    state_map_type *target_state_map = enkf_fs_get_state_map(target_fs);

    if (target_state_map != source_state_map) {
        state_map_set_from_inverted_mask(target_state_map, ens_mask,
                                         STATE_PARENT_FAILURE);
        state_map_set_from_mask(target_state_map, ens_mask, STATE_INITIALIZED);
        enkf_fs_fsync(target_fs);
    }

    int_vector_free(ens_active_list);
    meas_data_free(meas_data);
    bool_vector_free(ens_mask);

    return true;
}
} // namespace analysis
