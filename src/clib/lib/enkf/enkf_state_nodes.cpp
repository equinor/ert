#include <ert/enkf/enkf_state.hpp>
#include <stdio.h>

static bool enkf_state_has_node(const enkf_state_type *enkf_state,
                                const char *node_key) {
    bool has_node = hash_has_key(enkf_state->node_hash, node_key);
    return has_node;
}

static void enkf_state_del_node(enkf_state_type *enkf_state,
                                const char *node_key) {
    if (hash_has_key(enkf_state->node_hash, node_key))
        hash_del(enkf_state->node_hash, node_key);
    else
        fprintf(stderr,
                "%s: tried to remove node:%s which is not in state - internal "
                "error?? \n",
                __func__, node_key);
}

/**
   The enkf_state inserts a reference to the node object. The
   enkf_state object takes ownership of the node object, i.e. it will
   free it when the game is over.

   Observe that if the node already exists the existing node will be
   removed (freed and so on ... ) from the enkf_state object before
   adding the new; this was previously considered a run-time error.
*/
void enkf_state_add_node(enkf_state_type *enkf_state, const char *node_key,
                         const enkf_config_node_type *config) {
    if (enkf_state_has_node(enkf_state, node_key))
        enkf_state_del_node(
            enkf_state,
            node_key); /* Deleting the old instance (if we had one). */
    {
        enkf_node_type *enkf_node = enkf_node_alloc(config);
        hash_insert_hash_owned_ref(enkf_state->node_hash, node_key, enkf_node,
                                   enkf_node_free__);
    }
}
