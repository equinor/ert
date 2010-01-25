#include <util.h>
#include <well_history.h>
#include <sched_history.h>
#include <hash.h>
#include <sched_kw.h>
#include <sched_file.h>
#include <time_t_vector.h>



struct sched_history_struct {
  hash_type           * well_history;   /* Hash table of well_history_type instances. */
  time_t_vector_type  * time;            
};



static void sched_history_realloc( sched_history_type * sched_history ) {
  if (sched_history->well_history != NULL)
    hash_free( sched_history->well_history );
  sched_history->well_history = hash_alloc();


  if (sched_history->time != NULL)
    time_t_vector_free(sched_history->time);
  sched_history->time         = time_t_vector_alloc( 0 , 0 ); 
}



sched_history_type * sched_history_alloc( ) {
  sched_history_type * sched_history = util_malloc( sizeof * sched_history , __func__ );

  sched_history->well_history = NULL;
  sched_history->time         = NULL;
  sched_history_realloc( sched_history );
  
  return sched_history;
}



void sched_history_free( sched_history_type * sched_history ) {
  time_t_vector_free( sched_history->time );
  hash_free( sched_history->well_history );
  free( sched_history );
}


well_history_type * sched_history_get_well( sched_history_type * sched_history , const char * well_name ) {
  if (!hash_has_key( sched_history->well_history , well_name))
    hash_insert_hash_owned_ref( sched_history->well_history , well_name , well_history_alloc( well_name ) , well_history_free__);
  
  return hash_get( sched_history->well_history , well_name );
}





void sched_history_update( sched_history_type * sched_history, const sched_file_type * sched_file ) {

  sched_history_realloc( sched_history );
  {
    int block_nr;
    stringlist_type * well_list = stringlist_alloc_new();    
    for (block_nr = 0; block_nr < sched_file_get_num_restart_files( sched_file ); block_nr++) {
      const sched_block_type * block = sched_file_iget_block( sched_file , block_nr );
      int kw_nr;
      int report_step = block_nr;

      for (kw_nr = 0; kw_nr < sched_block_get_size( block ); kw_nr++) {
        const sched_kw_type * kw = sched_block_iget_kw( block , kw_nr );
        sched_kw_type_enum kw_type = sched_kw_get_type( kw );
        switch( kw_type ) {
        case(WCONHIST):
          {
            const sched_kw_wconhist_type * wconhist = sched_kw_get_data( kw );
            sched_kw_wconhist_init_well_list( wconhist , well_list );
            int iw;
            for (iw = 0; iw < stringlist_get_size( well_list ); iw++) {
              const char * well_name           = stringlist_iget( well_list , iw );
              well_history_type * well_history = sched_history_get_well( sched_history , well_name );
              
              sched_kw_wconhist_update_state( kw , well_history_get_wconhist( well_history ) , well_name , report_step );
            }
          }
          break;
        case(WCONINJH):
          /* ... */
          break;
        default:
          /*   */
          break;
        }
      }
    }
    stringlist_free( well_list );
  }
}
