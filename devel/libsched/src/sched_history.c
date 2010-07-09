#include <util.h>
#include <well_history.h>
#include <sched_history.h>
#include <hash.h>
#include <sched_kw.h>
#include <sched_kw_wconhist.h>
#include <sched_file.h>
#include <sched_types.h>
#include <time_t_vector.h>
#include <size_t_vector.h>
#include <well_index.h>



struct sched_history_struct {
  hash_type           * well_history;   /* Hash table of well_history_type instances. */
  hash_type           * group_history;   
  time_t_vector_type  * time;            
  hash_type           * index;
  char                * sep_string;
};






/*****************************************************************/

well_history_type * sched_history_get_well( sched_history_type * sched_history , const char * well_name );



static void sched_history_install_well_index( sched_history_type * sched_history , well_index_type * well_index , const char ** var_list , const char * well_name) {
  int          index   = 0;
  char       * gen_key = NULL;
  const char * var     = var_list[ index ];
  bool  first          = true;

  while ( var != NULL ) {
    gen_key = util_realloc_sprintf( gen_key , "%s%s%s" , var , sched_history->sep_string , well_name );
    
    if (first) {
      first = false;
      hash_insert_hash_owned_ref( sched_history->index , gen_key , well_index , well_index_free__);
    } else
      hash_insert_ref( sched_history->index , gen_key , well_index );
    
    index++;
    var  = var_list[ index ];
  }
  
  if (first)
    util_abort("%s: internal error - empty var_list \n",__func__);
  free( gen_key );
}


#define VAR_LIST(...) (const char *[]) { __VA_ARGS__ , NULL  }

void sched_history_install_index( sched_history_type * sched_history ) {
  /*1: Installing well based keys like WOPRH. */
  {
    hash_iter_type * well_iter = hash_iter_alloc( sched_history->well_history );
    while (!hash_iter_is_complete( well_iter )) {
      const char * well_name         = hash_iter_get_next_key( well_iter );
      const well_history_type * well = hash_get( sched_history->well_history , well_name );
      
      /* WOPR */
      {
        well_index_type * well_index = well_index_alloc( well , WCONHIST , wconhist_state_iget_WOPRH );
        sched_history_install_well_index( sched_history , well_index , VAR_LIST("WOPR" , "WOPRH") , well_name);
      }
      
      
      /* WGPR */
      {
        well_index_type * well_index = well_index_alloc( well , WCONHIST , wconhist_state_iget_WGPRH );
        sched_history_install_well_index( sched_history , well_index , VAR_LIST("WGPR" , "WGPRH") , well_name);
      }
      
      
      /* WWPR */
      {
        well_index_type * well_index = well_index_alloc( well , WCONHIST , wconhist_state_iget_WWPRH );
        sched_history_install_well_index( sched_history , well_index , VAR_LIST("WWPR" , "WWPRH") , well_name);
      }
      
      
      /* WWCT */
      {
        well_index_type * well_index = well_index_alloc( well , WCONHIST , wconhist_state_iget_WWCTH );
        sched_history_install_well_index( sched_history , well_index , VAR_LIST("WWCT" , "WWCTH") , well_name);
      }

      /* WGOR */
      {
        well_index_type * well_index = well_index_alloc( well , WCONHIST , wconhist_state_iget_WGORH );
        sched_history_install_well_index( sched_history , well_index , VAR_LIST("WGOR" , "WGORH") , well_name);
      }

      /* WGPT */
      {
        well_index_type * well_index = well_index_alloc( well , WCONHIST , wconhist_state_iget_WGPTH );
        sched_history_install_well_index( sched_history , well_index , VAR_LIST("WGPT" , "WGPTH") , well_name);
      }

      /* WOPT */
      {
        well_index_type * well_index = well_index_alloc( well , WCONHIST , wconhist_state_iget_WOPTH );
        sched_history_install_well_index( sched_history , well_index , VAR_LIST("WOPT" , "WOPTH") , well_name);
      }
      
      /* WWPT */
      {
        well_index_type * well_index = well_index_alloc( well , WCONHIST , wconhist_state_iget_WWPTH );
        sched_history_install_well_index( sched_history , well_index , VAR_LIST("WWPT" , "WWPTH") , well_name);
      }
      
    }
    hash_iter_free( well_iter );
  }
}

#undef VAR_LIST


double sched_history_iget( const sched_history_type * sched_history , const char * key , int report_step) {
  well_index_type * index = hash_get( sched_history->index , key );
  
  return well_history_iget( index , report_step );
}





static void sched_history_realloc( sched_history_type * sched_history ) {
  if (sched_history->well_history != NULL)
    hash_free( sched_history->well_history );
  sched_history->well_history = hash_alloc();
  
  if (sched_history->time != NULL)
    time_t_vector_free(sched_history->time);
  sched_history->time         = time_t_vector_alloc( 0 , 0 ); 
}



sched_history_type * sched_history_alloc( const char * sep_string ) {
  sched_history_type * sched_history = util_malloc( sizeof * sched_history , __func__ );

  sched_history->well_history = NULL;
  sched_history->time         = NULL;
  sched_history->index        = hash_alloc();
  sched_history->sep_string   = util_alloc_string_copy( sep_string );
  sched_history_realloc( sched_history );
  
  return sched_history;
}



void sched_history_free( sched_history_type * sched_history ) {
  time_t_vector_free( sched_history->time );
  hash_free( sched_history->well_history );
  hash_free( sched_history->index );
  free( sched_history->sep_string );
  free( sched_history );
}


well_history_type * sched_history_get_well( sched_history_type * sched_history , const char * well_name ) {
  if (!hash_has_key( sched_history->well_history , well_name))
    hash_insert_hash_owned_ref( sched_history->well_history , well_name , well_history_alloc( well_name , sched_history->time ) , well_history_free__);
  
  return hash_get( sched_history->well_history , well_name );
}





void sched_history_update( sched_history_type * sched_history, const sched_file_type * sched_file ) {

  sched_history_realloc( sched_history );
  {
    int block_nr;
    stringlist_type * well_list = stringlist_alloc_new();    
    for (block_nr = 0; block_nr < sched_file_get_num_restart_files( sched_file ); block_nr++) {
      sched_block_type * block = sched_file_iget_block( sched_file , block_nr );
      int kw_nr;
      int report_step = block_nr;

      time_t_vector_iset( sched_history->time , block_nr , sched_file_iget_block_end_time( sched_file , block_nr));
      for (kw_nr = 0; kw_nr < sched_block_get_size( block ); kw_nr++) {
        sched_kw_type * kw         = sched_block_iget_kw( block , kw_nr );
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
              well_history_add_keyword( well_history , kw , report_step);
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
