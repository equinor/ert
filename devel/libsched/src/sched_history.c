#include <util.h>
#include <well_history.h>
#include <group_history.h>
#include <sched_history.h>
#include <hash.h>
#include <sched_kw.h>
#include <sched_kw_wconhist.h>
#include <sched_kw_welspecs.h>
#include <sched_kw_wconinje.h>
#include <sched_kw_wconinjh.h>
#include <sched_kw_gruptree.h>
#include <sched_file.h>
#include <sched_types.h>
#include <time_t_vector.h>
#include <size_t_vector.h>
#include <well_index.h>
#include <group_index.h>


struct sched_history_struct {
  hash_type           * well_history;   /* Hash table of well_history_type instances. */
  hash_type           * group_history;   
  time_t_vector_type  * time;            
  hash_type           * index;
  char                * sep_string;
};


#define FIELD_GROUP           "FIELD"





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



static void sched_history_install_group_index( sched_history_type * sched_history , group_index_type * group_index , const char ** var_list , const char * group_name) {
  int          index   = 0;
  char       * gen_key = NULL;
  const char * var     = var_list[ index ];
  bool  first          = true;

  while ( var != NULL ) {
    gen_key = util_realloc_sprintf( gen_key , "%s%s%s" , var , sched_history->sep_string , group_name );
    
    if (first) {
      first = false;
      hash_insert_hash_owned_ref( sched_history->index , gen_key , group_index , group_index_free__);
    } else
      hash_insert_ref( sched_history->index , gen_key , group_index );
    
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

      /* WWIRH - this can be got from _either_ the WCONINJH keyowrord
         or the WCONINJE keyword (provided the latter is in rate
         controlled mode. ) */
      {
        well_index_type * well_index = well_index_alloc( well , WCONINJH , wconinjh_state_iget_WWIRH );   /* The first type */
        well_index_add_type( well_index , WCONINJE , wconinje_state_iget_WWIRH );                         /* The second type */
        sched_history_install_well_index( sched_history , well_index , VAR_LIST("WWIRH" , "WWIR") , well_name);
      }
    }
    hash_iter_free( well_iter );
  }



  /*2: Installing group based indices */
  {
    hash_iter_type * group_iter = hash_iter_alloc( sched_history->group_history );
    while (!hash_iter_is_complete( group_iter )) {
      const char * group_name          = hash_iter_get_next_key( group_iter );
      const group_history_type * group = hash_get( sched_history->group_history , group_name );
      
      /* GOPR */
      {
        group_index_type * group_index = group_index_alloc( group , group_history_iget_GOPRH );
        sched_history_install_group_index( sched_history , group_index , VAR_LIST("GOPR" , "GOPRH") , group_name);
      }

      /* GGPR */
      {
        group_index_type * group_index = group_index_alloc( group , group_history_iget_GGPRH );
        sched_history_install_group_index( sched_history , group_index , VAR_LIST("GGPR" , "GGPRH") , group_name);
      }

      /* GWPR */
      {
        group_index_type * group_index = group_index_alloc( group , group_history_iget_GWPRH );
        sched_history_install_group_index( sched_history , group_index , VAR_LIST("GWPR" , "GWPRH") , group_name);
      }
      
    }
    hash_iter_free( group_iter );
  }


  /*3: Installing field based indices (which is just an alias to the FIELD group); */
  {
    const group_history_type * group = hash_get( sched_history->group_history , FIELD_GROUP );
    
    /* FWPRH */
    {
      group_index_type * group_index   = group_index_alloc( group , group_history_iget_GWPRH );
      hash_insert_hash_owned_ref( sched_history->index , "FWPRH" , group_index , group_index_free__ );
      hash_insert_ref( sched_history->index , "FWPR" , group_index);
    }

    /* FOPRH */
    {
      group_index_type * group_index   = group_index_alloc( group , group_history_iget_GOPRH );
      hash_insert_hash_owned_ref( sched_history->index , "FOPRH" , group_index , group_index_free__ );
      hash_insert_ref( sched_history->index , "FOPR" , group_index);
    }

    /* FGPRH */
    {
      group_index_type * group_index   = group_index_alloc( group , group_history_iget_GGPRH );
      hash_insert_hash_owned_ref( sched_history->index , "FGPRH" , group_index , group_index_free__ );
      hash_insert_ref( sched_history->index , "FGPR" , group_index);
    }


  }
      


}
#undef VAR_LIST




double sched_history_iget( const sched_history_type * sched_history , const char * key , int report_step) {
  void * index = hash_get( sched_history->index , key );

  if (well_index_is_instance( index ))
    return well_history_iget( index , report_step );
  else
    return group_history_iget( index , report_step );
}





static void sched_history_realloc( sched_history_type * sched_history ) {
  if (sched_history->well_history != NULL)
    hash_free( sched_history->well_history );
  sched_history->well_history = hash_alloc();
  
  if (sched_history->group_history != NULL)
    hash_free( sched_history->group_history );
  sched_history->group_history = hash_alloc();
  
  

  if (sched_history->time != NULL)
    time_t_vector_free(sched_history->time);
  sched_history->time         = time_t_vector_alloc( 0 , 0 ); 
}



sched_history_type * sched_history_alloc( const char * sep_string ) {
  sched_history_type * sched_history = util_malloc( sizeof * sched_history , __func__ );

  sched_history->well_history  = NULL;
  sched_history->group_history = NULL;
  sched_history->time         = NULL;
  sched_history->index        = hash_alloc();
  sched_history->sep_string   = util_alloc_string_copy( sep_string );
  sched_history_realloc( sched_history );
  
  return sched_history;
}



void sched_history_free( sched_history_type * sched_history ) {
  time_t_vector_free( sched_history->time );
  hash_free( sched_history->well_history );
  hash_free( sched_history->group_history );
  hash_free( sched_history->index );
  free( sched_history->sep_string );
  free( sched_history );
}


well_history_type * sched_history_get_well( sched_history_type * sched_history , const char * well_name ) {
  return hash_get( sched_history->well_history , well_name );
}


group_history_type * sched_history_get_group( sched_history_type * sched_history , const char * group_name ) {
  return hash_get( sched_history->group_history , group_name );
}



static void sched_history_add_wells( sched_history_type * sched_history , const sched_kw_welspecs_type * welspecs , const stringlist_type * wells) {
  for (int iw = 0; iw < stringlist_get_size( wells ); iw++) {
    const char * well = stringlist_iget( wells , iw );
    if (!hash_has_key( sched_history->well_history , well)) 
      hash_insert_hash_owned_ref( sched_history->well_history , well , well_history_alloc( well , sched_history->time ), well_history_free__ );
    
    /* Could possibly extract more information from the welspecs
       keyword and update well_history object here, but it does not
       seem to contain any more interesting info???
    */
    
  }
}


static void sched_history_add_group( sched_history_type * sched_history , group_history_type * new_group, group_history_type * parent_group , int report_step ) {
  hash_insert_hash_owned_ref( sched_history->group_history , group_history_get_name( new_group ) , new_group , group_history_free__ );
  if (parent_group == NULL)
    parent_group = sched_history_get_group( sched_history , FIELD_GROUP );
  
  group_history_add_child( parent_group , new_group , group_history_get_name( new_group ) , report_step );
}


/**
   Because the FIELD group is added without any parent; it does not
   use the standard sched_history_group_add() function. */

static void sched_history_add_FIELD_group( sched_history_type * sched_history ) {
  group_history_type * field_group = group_history_alloc( FIELD_GROUP , sched_history->time );
  hash_insert_hash_owned_ref( sched_history->group_history , FIELD_GROUP , field_group , group_history_free__ );
}


void sched_history_fprintf_group_structure( sched_history_type * sched_history , int report_step ) {
  group_history_type * field_group = sched_history_get_group( sched_history , FIELD_GROUP );
  group_history_fprintf( field_group , report_step , true ,stdout );
}


static void sched_history_add_groups_gruptree( sched_history_type * sched_history , const sched_kw_gruptree_type * gruptree , int report_step , const stringlist_type * child_groups , const stringlist_type * parent_groups) {
  for (int i = 0; i < stringlist_get_size( child_groups ); i++) {
    const char * parent_group_name  = stringlist_iget( parent_groups , i );
    const char * child_group_name   = stringlist_iget( child_groups , i );
    group_history_type * parent_group;
    group_history_type * child_group;


    if (!hash_has_key( sched_history->group_history , parent_group_name )) 
      sched_history_add_group( sched_history , group_history_alloc( parent_group_name , sched_history->time ) , NULL , report_step );
    parent_group = sched_history_get_group( sched_history , parent_group_name );

    if (!hash_has_key( sched_history->group_history , child_group_name )) 
      sched_history_add_group( sched_history , group_history_alloc( child_group_name , sched_history->time ) , parent_group , report_step );
    child_group = sched_history_get_group( sched_history , child_group_name );

    group_history_add_child( parent_group , child_group , child_group_name , report_step);
  }
}


/**
   When new wells are added with the WELSPECS keyword their parent
   group is implicitly introduced as the second argument of the
   WELSPEC keyword, in addition the GRUPTREE keyword will also
   implicitly introduce groups. 

   This functions creates group_history objects for all the groups
   introduced by the WELSPECS keyword, and attach wells to them (the
   input parameters @wells and @groups come driectly from the welspecs
   keyword, via the sched_kw_welspecs_init_child_parent_list()
   function.
*/


static void sched_history_add_groups_welspecs( sched_history_type * sched_history , const sched_kw_welspecs_type * welspecs , int report_step , const stringlist_type * wells, const stringlist_type * groups) {
  for (int i = 0; i < stringlist_get_size( groups ); i++) {
    const char * group_name  = stringlist_iget( groups , i );
    const char * well_name   = stringlist_iget( wells , i );
    well_history_type * well = sched_history_get_well( sched_history , well_name );
    group_history_type * group;
    if (!hash_has_key( sched_history->group_history , group_name )) 
      sched_history_add_group( sched_history , group_history_alloc( group_name , sched_history->time ) , NULL , report_step );
    
    group = sched_history_get_group( sched_history , group_name );
    group_history_add_child( group , well , well_name , report_step);
  }
}




void sched_history_update( sched_history_type * sched_history, const sched_file_type * sched_file ) {
  
  sched_history_realloc( sched_history );
  sched_history_add_FIELD_group( sched_history );
  {
    int block_nr;
    stringlist_type * well_list  = stringlist_alloc_new();    
    stringlist_type * group_list = stringlist_alloc_new();

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
        case(WCONINJE):
          {
            const sched_kw_wconinje_type * wconinje = sched_kw_get_data( kw );
            sched_kw_wconinje_init_well_list( wconinje , well_list );
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
        case(WELSPECS):
          {
            const sched_kw_welspecs_type * welspecs = sched_kw_get_data( kw );
            sched_kw_welspecs_init_child_parent_list( welspecs , well_list , group_list );
            sched_history_add_wells( sched_history , welspecs , well_list );
            sched_history_add_groups_welspecs( sched_history , welspecs , report_step , well_list , group_list );
          }
          break;
        case(GRUPTREE):
          {
            const sched_kw_gruptree_type * gruptree = sched_kw_get_data( kw );
            stringlist_type * parent_group_list = group_list;
            stringlist_type * child_group_list  = well_list;

            sched_kw_gruptree_init_child_parent_list( gruptree , child_group_list , parent_group_list );
            sched_history_add_groups_gruptree( sched_history , gruptree , report_step , well_list , group_list );
          }
          break;
        default:
          /*   */
          break;
        }
      }
    }
    stringlist_free( well_list );
    stringlist_free( group_list );
  }
  sched_history_install_index( sched_history );
}
