#include <hash.h>
#include <stringlist.h>
#include <util.h>
#include <sched_file.h>
#include <sched_util.h>
#include <vector.h>
#include <parser.h>

/* This sched_file.c contains code for internalizing an ECLIPSE
   schedule file.

   Two structs are defined in this file:

    1. The sched_file_struct, which can be accessed externaly
       through various interface functions.

    2. The sched_block_struct, which is for internal use.

   The internalization function 'sched_file_parse' splits the ECLIPSE
   schedule file into a sequence of 'sched_block_type's, where a single 
   block contains one or more keywords. Except for the first block, which
   is empty per definition, the last keyword in a block will always be
   a timing keyword like DATES or TSTEP. Thus, the number of blocks in
   the sched_file_struct will always cooincide with the number of 
   restart files in the ECLIPSE simulation. In order to make this work,
   TSTEP and DATES keyword containing multiple data, are split into
   a sequence of keywords. 

   Note the following:

   1. This implies that scheduling data after the last timing
      keyword is irrelevant. This is similar to how ECLIPSE works.

   2. Scheduling data after keyword END is ignored, since this
      is interpreted as the end of the SCHEDULE section.

*/

#define SCHED_FILE_TYPE_ID 677198

typedef struct sched_block_struct sched_block_type;

struct sched_block_struct {
  vector_type * kw_list;           /* A list of sched_kw's in the block.   */
  time_t        block_start_time;  
  time_t        block_end_time;
};



struct sched_file_struct {
  UTIL_TYPE_ID_DECLARATION;
  vector_type     * kw_list;
  vector_type     * kw_list_by_type;
  vector_type     * blocks;           /* A list of chronologically sorted sched_block_type's. */
  stringlist_type * files;            /* The name of the files which have been parsed to generate this sched_file instance. */
  time_t            start_time;       /* The start of the simulation. */
};



/************************************************************************/



static sched_block_type * sched_block_alloc_empty()
{
  sched_block_type * block = util_malloc(sizeof * block, __func__);
  
  block->kw_list = vector_alloc_new();

  return block;
}


static void sched_block_debug_fprintf( const sched_block_type * block ) {
  util_fprintf_date( block->block_start_time , stdout );
  printf(" -- ");
  util_fprintf_date( block->block_end_time , stdout );
  printf("\n");
  {
    int i;
    for (i=0; i < vector_get_size( block->kw_list ); i++) {
      const sched_kw_type * sched_kw = vector_iget_const( block->kw_list , i);
      printf("%s \n",sched_kw_get_type_name( sched_kw ));
    }
  }
}


static void sched_block_free(sched_block_type * block)
{
  vector_free(block->kw_list);

  free(block);
}



static void sched_block_free__(void * block)
{    
  sched_block_free( (sched_block_type *) block);
}



static void sched_block_add_kw(sched_block_type * block, const sched_kw_type * kw)
{
  vector_append_ref(block->kw_list , kw );
}



static sched_kw_type * sched_block_iget_kw(sched_block_type * block, int i)
{
  return vector_iget( block->kw_list , i);
}



static void sched_block_fwrite(sched_block_type * block, FILE * stream)
{
  int len = vector_get_size(block->kw_list);
  util_fwrite(&len, sizeof len, 1, stream, __func__);

  for(int i=0; i<len; i++)
  {
    sched_kw_type * sched_kw = sched_block_iget_kw(block, i);
    sched_kw_fwrite(sched_kw, stream);
  }

  util_fwrite(&block->block_start_time, sizeof block->block_start_time, 1, stream, __func__);
  util_fwrite(&block->block_end_time  , sizeof block->block_end_time,   1, stream, __func__);
}



static sched_block_type * sched_block_fread_alloc(FILE * stream)
{
  sched_block_type * block = sched_block_alloc_empty();
  int len;
  bool at_eof = false;

  util_fread(&len, sizeof len, 1, stream, __func__);

  for(int i=0; i<len; i++)
  {
    sched_kw_type * sched_kw = sched_kw_fread_alloc(stream, &at_eof);
    sched_block_add_kw(block, sched_kw);
  }
 
  util_fread(&block->block_start_time, sizeof block->block_start_time, 1, stream, __func__);
  util_fread(&block->block_end_time,   sizeof block->block_end_time  , 1, stream, __func__);
  return block;
}



static void sched_block_fprintf(const sched_block_type * block, FILE * stream)
{
  int i;
  for (i=0; i < vector_get_size(block->kw_list); i++) {
    const sched_kw_type * sched_kw = vector_iget_const( block->kw_list , i);
    sched_kw_fprintf(sched_kw, stream);
  }
}



static int sched_block_get_size(const sched_block_type * block)
{
  return vector_get_size(block->kw_list);
}



static sched_kw_type * sched_block_iget_kw_ref(const sched_block_type * block, int i)
{
  return vector_iget(block->kw_list , i);
}



static sched_kw_type * sched_block_get_last_kw_ref(const sched_block_type * block)
{
  int last_index = vector_get_size( block->kw_list ) - 1;
  return sched_block_iget_kw_ref( block , last_index );
}



static void sched_file_add_block(sched_file_type * sched_file, sched_block_type * block)
{
  vector_append_owned_ref(sched_file->blocks , block , sched_block_free__);
}



static sched_block_type * sched_file_iget_block_ref(const sched_file_type * sched_file, int i)
{
  return vector_iget(sched_file->blocks , i);
}



static void sched_file_build_block_dates(sched_file_type * sched_file)
{
  int num_restart_files = sched_file_get_num_restart_files(sched_file);
  time_t curr_time, new_time;

  if(num_restart_files < 1)
    util_abort("%s: Error - empty sched_file - aborting.\n", __func__);

  /* Special case for block 0. */
  sched_block_type * sched_block = sched_file_iget_block_ref(sched_file, 0);
  sched_block->block_start_time  = sched_file->start_time ;
  sched_block->block_end_time    = sched_file->start_time ;

  curr_time = sched_file->start_time;
  for(int i=1; i<num_restart_files; i++)
  {
    sched_block = sched_file_iget_block_ref(sched_file, i);
    sched_block->block_start_time = curr_time;
    
    sched_kw_type * timing_kw = sched_block_get_last_kw_ref(sched_block);
    new_time = sched_kw_get_new_time(timing_kw, curr_time);
    
    if(curr_time > new_time)
      util_abort("%s: Schedule file contains negative timesteps - aborting.\n",__func__);
    
    curr_time = new_time;
    sched_block->block_end_time = curr_time;
  }
}




/******************************************************************************/



static void sched_file_add_kw( sched_file_type * sched_file , const sched_kw_type * kw) {
  vector_append_owned_ref( sched_file->kw_list , kw , sched_kw_free__);
}


static void sched_file_update_index( sched_file_type * sched_file ) {
  int ikw;


  /* By type index */
  {
    if (sched_file->kw_list_by_type != NULL) 
      vector_free( sched_file->kw_list_by_type );
    sched_file->kw_list_by_type = vector_alloc_NULL_initialized( NUM_SCHED_KW_TYPES );
    for (ikw = 0; ikw < vector_get_size( sched_file->kw_list ); ikw++) {
      const sched_kw_type * kw = vector_iget_const( sched_file->kw_list , ikw );
      sched_type_enum type     = sched_kw_get_type( kw );
      {
        vector_type * tmp        = vector_iget( sched_file->kw_list_by_type , type );
        
        if (tmp == NULL) {
          tmp = vector_alloc_new();
          vector_iset_owned_ref( sched_file->kw_list_by_type , type , tmp , vector_free__ );
        }
        
        vector_append_ref( tmp , kw );
      }
    }
  }

  
  
  /* Block based on restart number. */
  {
    time_t current_time;
    sched_block_type * current_block;
    vector_clear( sched_file->blocks );

    /* 
       Adding a pseudo block at the start which runs from the start of
       time (i.e. EPOCH start 01/01/1970) to simulation start.
    */
    current_block = sched_block_alloc_empty( 0 );
    current_block->block_start_time  = sched_file->start_time;//-1;     /* Need this funny node - hhmmmmmm */
    current_block->block_end_time    = sched_file->start_time;
    sched_file_add_block( sched_file , current_block );
    
    current_block = sched_block_alloc_empty( 0 );
    current_block->block_start_time  = sched_file->start_time;
    current_time = sched_file->start_time;
    
    for (ikw = 0; ikw < vector_get_size( sched_file->kw_list ); ikw++) {
      const sched_kw_type * kw = vector_iget_const( sched_file->kw_list , ikw );
      sched_type_enum type     = sched_kw_get_type( kw );
      {
        sched_block_add_kw( current_block , kw );
        if(type == DATES || type == TSTEP || type == TIME) {
          /**
             Observe that when we enocunter a time-based keyword we do the following:
             
               1. Finish the the current block by setting the end_time
                  field and add this block to the sched_file
                  structure.

               2. Create a new block starting at current time.

              ------- 

              Blocks are not actually added to the sched_file instance
              before they are terminated with a DATES/TSTEP
              keyword. This implies that keywords which come after the
              last DATES/TSTEP keyword are lost.
          */
               
          current_time = sched_kw_get_new_time( kw , current_time );

          /* Finishing off the current block, and adding it to the sched_file. */
          current_block->block_end_time = current_time;
          sched_file_add_block( sched_file , current_block );
          
          /* Creating a new block - not yet added to the sched_file. */
          current_block = sched_block_alloc_empty( vector_get_size( sched_file->blocks ));
          current_block->block_start_time = current_time;
        }
      }
    }
    /*
      Free the last block, which has not been added to the sched_file
      object.
    */
    sched_block_free( current_block );
  }
}


sched_file_type * sched_file_alloc(time_t start_time)
{
  sched_file_type * sched_file = util_malloc(sizeof * sched_file, __func__);
  UTIL_TYPE_ID_INIT( sched_file , SCHED_FILE_TYPE_ID);
  sched_file->kw_list          = vector_alloc_new();
  sched_file->kw_list_by_type  = NULL;
  sched_file->blocks           = vector_alloc_new();
  sched_file->files    	       = stringlist_alloc_new();
  sched_file->start_time       = start_time;
  return sched_file;
}

UTIL_SAFE_CAST_FUNCTION(sched_file , SCHED_FILE_TYPE_ID);


void sched_file_free(sched_file_type * sched_file)
{
  vector_free( sched_file->blocks );
  vector_free( sched_file->kw_list );
  if (sched_file->kw_list_by_type != NULL)
    vector_free( sched_file->kw_list_by_type );

  stringlist_free( sched_file->files );
  free(sched_file);
}



/**
   This function parses 'further', i.e typically adding another schedule
   file to the sched_file instance.
*/

void sched_file_parse_append(sched_file_type * sched_file , const char * filename) {
  /* 
     We start by writing a new copy of the file with all comments
     stripped out. The remaining parsing is done on this file with no
     comments.
  */
  char * tmp_base             = util_alloc_sprintf("enkf-schedule:%s" , filename);
  char * tmp_file             = util_alloc_tmp_file("/tmp" , tmp_base , true);
  {
    parser_type     * parser    = parser_alloc(" \t"  ,      /* Splitters */
                                               "\'\"" ,      /* Quoters   */
                                               "\n"   ,      /* Specials - splitters which will be kept. */  
                                               "\r"   ,      /* Delete set - these are just deleted. */
                                               "--"   ,      /* Comment start */
                                               "\n");        /* Comment end */  

    stringlist_type * tokens    = parser_tokenize_file( parser , filename , false );
    FILE * stream               = util_fopen(tmp_file , "w");

    stringlist_fprintf( tokens , " " , stream );
    parser_free( parser );
    stringlist_free( tokens );
    fclose(stream);
  }

  {
    bool at_eof      = false;
    sched_kw_type    * current_kw;
    
    FILE * stream = util_fopen(tmp_file , "r");
    stringlist_append_copy( sched_file->files , filename);
    current_kw     = sched_kw_fscanf_alloc(stream, &at_eof);
    while(!at_eof) {
      sched_type_enum type = sched_kw_get_type(current_kw);
      
      if(type == DATES || type == TSTEP || type == TIME) {
        int i , num_steps;
        sched_kw_type ** sched_kw_dates = sched_kw_restart_file_split_alloc(current_kw, &num_steps);
        sched_kw_free(current_kw);

        for(i=0; i<num_steps; i++) 
          sched_file_add_kw( sched_file , sched_kw_dates[i]);

        free(sched_kw_dates);   
      } else
        sched_file_add_kw( sched_file , current_kw);
      
      current_kw = sched_kw_fscanf_alloc(stream, &at_eof);
    } 
    
    fclose(stream);
    sched_file_build_block_dates(sched_file);
    sched_file_update_index( sched_file );
  }
  free( tmp_base );
  free( tmp_file );
}



void sched_file_parse(sched_file_type * sched_file, time_t start_date, const char * filename)
{
  /* 
     Add the first empty pseudo block - this runs from time -infty:start_date.
  */
  sched_file_add_block(sched_file , sched_block_alloc_empty());
  sched_file_parse_append( sched_file , filename );
}


sched_file_type * sched_file_parse_alloc(const char * filename , time_t start_date) {
  sched_file_type * sched_file = sched_file_alloc( start_date );
  sched_file_parse(sched_file , start_date , filename);
  return sched_file;
}



int sched_file_get_num_restart_files(const sched_file_type * sched_file)
{
  return vector_get_size(sched_file->blocks);
}



void sched_file_fprintf_i(const sched_file_type * sched_file, int last_restart_file, const char * file)
{
  FILE * stream = util_fopen(file, "w");
  int num_restart_files = sched_file_get_num_restart_files(sched_file);
  

  if (last_restart_file > num_restart_files) {
    util_abort("%s: you asked for restart nr:%d - the last available restart nr is: %d \n",__func__ , last_restart_file , num_restart_files);
    /* Must abort here because the calling scope is expecting to find last_restart_file.  */
  }
  
  for(int i=0; i<= last_restart_file; i++)
  {
    const sched_block_type * sched_block = vector_iget_const( sched_file->blocks , i);
    sched_block_fprintf(sched_block, stream);
  }
  fprintf(stream, "END\n");
  fclose(stream);
}

/* Writes the complete schedule file. */
void sched_file_fprintf(const sched_file_type * sched_file, const char * file)
{
  int num_restart_files = sched_file_get_num_restart_files(sched_file);
  sched_file_fprintf_i( sched_file , num_restart_files - 1 , file);
}



void sched_file_fwrite(const sched_file_type * sched_file, FILE * stream)
{
  int len = sched_file_get_num_restart_files(sched_file);
  
  util_fwrite(&sched_file->start_time , sizeof sched_file->start_time , 1 , stream , __func__);
  util_fwrite(&len, sizeof len, 1, stream, __func__);

  for(int i=0; i<len; i++)
  {
    sched_block_type * block = sched_file_iget_block_ref(sched_file, i);
    sched_block_fwrite(block, stream);
  }
}



sched_file_type * sched_file_fread_alloc(FILE * stream)
{

  time_t start_time;
  util_fwrite(&start_time , sizeof start_time , 1 , stream , __func__);
  {
    int len;
    
    sched_file_type * sched_file = sched_file_alloc( start_time );
    util_fread(&len, sizeof len, 1, stream, __func__);
    
    for(int i=0; i<len; i++)
      {
	sched_block_type * block = sched_block_fread_alloc(stream);
	sched_file_add_block(sched_file, block);
      }
    
    return sched_file;
  }
}


/*
  const char * sched_file_get_filename(const sched_file_type * sched_file) {
  return sched_file->filename;
  }
*/


int sched_file_get_restart_nr_from_time_t(const sched_file_type * sched_file, time_t time)
{
  int num_restart_files = sched_file_get_num_restart_files(sched_file);
  for(int i=0; i<num_restart_files; i++)
  {
    time_t block_end_time = sched_file_iget_block_end_time(sched_file, i);
    if(block_end_time > time)
      util_abort("%s: Time variable does not cooincide with any restart file. Aborting.\n", __func__);
    else if(block_end_time == time)
      return i; 
  }
  
  // If we are here, time did'nt correspond a restart file. Abort.
  util_abort("%s: Time variable does not cooincide with any restart file. Aborting.\n", __func__);
  return 0;
}


/**
   This function finds the restart_nr for the a number of days after
   simulation start.
*/

int sched_file_get_restart_nr_from_days(const sched_file_type * sched_file , double days) {
  time_t time = sched_file_iget_block_start_time(sched_file, 0);
  util_inplace_forward_days( &time , days);
  return sched_file_get_restart_nr_from_time_t(sched_file , time);
}



time_t sched_file_iget_block_start_time(const sched_file_type * sched_file, int i)
{
  sched_block_type * block = sched_file_iget_block_ref(sched_file, i);
  return block->block_start_time;
}



time_t sched_file_iget_block_end_time(const sched_file_type * sched_file, int i)
{
  sched_block_type * block = sched_file_iget_block_ref(sched_file, i);
  return block->block_end_time;
}


double sched_file_iget_block_start_days(const sched_file_type * sched_file, int i)
{
  sched_block_type * block = sched_file_iget_block_ref(sched_file, i);
  return util_difftime_days( sched_file->start_time , block->block_start_time );
}


double sched_file_iget_block_end_days(const sched_file_type * sched_file, int i)
{
  sched_block_type * block = sched_file_iget_block_ref(sched_file, i);
  return util_difftime_days( sched_file->start_time , block->block_end_time );
}


double sched_file_get_sim_days(const sched_file_type * sched_file , int report_step) {
  return sched_file_iget_block_end_days( sched_file , report_step );
}


time_t sched_file_get_sim_time(const sched_file_type * sched_file , int report_step) {
  return sched_file_iget_block_end_time( sched_file , report_step );
}





int sched_file_iget_block_size(const sched_file_type * sched_file, int block_nr)
{
  sched_block_type * block = sched_file_iget_block_ref(sched_file, block_nr);
  return sched_block_get_size(block);
}



sched_kw_type * sched_file_ijget_block_kw_ref(const sched_file_type * sched_file, int block_nr, int kw_nr)
{
  sched_block_type * block = sched_file_iget_block_ref(sched_file, block_nr);
  sched_kw_type * sched_kw = sched_block_iget_kw_ref(block, kw_nr);
  return sched_kw;
}



static void __sched_file_summarize_line(int restart_nr , time_t start_time , time_t t , FILE * stream) {
  double days    = util_difftime( start_time , t , NULL , NULL , NULL , NULL) / (24 * 3600);
  int mday , month , year;
  
  util_set_date_values(t , &mday , &month , &year);
  fprintf(stream , "%02d/%02d/%04d   %7.1f days     %04d \n", mday , month , year , days , restart_nr);
}




void sched_file_summarize(const sched_file_type * sched_file , FILE * stream) {
  int len            = sched_file_get_num_restart_files(sched_file);
  time_t  start_time = sched_file_iget_block_start_time(sched_file , 0);
  for(int i=1; i<len; i++) {
    time_t t = sched_file_iget_block_start_time(sched_file , i);
    __sched_file_summarize_line(i - 1 , start_time , t , stream);
  }
  {
    time_t t = sched_file_iget_block_end_time(sched_file , len - 1);
    __sched_file_summarize_line(len - 1 , start_time , t , stream);
  }
}


/** 
    deep_copy is NOT implemented. With shallow_copy you get a new
    container (i.e. vector) , but the node content is unchanged.
*/

sched_file_type * sched_file_alloc_copy(const sched_file_type * src , bool deep_copy) {
  int ikw;
  sched_file_type * target = sched_file_alloc(src->start_time);
  
  for (ikw = 0; ikw < vector_get_size( src->kw_list ); ikw++) {
    sched_kw_type * kw = vector_iget( src->kw_list , ikw );
    sched_file_add_kw( target , kw );
  }
                                                                
  
  {
    int i;
    for (i = 0; i < stringlist_get_size( src->files ); i++) {
      if (deep_copy)
        stringlist_append_copy( target->files , stringlist_iget(src->files , i));
      else
        stringlist_append_ref( target->files , stringlist_iget(src->files , i));
    }
  }

  sched_file_update_index( target );
  return target;
}


/*****************************************************************/


static void sched_file_update_block(sched_block_type * block , 
				    int restart_nr, 
				    sched_type_enum kw_type , 
				    sched_file_callback_ftype * callback,
				    void * arg) {
  int ikw;
  for (ikw = 0; ikw < vector_get_size(block->kw_list); ikw++) {
    sched_kw_type * sched_kw = sched_block_iget_kw( block , ikw);
    if (sched_kw_get_type( sched_kw ) == kw_type)
      callback( sched_kw_get_data( sched_kw) , restart_nr , arg);  /* Calling back to 'user-space' to actually do the update. */
  }
}



/**
   This function is designed to facilitate 'user-space' update of the
   keywords in the schedule file based on callbacks. The function is
   called with two report steps, a type ID of the sched_kw type which
   should be updated, and a function pointer which will be invoked on
   all the relevant keywords. 
*/



void sched_file_update_blocks(sched_file_type * sched_file, 
			      int restart1 , 
			      int restart2 , 
			      sched_type_enum kw_type,
			      sched_file_callback_ftype * callback,
			      void * callback_arg) {

  int restart_nr;
  if (restart2 > sched_file_get_num_restart_files(sched_file))
    restart2 = sched_file_get_num_restart_files(sched_file) - 1;
  
  for (restart_nr = restart1; restart_nr <= restart2; restart_nr++) {
    sched_block_type * sched_block = sched_file_iget_block_ref( sched_file , restart_nr );
    sched_file_update_block( sched_block , restart_nr , kw_type , callback , callback_arg);
  }
}


/** 
    Update a complete schedule file by using callbacks to
    'user-space'. Say for instance you want to scale up the oilrate in
    well P1. This could be achieved with the following code:

       -- This function is written by the user of the library - in a remote scope.

       void increase_orat_callback(void * void_kw , int restart_nr , void * arg) {
          double scale_factor  = *(( double * ) arg);
	  sched_kw_wconhist_type * kw = sched_kw_wconhist_safe_cast( void_kw );
          sched_kw_wconhist_scale_orat( wconhist_kw , "P1" , scale_factor);
       }

       ....
       ....
       
       sched_file_update(sched_file , WCONHIST , increase_orat_callback , &scale_factor);

    Observe the following about the callback:
  
      * The sched_kw input argument comes as a void pointer, and an
        sched_kw_xxx_safe_cast() function should be used on input to
        check.

      * The user-space level does *NOT* have access to the internals
        of the sched_kw_xxxx type, so the library must provide
       	functions for the relevant state modifications.

      *	The last argumnt (void * arg) - can of course be anything and
        his brother.

*/


void sched_file_update(sched_file_type * sched_file, 
		       sched_type_enum kw_type,
		       sched_file_callback_ftype * callback,
		       void * callback_arg) {
  sched_file_update_blocks(sched_file , 1 , sched_file_get_num_restart_files(sched_file) - 1 , kw_type , callback , callback_arg);
}
