#include <hash.h>
#include <list.h>
#include <util.h>
#include <sched_file.h>

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


typedef struct sched_block_struct sched_block_type;

struct sched_block_struct {
  list_type * kw_list;      /* A list of sched_kw's in the block.   */
  time_t      block_start_time;
  time_t      block_end_time;
};



struct sched_file_struct {
  list_type * blocks;      /* A list of chronologically sorted
                              sched_block_type's. */
};



/************************************************************************/



static sched_block_type * sched_block_alloc_empty()
{
  sched_block_type * block = util_malloc(sizeof * block, __func__);
  
  block->kw_list = list_alloc();

  return block;
}



static void sched_block_free(sched_block_type * block)
{
  list_free(block->kw_list);

  free(block);
}



static void sched_block_free__(void * block)
{    
  sched_block_free( (sched_block_type *) block);
}



static void sched_block_add_kw(sched_block_type * block, sched_kw_type * kw)
{
  list_append_list_owned_ref(block->kw_list, kw, sched_kw_free__);
}



static sched_kw_type * sched_block_iget_kw(sched_block_type * block, int i)
{
  list_node_type * sched_kw_node = list_iget_node(block->kw_list, i);
  sched_kw_type * sched_kw  = list_node_value_ptr(sched_kw_node);
  return sched_kw;
}



static void sched_block_fwrite(sched_block_type * block, FILE * stream)
{
  int len = list_get_size(block->kw_list);
  util_fwrite(&len, sizeof len, 1, stream, __func__);

  for(int i=0; i<len; i++)
  {
    sched_kw_type * sched_kw = sched_block_iget_kw(block, i);
    sched_kw_fwrite(sched_kw, stream);
  }

  util_fwrite(&block->block_start_time, sizeof block->block_start_time, 1, stream, __func__);
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
  return block;
}



static void sched_block_fprintf(const sched_block_type * block, FILE * stream)
{
  list_node_type * sched_kw_node = list_get_head(block->kw_list);
  while(sched_kw_node != NULL)
  {
    const sched_kw_type * sched_kw = list_node_value_ptr(sched_kw_node);
    sched_kw_fprintf(sched_kw, stream);
    sched_kw_node = list_node_get_next(sched_kw_node);
  }
}



static int sched_block_get_size(const sched_block_type * block)
{
  return list_get_size(block->kw_list);
}



static sched_kw_type * sched_block_iget_kw_ref(const sched_block_type * block, int i)
{
  list_node_type * sched_kw_node = list_iget_node(block->kw_list, i);
  sched_kw_type * sched_kw = list_node_value_ptr(sched_kw_node);
  return sched_kw;
}



static sched_kw_type * sched_block_get_last_kw_ref(const sched_block_type * block)
{
  list_node_type * sched_kw_node = list_get_tail(block->kw_list);
  sched_kw_type  * sched_kw = (sched_kw_type *) list_node_value_ptr(sched_kw_node);
  return sched_kw;
}



static void sched_file_add_block(sched_file_type * sched_file, sched_block_type * block)
{
  list_append_list_owned_ref(sched_file->blocks, block, sched_block_free__);
}



static sched_block_type * sched_file_iget_block_ref(const sched_file_type * sched_file, int i)
{
  list_node_type * sched_block_node = list_iget_node(sched_file->blocks, i);
  sched_block_type * sched_block  = list_node_value_ptr(sched_block_node);
  return sched_block;
}



static void sched_file_build_block_dates(sched_file_type * sched_file, time_t start_date)
{
  int num_restart_files = sched_file_get_nr_restart_files(sched_file);
  time_t curr_time, new_time;

  if(num_restart_files < 1)
    util_abort("%s: Error - empty sched_file - aborting.\n", __func__);

  /* Special case for block 0. */
  sched_block_type * sched_block = sched_file_iget_block_ref(sched_file, 0);
  sched_block->block_start_time  = start_date ;
  sched_block->block_end_time    = start_date ;

  curr_time = start_date;
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



sched_file_type * sched_file_alloc()
{
  sched_file_type * sched_file = util_malloc(sizeof * sched_file, __func__);
  sched_file->blocks = list_alloc();
  return sched_file;
}



void sched_file_free(sched_file_type * sched_file)
{
  list_free(sched_file->blocks);
  free(sched_file);
}



void sched_file_parse(sched_file_type * sched_file, time_t start_date, const char * filename)
{
  bool at_eof = false;

  sched_kw_type    * current_kw;
  sched_block_type * current_block;

  /* Add the first empty pseudo block. */
  sched_file_add_block(sched_file, sched_block_alloc_empty());

  FILE * stream = util_fopen(filename, "r");

  current_block = sched_block_alloc_empty();
  current_block->block_start_time = start_date;
  
  current_kw = sched_kw_fscanf_alloc(stream, &at_eof);
  while(!at_eof)
  { 
    sched_type_enum type = sched_kw_get_type(current_kw);

    if(type == DATES || type == TSTEP || type == TIME)
    {
      int num_steps;
      sched_kw_type ** sched_kw_dates = sched_kw_restart_file_split_alloc(current_kw, &num_steps);
      sched_kw_free(current_kw);
      for(int i=0; i<num_steps; i++)
      {
        sched_block_add_kw(current_block, sched_kw_dates[i]);
        sched_file_add_block(sched_file, current_block);
        current_block = sched_block_alloc_empty();
      }
      free(sched_kw_dates); /* Note: This is *not* the storage! */
    }
    else{
      sched_block_add_kw(current_block, current_kw);
    }

    current_kw = sched_kw_fscanf_alloc(stream, &at_eof);
  } 
  fclose(stream);
  sched_block_free(current_block); /* Free the last non-proper block. */

  sched_file_build_block_dates(sched_file, start_date);
}



int sched_file_get_nr_restart_files(const sched_file_type * sched_file)
{
  return list_get_size(sched_file->blocks);
}



void sched_file_fprintf_i(const sched_file_type * sched_file, int last_restart_file, const char * file)
{
  FILE * stream = util_fopen(file, "w");
  for(int i=0; i<=last_restart_file; i++)
  {
    list_node_type * sched_block_node = list_iget_node(sched_file->blocks, i);
    const sched_block_type * sched_block = list_node_value_ptr(sched_block_node);
    sched_block_fprintf(sched_block, stream);
  }
  fprintf(stream, "END\n");
  fclose(stream);
}



void sched_file_fwrite(const sched_file_type * sched_file, FILE * stream)
{
  int len = sched_file_get_nr_restart_files(sched_file);

  util_fwrite(&len, sizeof len, 1, stream, __func__);

  for(int i=0; i<len; i++)
  {
    sched_block_type * block = sched_file_iget_block_ref(sched_file, i);
    sched_block_fwrite(block, stream);
  }
}



sched_file_type * sched_file_fread_alloc(FILE * stream)
{

  int len;
  sched_file_type * sched_file = sched_file_alloc();
  util_fread(&len, sizeof len, 1, stream, __func__);

  for(int i=0; i<len; i++)
  {
    sched_block_type * block = sched_block_fread_alloc(stream);
    sched_file_add_block(sched_file, block);
  }

  return sched_file;
}



int sched_file_get_restart_file_from_time_t(const sched_file_type * sched_file, time_t time)
{
  int num_restart_files = sched_file_get_nr_restart_files(sched_file);
  for(int i=0; i<num_restart_files; i++)
  {
    time_t block_end_time = sched_file_iget_block_end_time(sched_file, i);
    if(block_end_time > time)
    {
      util_abort("%s: Time variable does not cooincide with any restart file. Aborting.\n", __func__);
    }
    else if(block_end_time == time)
    {
      return i; /* ECLIPSE Counting. */
    }
  }

  // If we are here, time did'nt correspond a restart file. Abort.
  util_abort("%s: Time variable does not cooincide with any restart file. Aborting.\n", __func__);
  return 0;
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
