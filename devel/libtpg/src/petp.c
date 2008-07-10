#include <petp.h>
#include <config.h>
#include <util.h>
#include <ecl_kw.h>
#include <field_config.h>
#include <field.h>
#include <tpgzone_util.h>



typedef struct petp_item_struct petp_item_type;



struct petp_item_struct{
  double * data;
  char   * ecl_filename;
  char   * ecl_kw;
};



struct petp_struct
{
  int num_facies;
  int num_target_files;
  petp_item_type ** petp_items;
};



static petp_item_type * petp_item_alloc(int num_facies)
{
  petp_item_type * petp_item = util_malloc(sizeof * petp_item, __func__);
  petp_item->data            = util_malloc(num_facies * sizeof * petp_item->data, __func__);
  return petp_item;
}



static void petp_item_free(petp_item_type * petp_item)
{
  free(petp_item->data);
  free(petp_item->ecl_filename);
  free(petp_item->ecl_kw);
  free(petp_item);
}



static petp_item_type * petp_fscanf_alloc_item(const char * ecl_kw, const char * ecl_filename,
                                               const char * config_filename, const hash_type * facies_kw_hash)
{
  int     num_facies = hash_get_size(     facies_kw_hash);
  /*
    Ugly cast.
  */
  char ** facies_kw  = hash_alloc_keylist((hash_type *) facies_kw_hash);

  petp_item_type * petp_item = petp_item_alloc(num_facies);

  petp_item->ecl_kw       = util_alloc_string_copy(ecl_kw      );
  petp_item->ecl_filename = util_alloc_string_copy(ecl_filename);

  config_type * config = config_alloc(false);
  {
    int i;
    for(i=0; i<num_facies; i++)
      config_init_item(config, facies_kw[i], 0, NULL, true, false, 0, NULL, 1, 1, NULL);
  }
  config_parse(config, config_filename, ENKF_COM_KW);

  {
    int i, facies_int;
    const char * double_str;
    for(i=0; i<num_facies; i++)
    {
      facies_int = hash_get_int(facies_kw_hash, facies_kw[i]);
      double_str = config_iget(config, facies_kw[i], 0);
      if(!util_sscanf_double(double_str, &(petp_item->data[facies_int]) ))
      {
        util_abort("%s: Failed to parse a double from %s - aborting.\n", __func__, double_str);
      }
    }
  }

  config_free(config);
  util_free_stringlist(facies_kw, num_facies);  

  return petp_item;
}



static double * petp_item_apply_alloc(const petp_item_type * petp_item, const int * facies, int size)
{
  int i;
  double * d = util_malloc(size * sizeof *d, __func__);

  for(i=0; i<size; i++)
    d[i] = petp_item->data[ facies[i] ];

  return d;
}



static void petp_item_fwrite(const petp_item_type * petp_item, const int * facies, const int * blocks,
                             int size, const char * grid_file, bool endian_flip)
{
  field_config_type * field_config = tpgzone_field_config_alloc__(petp_item->ecl_filename, petp_item->ecl_kw, grid_file, endian_flip);
  field_type        * field        = field_alloc(field_config);

  field_fload_auto(field, petp_item->ecl_filename, endian_flip);

  {
    double * data = petp_item_apply_alloc(petp_item, facies, size);
    field_indexed_set(field, ecl_double_type, size, blocks, data);
    field_ecl_write(field, petp_item->ecl_filename);
    free(data);
  }

  field_free(field);
  field_config_free(field_config);
}



static petp_type * petp_alloc_empty(int num_target_files)
{
  petp_type * petp        = util_malloc(sizeof * petp, __func__);
  petp->num_target_files  = num_target_files;
  petp->petp_items        = util_malloc(num_target_files * sizeof petp->petp_items, __func__);

  return petp;
}



/************************************************************************************/



void petp_write(const petp_type * petp, const int * facies, const int * blocks, int size, const char * grid_file, bool endian_flip)
{
  int i;

  for(i=0; i<petp->num_target_files; i++)
    petp_item_fwrite(petp->petp_items[i], facies, blocks, size, grid_file, endian_flip);

}



void petp_free(petp_type * petp)
{
  int i;
  for(i=0; i<petp->num_target_files; i++)
    petp_item_free(petp->petp_items[i]);

  free(petp->petp_items);
  free(petp);
}



petp_type * petp_fscanf_alloc(const char * filename, const hash_type * facies_kw_hash)
{
  int  num_target_files;
  char ** target_keys;

  petp_type * petp;

  config_type * config = config_alloc(true);
  config_parse(config, filename, ENKF_COM_KW);

  target_keys = config_alloc_active_list(config, &num_target_files);

  if(num_target_files == 0)
    util_abort("%s: No target fields - aborting.\n",__func__);

  petp = petp_alloc_empty(num_target_files);

  {
    int i;
    const char *item_config_filename;
    const char *ecl_filename;
    const char *ecl_kw;
    for(i=0; i<num_target_files; i++)
    {
     if(config_get_argc(config, target_keys[i]) != 2)
        util_abort("%s: Wrong formatting. Correct format is <ECL_KW> <ECL_FILE> <PETROPHYSICS_FILE> - aborting.\n",__func__);

     ecl_kw               = target_keys[i];
     ecl_filename         = config_iget(config, target_keys[i], 0);
     item_config_filename = config_iget(config, target_keys[i], 1);

     petp->petp_items[i]  = petp_fscanf_alloc_item(ecl_kw, ecl_filename, item_config_filename, facies_kw_hash);
    }
  }

  config_free(config);
  util_free_stringlist(target_keys, num_target_files);
  return petp;
};



