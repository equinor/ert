#include <stdbool.h>
#include <string.h>
#include <util.h>
#include <gruptree.h>

typedef struct grup_struct grup_type;
typedef struct well_struct well_type;

struct grup_struct{
  bool   isleaf;
  bool   isfield;             /* Field node can have both wells and grups as children. */
  char * name;

  const grup_type * parent;
  hash_type       * children; /* If not a leaf grup, pointers to other grups,
                                 if leaf grup, pointers to wells. Special case
                                 for FIELD, which can contain both wells and
                                 grups. */
};

struct well_struct{
  char            * name;
  const grup_type * parent;
};

struct gruptree_struct{
  hash_type * grups;
  hash_type * wells;
};



/*************************************************************************/



static grup_type * grup_alloc(const char * name, const grup_type * parent)
{
   grup_type * grup = util_malloc(sizeof * grup, __func__);

   grup->isleaf   = true;
   grup->isfield  = false;
   grup->name     = util_alloc_string_copy(name);
   grup->parent   = parent;
   grup->children = hash_alloc();

   return grup;
}



static void grup_free(grup_type * grup)
{
  free(grup->name);
  hash_free(grup->children);
  free(grup);
}



static void grup_free__(void * grup)
{
  grup_free( (grup_type *) grup);
}



static const char * grup_get_parent_name(const grup_type * grup)
{
  if(grup->parent != NULL)
    return grup->parent->name;
  else
    return NULL;
}



static well_type * well_alloc(const char * name, const grup_type * parent)
{
  well_type * well = util_malloc(sizeof * well, __func__);
  well->name = util_alloc_string_copy(name);
  well->parent = parent;
  return well;
}


static void well_free(well_type * well)
{
  free(well->name);
  free(well);
}



static void well_free__(void * well)
{
  well_free( (well_type *) well);
}



static const char * well_get_parent_name(const well_type * well)
{
  return well->parent->name;
}



static void gruptree_well_hash_iter__(gruptree_type * gruptree, const char * grupname, hash_type * well_hash)
{
  /*
    Be aware!

    This function does not handle grupname "FIELD".
  */
  if(!hash_has_key(gruptree->grups, grupname))
    util_abort("%s: Internal error.\n", __func__);
  
  grup_type * grup = hash_get(gruptree->grups, grupname);
  if(!grup->isleaf)
  {
    int size = hash_get_size(grup->children);
    char ** keylist = hash_alloc_keylist(grup->children);
    for(int i=0; i<size; i++)
    {
      gruptree_well_hash_iter__(gruptree, keylist[i], well_hash);
    }
    util_free_stringlist(keylist, size);
  }
  else
  {
    int size = hash_get_size(grup->children);
    char ** keylist = hash_alloc_keylist(grup->children);
    for(int i=0; i<size; i++)
    {
      well_type * well = hash_get(gruptree->wells, keylist[i]);
      hash_insert_ref(well_hash, keylist[i], well);
    }
    util_free_stringlist(keylist, size);
  }
}



static gruptree_type * gruptree_alloc_empty()
{
  gruptree_type * gruptree = util_malloc(sizeof * gruptree, __func__);
  gruptree->grups = hash_alloc();
  gruptree->wells = hash_alloc();
  return gruptree;
}



/*************************************************************************/


gruptree_type * gruptree_alloc()
{
  gruptree_type * gruptree = gruptree_alloc_empty();

  grup_type * field = grup_alloc("FIELD", NULL);
  field->isfield    = true;
  hash_insert_hash_owned_ref(gruptree->grups, "FIELD", field, grup_free__);

  return gruptree;
}



void gruptree_free(gruptree_type * gruptree)
{
  hash_free(gruptree->grups);
  hash_free(gruptree->wells);
  free(gruptree);
}



void gruptree_register_grup(gruptree_type * gruptree, const char * name, const char * parent_name)
{
  grup_type * parent;
  grup_type * newgrp;

  if(parent_name == NULL)
    util_abort("%s: Trying to insert group %s with NULL parent - aborting.\n", __func__, name);
  if(strcmp(name, parent_name) == 0)
    util_abort("%s: Trying to insert group %s with itself as parent - aborting.\n", __func__, name);

  if(strcmp(name, "FIELD") == 0)
    util_abort("%s: Internal error - insertion of group FIELD is not allowed - aborting.\n", __func__);

  //////////////////////////////////////////////////////////

  if(!hash_has_key(gruptree->grups, parent_name))
    gruptree_register_grup(gruptree, parent_name, "FIELD");

  parent = hash_get(gruptree->grups, parent_name);

  if(parent->isleaf && !parent->isfield && hash_get_size(parent->children) > 0)
  {
    util_abort("%s: Group %s contains wells, cannot contain other groups.\n", __func__, parent_name);
  }

  if(hash_has_key(gruptree->grups, name))
  {
    newgrp = hash_get(gruptree->grups, name);
    hash_del(newgrp->parent->children, name);

    newgrp->parent = parent;
  }
  else
  {
    newgrp = grup_alloc(name, parent);
    hash_insert_hash_owned_ref(gruptree->grups, name, newgrp, grup_free__);
  }

  parent->isleaf = false;
  hash_insert_ref(parent->children, name, newgrp);
}



void gruptree_register_well(gruptree_type * gruptree, const char * name, const char * parent_name)
{
  grup_type * parent;
  well_type * well;

  if(!hash_has_key(gruptree->grups, parent_name))
    gruptree_register_grup(gruptree, parent_name, "FIELD");

  parent = hash_get(gruptree->grups, parent_name);

  if(!parent->isleaf && !parent->isfield)
    util_abort("%s: Group %s is not FIELD and contains other groups, cannot contain wells.\n", __func__, parent_name);

  if(hash_has_key(gruptree->wells, name))
  {
    well = hash_get(gruptree->wells, name);
    hash_del(well->parent->children, name);
    well->parent = parent;
  }
  else
  {
    well = well_alloc(name, parent);
    hash_insert_hash_owned_ref(gruptree->wells, name, well, well_free__);
  }
  hash_insert_ref(well->parent->children, name, well);
}



char ** gruptree_alloc_grup_well_names(gruptree_type * gruptree, const char * grupname, int * num_wells)
{
  char ** well_names;
  
  if(!hash_has_key(gruptree->grups, grupname))
    util_abort("%s: Group %s does not exist.\n", __func__, grupname);

  if(strcmp(grupname, "FIELD") == 0)
  {
    *num_wells  = hash_get_size(gruptree->wells);
    well_names = hash_alloc_keylist(gruptree->wells);
  }
  else
  {
    hash_type * well_hash = hash_alloc();
    gruptree_well_hash_iter__(gruptree, grupname, well_hash);

    *num_wells = hash_get_size(well_hash);
    well_names = hash_alloc_keylist(well_hash);

    hash_free(well_hash);
  }

  return well_names;
}



gruptree_type * gruptree_copyc(const gruptree_type * gruptree)
{
  gruptree_type * gruptree_new = gruptree_alloc();

  {
    int num_grups = hash_get_size(gruptree->grups);
    char ** grup_list = hash_alloc_keylist(gruptree->grups);
    for(int i=0; i<num_grups; i++)
    {
      if(strcmp(grup_list[i], "FIELD") == 0)
        continue;

      grup_type * grup  = hash_get(gruptree->grups, grup_list[i]);
      gruptree_register_grup(gruptree_new, grup_list[i], grup_get_parent_name(grup));
    }
    util_free_stringlist(grup_list, num_grups);
  }

  {
    int num_wells = hash_get_size(gruptree->wells);
    char ** well_list = hash_alloc_keylist(gruptree->wells);
    for(int i=0; i<num_wells; i++)
    {
      well_type * well  = hash_get(gruptree->wells, well_list[i]);
      gruptree_register_well(gruptree_new, well_list[i], well_get_parent_name(well));
    }
    util_free_stringlist(well_list, num_wells);

  }
  
  return gruptree_new;
}



void gruptree_fwrite(const gruptree_type * gruptree, FILE * stream)
{
  {
    int num_grups = hash_get_size(gruptree->grups);
    char ** grup_list = hash_alloc_keylist(gruptree->grups);

    util_fwrite(&num_grups, sizeof num_grups, 1, stream, __func__);

    for(int i=0; i<num_grups; i++)
    {
      if(strcmp(grup_list[i], "FIELD") == 0)
        continue;

      grup_type * grup  = hash_get(gruptree->grups, grup_list[i]);
      const char * parent_name = grup_get_parent_name(grup);

      util_fwrite_string(grup_list[i], stream);
      util_fwrite_string(parent_name, stream);
    }
    util_free_stringlist(grup_list, num_grups);
  }

  {
    int num_wells = hash_get_size(gruptree->wells);
    char ** well_list = hash_alloc_keylist(gruptree->wells);

    util_fwrite(&num_wells, sizeof num_wells, 1, stream, __func__);

    for(int i=0; i<num_wells; i++)
    {
      well_type * well  = hash_get(gruptree->wells, well_list[i]);
      const char * parent_name = well_get_parent_name(well);
      util_fwrite_string(well_list[i], stream);
      util_fwrite_string(parent_name,  stream);
    }
    util_free_stringlist(well_list, num_wells);

  }
  
}



gruptree_type * gruptree_fread_alloc(FILE * stream)
{
  gruptree_type * gruptree = gruptree_alloc();
  {
    int num_grups;
    util_fread(&num_grups, sizeof num_grups, 1, stream, __func__);
    for(int i=0; i<num_grups-1; i++) /* Skipping FIELD. */
    {
      char * name    = util_fread_alloc_string(stream);
      char * parent  = util_fread_alloc_string(stream);

      gruptree_register_grup(gruptree, name, parent);

      free(name);
      free(parent);
    }
  }

  {
    int num_wells;
    util_fread(&num_wells, sizeof num_wells, 1, stream, __func__);
    for(int i=0; i<num_wells; i++)
    {
      char * name    = util_fread_alloc_string(stream);
      char * parent  = util_fread_alloc_string(stream);

      gruptree_register_well(gruptree, name, parent);

      free(name);
      free(parent);
    }
  }

  return gruptree;
}






/**********************************************************************************/



void gruptree_printf_grup_wells(gruptree_type * gruptree, const char * grupname)
{
  int num_wells;
  char ** well_names = gruptree_alloc_grup_well_names(gruptree, grupname, &num_wells);

  printf("WELLS IN GROUP %s:\n", grupname);
  printf("-----------------------\n");
  
  for(int i=0; i<num_wells; i++)
    printf("%s\n", well_names[i]);

  printf("\n");
  util_free_stringlist(well_names, num_wells);

}





