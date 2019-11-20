/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'gruptree.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <stdbool.h>
#include <string.h>

#include <ert/util/util.hpp>
#include <ert/util/hash.hpp>

#include <ert/sched/gruptree.hpp>


/*
  The gruptree struct is made for internalizing the
  GRUPTREE keyword from a schedule section.

  It provides functions which allows for easy
  manipulating of, and access to, the tree structure.

  Since the gruptree may change during simulation,
  and the changes are usually only given in differences,
  i.e. the whole gruptree is NOT restated, we provide a
  way to easily create a new gruptree to reflect the changes.



  EXAMPLE:

  Schedule file:
  ----------------------------

  SKIPREST
  ....
  ....

  GRUPTREE
    GRPA FIELD/
    GRPB FIELD/
    INJE GRPA/
  /

  TSTEP
    100 /
  ....
  ....

  GRUPTREE
    INJE GRPB /
    PROD GRPA /
  /

  TSTEP
    200 /
  ...
  ...
  ----------------------------

  Now, after the first timestep, INJE is no longer
  a subgroup of GRPA, but GRPB. However, the rest
  of the GRUPTREE at the first time step is still
  valid! Note also that a new group PROD has been
  added.

  Suppose that you want to create two gruptree's
  one for each time step. This can be done as follows.


  C code:
  ----------------------------

  gruptree_type * tree_one = gruptree_alloc();
  gruptree_register_grup(tree_two, "GRPA", "FIELD");
  gruptree_register_grup(tree_two, "GRPB", "FIELD");
  gruptree_register_grup(tree_two, "INJE", "GRPA");

  gruptree_register_grup(tree_two, "INJE", "GRPB");
  gruptree_register_grup(tree_two, "PROD", "GRPA");

  .....
  .....
*/



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
   grup_type * grup = (grup_type*)util_malloc(sizeof * grup);

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
  well_type * well = (well_type*)util_malloc(sizeof * well);
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



/**
   This function is called recursively ...
*/
static void gruptree_well_hash_iter__(gruptree_type * gruptree, const char * grupname, hash_type * well_hash)
{

  if(!hash_has_key(gruptree->grups, grupname))
    util_abort("%s: Internal error - grupname %s is not in hash.\n", __func__, grupname);

  grup_type * grup = (grup_type*)hash_get(gruptree->grups, grupname);
  if(grup->isfield)
    util_abort("%s: Internal error - no support for grups with isfield flag.\n", __func__);

  if(!grup->isleaf)
  {
    int size = hash_get_size(grup->children);
    char ** keylist = hash_alloc_keylist(grup->children);

    for(int i=0; i<size; i++)
      gruptree_well_hash_iter__(gruptree, keylist[i], well_hash);  /* Recursive call - should NOT use hash_iter interface.*/

    util_free_stringlist(keylist, size);
  }
  else
  {
    int size = hash_get_size(grup->children);
    char ** keylist = hash_alloc_keylist(grup->children);
    for(int i=0; i<size; i++)
    {
      well_type * well = (well_type*)hash_get(gruptree->wells, keylist[i]);
      hash_insert_ref(well_hash, keylist[i], well);
    }
    util_free_stringlist(keylist, size);
  }
}



static gruptree_type * gruptree_alloc_empty()
{
  gruptree_type * gruptree = (gruptree_type*)util_malloc(sizeof * gruptree);
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


void gruptree_register_grup(gruptree_type * gruptree, const char * name, const char * parent_name)
{
  grup_type * parent;
  grup_type * newgrp;

  //////////////////////////////////////////////////////////

  if(name == NULL)
    util_abort("%s: Trying to insert group %s with NULL name - aborting.\n", __func__, name);
  if(parent_name == NULL)
    util_abort("%s: Trying to insert group %s with NULL parent - aborting.\n", __func__, name);
  if(strcmp(name, parent_name) == 0)
    util_abort("%s: Trying to insert group %s with itself as parent - aborting.\n", __func__, name);

  if(strcmp(name, "FIELD") == 0)
    util_abort("%s: Internal error - insertion of group FIELD is not allowed - aborting.\n", __func__);

  //////////////////////////////////////////////////////////

  if(!hash_has_key(gruptree->grups, parent_name))
    gruptree_register_grup(gruptree, parent_name, "FIELD");

  parent = (grup_type*)hash_get(gruptree->grups, parent_name);

  if(parent->isleaf && !parent->isfield && hash_get_size(parent->children) > 0)
  {
    util_abort("%s: Group %s contains wells, cannot contain other groups.\n", __func__, parent_name);
  }

  if(hash_has_key(gruptree->grups, name))
  {
    newgrp = (grup_type*)hash_get(gruptree->grups, name);
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

  parent = (grup_type*)hash_get(gruptree->grups, parent_name);

  if(!parent->isleaf && !parent->isfield)
    util_abort("%s: Group %s is not FIELD and contains other groups, cannot contain wells.\n", __func__, parent_name);

  if(hash_has_key(gruptree->wells, name))
  {
    well = (well_type*)hash_get(gruptree->wells, name);
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


char ** gruptree_alloc_grup_well_list(gruptree_type * gruptree, const char * grupname, int * num_wells)
{
  char ** well_names;

  if(!hash_has_key(gruptree->grups, grupname))
    util_abort("%s: Group %s is not present in the gruptree.\n", __func__, grupname);

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
