#include <stdlib.h>
#include <string.h>
#include <fs_index.h>
#include <path_fmt.h>
#include <enkf_types.h>
#include <util.h>


typedef struct fs_index_node_struct fs_index_node_type;

struct fs_index_node_struct {
  enkf_impl_type  impl_type;
  enkf_var_type   var_type;
  char           *kw;
};


struct fs_index_struct {
  path_fmt_type * path;
};


/*****************************************************************/

static fs_index_node_type * fs_index_node_alloc(char * kw , enkf_var_type var_type , enkf_impl_type impl_type) {
  fs_index_node_type * index_node = malloc(sizeof * index_node);
  index_node->kw        = kw;
  index_node->var_type  = var_type;
  index_node->impl_type = impl_type;
  return index_node;
}


static fs_index_node_type * fs_index_node_fread_alloc(FILE * stream) {
  enkf_impl_type impl_type;
  enkf_var_type  var_type;
  char * kw;

  fread(&impl_type , sizeof impl_type , 1 , stream );
  fread(&var_type  , sizeof var_type  , 1 , stream );
  kw = util_fread_alloc_string(stream);
  return fs_index_node_alloc(kw , var_type , impl_type);
}


static void fs_index_node_fwrite_data(const char * kw , enkf_var_type var_type , enkf_impl_type impl_type , FILE * stream) {
  fwrite(&impl_type, sizeof impl_type , 1  , stream );
  fwrite(&var_type , sizeof var_type  , 1  , stream );
  util_fwrite_string(kw , stream);
}

static void fs_index_node_fwrite(const fs_index_node_type * index_node , FILE * stream) {
  fs_index_node_fwrite_data(index_node->kw , index_node->var_type , index_node->impl_type , stream);
}


static void fs_index_node_free(fs_index_node_type * index_node) {
  free(index_node->kw);
  free(index_node);
}


/*****************************************************************/


fs_index_type * fs_index_alloc(const char * root_path , const char * index_path) {
  fs_index_type * fs_index = malloc(sizeof * fs_index);
  {
    char * path = util_alloc_full_path(root_path , index_path);
    fs_index->path = path_fmt_alloc_directory_fmt(path , true);
    free(path);
  }
  return fs_index;
}



static fs_index_node_type ** fs_index_node_list_fread_alloc(const char * index_file , int * _index_size) {
  int index_size = 0;
  fs_index_node_type ** node_list = NULL;
  if (util_file_exists(index_file)) {
    FILE * stream  = util_fopen(index_file , "r");
    int inode;
    index_size = util_fread_int(stream);
    node_list = util_malloc(index_size * sizeof * node_list , __func__);
    for (inode = 0; inode < index_size; inode++) 
      node_list[inode] = fs_index_node_fread_alloc(stream);
    fclose(stream);
  }
  *_index_size = index_size;
  return node_list;
}

static void fs_index_free_list(fs_index_node_type ** node_list , int index_size) {
  int inode;
  for (inode = 0; inode < index_size; inode++) 
    if (node_list[inode] != NULL) fs_index_node_free(node_list[inode]);
  free(node_list);
}



bool fs_index_has_node(fs_index_type *fs_index , int iens , const char *kw) {
  char * index_file = path_fmt_alloc_file(fs_index->path , iens , "index");
  int    index_size , inode;
  bool   has_node = false;
  fs_index_node_type **node_list = fs_index_node_list_fread_alloc(index_file , &index_size);
  for (inode = 0; inode < index_size; inode++) 
    if (strcmp(kw , node_list[inode]->kw) == 0) has_node = true;
  
  fs_index_free_list(node_list , index_size);
  free(index_file);
  return has_node;
}



void fs_index_add_node(fs_index_type *fs_index , int iens , const char *kw , enkf_var_type var_type , enkf_impl_type impl_type) {
  char * index_file = path_fmt_alloc_file(fs_index->path , iens , "index");
  int    index_size;
  fs_index_node_type **node_list = fs_index_node_list_fread_alloc(index_file , &index_size);
  {
    int inode;
    bool add_node = true;
    for (inode = 0; inode < index_size; inode++) 
      if (strcmp(kw , node_list[inode]->kw) == 0) 
	add_node = false;

    {
      FILE * stream = util_fopen(index_file , "w");
      if (add_node) 
	util_fwrite_int(index_size + 1 , stream);
      else
	util_fwrite_int(index_size     , stream);
      
      for (inode = 0; inode < index_size; inode++) 
	fs_index_node_fwrite(node_list[inode] , stream);
    
      if (add_node)
	fs_index_node_fwrite_data(kw , var_type , impl_type , stream);
      fclose(stream);
    }
  }
  
  fs_index_free_list(node_list , index_size);
  free(index_file);
}
 
 
 
 void fs_index_free(fs_index_type * fs_index) {
   path_fmt_free(fs_index->path);
   free(fs_index);
   fs_index = NULL;
 }
 

