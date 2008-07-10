#include <util.h>
#include <trs.h>
#include <stdbool.h>
#include <string.h>

typedef bool (trs_func_type)(const double *, const double *, int);
typedef struct trs_node_type_struct trs_node_type;


struct trs_node_type_struct
{
  trs_func_type * func;       /* Truncation function */
  double        * arg;        /* Arguments to func   */
  int             new_facies; /* New facies type on func true */
};



struct trs_type_struct
{
  int              num_nodes;
  int              num_gauss_fields;
  trs_node_type ** trs_nodes;
};



/*****************************************************************/



static trs_node_type * trs_node_alloc_empty()
{
  trs_node_type * trs_node = util_malloc(sizeof * trs_node, __func__);
  return trs_node;
};



void trs_node_free(trs_node_type * trs_node)
{
  free(trs_node->arg);
  free(trs_node);
};



static bool trs_func_linear(const double * gauss, const double * arg, int num_gauss_fields)
{
  int i;
  double val;

  val = 0.0;
  for(i=0; i<num_gauss_fields; i++)
    val = val + gauss[i] * arg[i];

  if(val >= arg[num_gauss_fields])
    return true;
  else
    return false;
}



/*
 Think of this as an "erosion".

 Given value of the Gaussian fields stored in the double pointer,
 and the current facies type stored in cur_facies, a new facies type is returned
 according to an erosion rule stored in node.
*/
static int trs_node_apply(trs_node_type *node, int cur_facies, const double * gauss, int num_gauss_fields)
{
  if(node->func(gauss, node->arg, num_gauss_fields))
    return node->new_facies;
  else
    return cur_facies;
}



static trs_node_type * trs_node_linear_alloc_from_string(const char * line, int new_facies , int num_gauss_fields)
{
  trs_node_type * trs_node = trs_node_alloc_empty();

  trs_node->new_facies = new_facies;
  trs_node->arg        = util_malloc( (num_gauss_fields + 1) * sizeof * trs_node->arg,__func__);
  trs_node->func       = trs_func_linear;

  {
    int i,tokens;
    char ** token_list;

    util_split_string(line, " ", &tokens, &token_list);

    if(tokens != num_gauss_fields + 1)
    {
      util_free_stringlist(token_list,tokens);
      trs_node_free(trs_node);
      util_abort("%s: Need exactly %i tokens for linear truncation with %i Gaussian fields, got %i tokens - aborting.\n",
                 __func__,num_gauss_fields+1,num_gauss_fields,tokens);
    }

    for(i=0; i<tokens; i++)
    {
      if(!util_sscanf_double(token_list[i],&(trs_node->arg[i])))
        util_abort("%s: Failed to read a double from %s - aborting.\n",__func__,token_list[i]);
    }

    util_free_stringlist(token_list,tokens);
  }
  return trs_node;
}


static trs_type * trs_alloc_empty(int num_gauss_fields, int num_nodes)
{
  trs_type * trs = util_malloc(sizeof * trs,__func__);

  trs->num_nodes        = num_nodes;
  trs->num_gauss_fields = num_gauss_fields;
  trs->trs_nodes        = util_malloc(num_nodes * sizeof * trs->trs_nodes,__func__);
  
  return trs;
}



/*****************************************************************/



void trs_free(trs_type * trs)
{
  int i;
  for(i=0; i<trs->num_nodes; i++)
    trs_node_free(trs->trs_nodes[i]);

  free(trs->trs_nodes);
  free(trs);
}



trs_type * trs_fscanf_alloc(const char * filename, const hash_type * facies_kw_hash, int num_gauss_fields)
{
  /*
    We can't use a config_type struct when allocating the truncation scheme from file. The reason being that
    the order in the file is of paramount importance.
  */
  
  bool ateof;
  int facies_int, num_nodes, num_alloced;

  FILE * stream;
  char * facies_kw;
  char * trs_kw;
  char * line;

  trs_type * trs;

  stream = util_fopen(filename,"r");
  num_nodes = util_count_file_lines(stream);

  trs = trs_alloc_empty(num_gauss_fields, num_nodes);
  
  ateof = false;
  num_alloced = 0;
  while(!ateof)
  {
    facies_kw = util_fscanf_alloc_token(stream);
    if(facies_kw == NULL)
    {
      ateof = true;
      break;
    }
    if(!hash_has_key(facies_kw_hash,facies_kw))
    {
      fclose(stream); 
      util_abort("%s: Unknown facies type: %s - aborting.\n",__func__,facies_kw);
    }
    
    facies_int = hash_get_int(facies_kw_hash, facies_kw);

    trs_kw = util_fscanf_alloc_token(stream);
    if(trs_kw == NULL)
    {
      ateof = true;
      break;
    }
    
    line = util_fscanf_alloc_line(stream,&ateof);
    if(strcmp(trs_kw,"LINEAR") == 0)
    {
      trs->trs_nodes[num_alloced] = trs_node_linear_alloc_from_string(line, facies_int, num_gauss_fields);
      num_alloced++;
    }
    else
    {
      fclose(stream); 
      util_abort("%s: Unknown truncation scheme: %s - aborting.\n",__func__,trs_kw);
    }

    free(line);
    free(trs_kw);
    free(facies_kw);

  }
  fclose(stream); 

  if(num_alloced != num_nodes)
    util_abort("%s: Something is wrong - number of alloc'd nodes does not equal the number of lines in file %s - aborting.\n",__func__,filename);

  return trs;
};



int trs_apply_block(trs_type * trs, const double * gauss)
{
  int i, facies;

  facies = 0;
  for(i=0; i<trs->num_nodes; i++)
  {
    facies = trs_node_apply(trs->trs_nodes[i], facies, gauss, trs->num_gauss_fields);
  }
  return facies;
}



int * trs_apply_alloc(trs_type * trs, const double ** gauss_fields, int size)
{
  int i,j;
  int    * facies = util_malloc(size * sizeof * facies               , __func__);
  double * gauss  = util_malloc(trs->num_gauss_fields *sizeof * gauss, __func__);

  for(i=0; i<size; i++)
  {
    for(j=0; j<trs->num_gauss_fields; j++)
      gauss[j] = gauss_fields[j][i];

    facies[i] = trs_apply_block(trs, gauss);
  }


  free(gauss);
  return facies;
}
