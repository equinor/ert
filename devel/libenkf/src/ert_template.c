#include <template.h>
#include <stdbool.h>
#include <stdlib.h>
#include <vector.h>
#include <util.h>
#include <ert_template.h>

#define ERT_TEMPLATE_TYPE_ID  7731963
#define ERT_TEMPLATES_TYPE_ID 6677330

/* Singular - one template. */
struct ert_template_struct {
  UTIL_TYPE_ID_DECLARATION;
  template_type * template;
  char          * target_file;
};



/* Plural - many templates. */
struct ert_templates_struct {
  UTIL_TYPE_ID_DECLARATION;
  vector_type * templates;
};





ert_template_type * ert_template_alloc( const char * template_file , const char * target_file ) {
  ert_template_type * template = util_malloc( sizeof * template , __func__);
  UTIL_TYPE_ID_INIT(template , ERT_TEMPLATE_TYPE_ID);
  template->template    = template_alloc( template_file , true );
  template->target_file = util_alloc_string_copy( target_file );
  return template;
}


void ert_template_free( ert_template_type * template ) {
  free( template->target_file );
  template_free( template->template );
  free( template );
}


void ert_template_instantiate( ert_template_type * template , const char * path , const subst_list_type * arg_list) {
  char * target_file = util_alloc_filename( path , template->target_file , NULL );
  template_instansiate( template->template , target_file , arg_list );
  free( target_file );
}


void ert_template_add_arg( ert_template_type * template , const char * key , const char * value ) {
  template_add_arg( template->template , key , value );
}


UTIL_SAFE_CAST_FUNCTION( ert_template , ERT_TEMPLATE_TYPE_ID )

void ert_template_free__(void * arg) {
  ert_template_free( ert_template_safe_cast( arg ));
}


/*****************************************************************/


ert_templates_type * ert_templates_alloc( ) {
  ert_templates_type * templates = util_malloc( sizeof * templates , __func__);
  UTIL_TYPE_ID_INIT( templates , ERT_TEMPLATES_TYPE_ID );
  templates->templates = vector_alloc_new();
  return templates;
}


void ert_templates_free( ert_templates_type * ert_templates ) {
  vector_free( ert_templates->templates );
  free( ert_templates );
}


ert_template_type * ert_templates_add_template( ert_templates_type * ert_templates , const char * template_file , const char * target_file ) {
  ert_template_type * template = ert_template_alloc( template_file , target_file );
  vector_append_owned_ref( ert_templates->templates , template , ert_template_free__);
  return template;
}


void ert_templates_instansiate( ert_templates_type * ert_templates , const char * path , const subst_list_type * arg_list) {
  int i;
  for (i=0; i < vector_get_size(  ert_templates->templates ); i++) 
    ert_template_instantiate( vector_iget( ert_templates->templates , i ) , path , arg_list);
}
