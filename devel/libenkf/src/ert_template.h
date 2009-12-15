#ifndef __ERT_TEMPLATE_H__
#define __ERT_TEMPLATE_H__

#include <subst_list.h>

typedef struct ert_template_struct  ert_template_type;
typedef struct ert_templates_struct ert_templates_type;


ert_template_type * ert_template_alloc( const char * template_file , const char * target_file, subst_list_type * parent_subst) ;
void                ert_template_free( ert_template_type * ert_tamplete );
void                ert_template_instantiate( ert_template_type * ert_template , const char * path , const subst_list_type * arg_list ); 
void                ert_template_add_arg( ert_template_type * template , const char * key , const char * value );
void                ert_template_free__(void * arg);


ert_templates_type * ert_templates_alloc(subst_list_type * parent_subst);
void                 ert_templates_free( ert_templates_type * ert_templates );
ert_template_type  * ert_templates_add_template( ert_templates_type * ert_templates , const char * template_file , const char * target_file );
void                 ert_templates_instansiate( ert_templates_type * ert_templates , const char * path , const subst_list_type * arg_list);

#endif
