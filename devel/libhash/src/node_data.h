#ifndef __NODE_DATA_H__
#define __NODE_DATA_H__
#ifdef __cplusplus
extern "C" {
#endif

typedef const void * (  copyc_type) (const void *);
typedef void         (  del_type)   (void *);



typedef struct node_data_struct node_data_type;


void             node_data_free(void *);
const void     * node_data_copyc(const void *);
const void     * node_data_get_data(const node_data_type *);
node_data_type * node_data_alloc(int  , const void * );

#ifdef __cplusplus
}
#endif
#endif
