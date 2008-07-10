#ifndef __TPGZONE_H__
#define __TPGZONE_H__

typedef struct tpgzone_struct tpgzone_type;

tpgzone_type * tpgzone_fscanf_alloc(char *, bool);
void tpgzone_free(tpgzone_type *);
#endif
