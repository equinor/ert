#ifndef __TPGZONE_H__
#define __TPGZONE_H__

/*
  ECLIPSE keyword from which the gaussian fields are loaded.
*/
#define TPGZONE_GAUSS_ECL_KW "TPGGAUSS"

typedef struct tpgzone_struct tpgzone_type;

tpgzone_type * tpgzone_fscanf_alloc(char *, bool);
void tpgzone_free(tpgzone_type *);

void tpgzone_apply(const tpgzone_type *, bool);
#endif
