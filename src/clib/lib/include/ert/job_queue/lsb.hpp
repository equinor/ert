#ifndef ERT_LSB_H
#define ERT_LSB_H

#include <stdbool.h>

#include <lsf/lsbatch.h>

#include <ert/util/stringlist.hpp>

typedef struct lsb_struct lsb_type;

lsb_type *lsb_alloc();
void lsb_free(lsb_type *lsb);
bool lsb_ready(const lsb_type *lsb);

int lsb_initialize(const lsb_type *lsb);
int lsb_submitjob(const lsb_type *lsb, struct submit *, struct submitReply *);
int lsb_killjob(const lsb_type *lsb, int lsf_jobnr);
int lsb_openjob(const lsb_type *lsb, int lsf_jobnr);
struct jobInfoEnt *lsb_readjob(const lsb_type *lsb);
int lsb_closejob(const lsb_type *lsb);
char *lsb_sys_msg(const lsb_type *lsb);
stringlist_type *lsb_get_error_list(const lsb_type *lsb);

#endif
