#ifndef __RELPERM_CONFIG_H__
#define __RELPERM_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <scalar.h>
#include <scalar_config.h>
#include <hash.h>

typedef struct table_struct table_type;
typedef struct table_config_struct table_config_type;

typedef enum {COREY} func_type;
typedef enum {SWOF,SGOF,SLGOF,SWFN,SGFN,SOF2,SOF3,SOF32D} relptab_kw_type;

struct table_config_struct{
  char * eclipse_file;          /* The relperm table file name that should be specified in the Eclipse DATA file */ 
  relptab_kw_type relptab_kw;   /* SWOF, SLGOF = Famnr 1.  SWFN, SGFN, SOF2, SOF3 = Famnr 2:*/
  func_type func;               /* Relperm parameterisation function: COREY */
  table_type * tab;             /* Holds the specific keywords defined for each table in relperm_table.txt*/
  bool ecl_file_append;         /* Flag that tells if the eclipse_file name is used more than once in relperm_table.txt */
};

struct table_struct{
  char * swco;                 /* Connate water saturation */
  char * soco;                 /* Connate oil saturation */
  char * sgco;                 /* Connate gas saturation */
  char * sorg;                 /* Famnr1: Not used. Famnr2: The Residual Oil Saturation in the oil-water system*/
  char * ewat;                 /* Corey exponent, water */
  char * egas;                 /* Corey exponent, gas */
  char * eowa;                 /* Corey exponent, oil - water */
  char * eogw;                 /* Corey exponent, oil - gas, connate water*/
  char * scwa;                 /* Right endpoint of relperm, water*/
  char * scga;                 /* Right endpoint of relperm, gas   */
  char * scoi;                 /* Right endpoint of relperm, oil*/
  char * pewa;                 /* Capillary pressure water/oil at maximum water saturation*/
  char * pega;                 /* Capillary pressure gas/oil at maximum gas saturation*/
};

typedef struct {
  int             __type_id;
  char         ** kw_list;  /* A list of different keywords defined in column number 1 in relperm_config.txt */
  int            nso;       /* The resoultion of the relperm tables which include oil-saturation */
  int            nsw;       /* The resoultion of the relperm tables which include water-saturation */
  int            nsg;       /* The resoultion of the relperm tables which include gas-saturation */
  int          famnr;       /* The relperm family number. Family number: 1=SWOF,SLGOF,SGOF or 2=SWFN,SGFN,SOF3,SOF2 */
  int          num_tab;     /* Number of relperm tables that should be generated */

  enkf_var_type var_type;
  scalar_config_type * scalar_config;
  hash_type * index_hash;
  hash_type * ecl_file_hash;
  table_config_type ** table_config;
} relperm_config_type;

relperm_config_type * relperm_config_fscanf_alloc(const char *, const char *);
void relperm_config_ecl_write_table(const relperm_config_type *, const double *, const char *);
void relperm_config_ecl_write(const relperm_config_type *, const double *, FILE *,const char *);
void relperm_config_ecl_write_swof(FILE * ,const table_type  *, hash_type *, const double *, int, bool,func_type);
void relperm_config_ecl_write_sgof(FILE * ,const table_type  *, hash_type *, const double *, int, bool, func_type);
void relperm_config_ecl_write_slgof(FILE * ,const table_type *, hash_type *, const double *,int,bool,func_type);
void relperm_config_ecl_write_swfn(FILE * ,const table_type  *, hash_type *, const double *, int , bool,func_type);
void relperm_config_ecl_write_sgfn(FILE * ,const table_type  *, hash_type *, const double *, int, bool,func_type);
void relperm_config_ecl_write_sof2(FILE * ,const table_type  *, hash_type *, const double *);
void relperm_config_ecl_write_sof3(FILE * ,const table_type  *, hash_type *, const double *,int, bool,func_type);
table_config_type *  relperm_config_fscanf_table_config_alloc(FILE *,hash_type *);
table_type * relperm_config_table_alloc(char **, int);
/* Method that test if the table input in relperm_tab.txt match the input in relperm_config.txt*/
void relperm_config_check_tab_input(hash_type *,const table_config_type *,const int); 
void relperm_config_check_data(const table_type *, hash_type *, const double *, int);
bool relperm_config_check_ecl_file(hash_type *, char *, relptab_kw_type);
relptab_kw_type relperm_config_set_relptab_kw(char *);
void relperm_config_free(relperm_config_type *);
void relperm_config_table_config_free(table_config_type **, int);
void relperm_config_table_free(table_type *);

/*Generated headers */
GET_DATA_SIZE_HEADER(relperm);
VOID_FREE_HEADER(relperm_config);
SAFE_CAST_HEADER(relperm_config);

#ifdef __cplusplus
}
#endif
#endif
