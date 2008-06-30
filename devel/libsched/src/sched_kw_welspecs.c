typedef enum {OIL,WATER,GAS,LIQ} phase_type;

typedef struct
{
  char * name;
  bool   has_group;
  char * group;
  int    i;
  int    j;
  double md;
  phase_type phase;
} welspecs_type;
