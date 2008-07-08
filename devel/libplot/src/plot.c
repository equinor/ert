#include "plot.h"

plot_type *plot_alloc() {
     plot_type *p;

     p = malloc(sizeof * p);
     if (!p)
       return NULL;

     p->plots = list_alloc();

     return p;
}

void plot_free(plot_type *p) {
     list_node_type *node , *next_node;
     bool flag = false;

     if (!p) 
       return;
     
     node = list_get_head(p->plots);
     while (node != NULL) {
	  plot_item *tmp;
	  flag = true;
	  next_node = list_node_get_next(node);
	  tmp = list_node_value_ptr(node);

	  __plot_item_free(p->plots, tmp);
	  node = next_node;
     }

     list_free(p->plots);

     util_safe_free(p);

     /* Only end if there has been no plot windows */
     if (!flag)
       plend();
}

int plot_init(plot_type *p, bool debug, int argc, const char **argv) {
     char ver[80];
     int j;
     j = plparseopts(&argc, argv, PL_PARSE_FULL);

     p->debug = debug;

     if (debug) {
	plgver(ver);
	fprintf(stdout, "PLplot library version: %s\n", ver);
     }

     return j;
}


