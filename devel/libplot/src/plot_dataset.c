#include "plot.h"
#include "plot_dataset.h"

plot_dataset *plot_dataset_alloc() {
     plot_dataset *d;

     d = malloc(sizeof *d);
     if(!d)
       return NULL;

     return d;
}

void plot_dataset_free(plot_dataset *d) {
     util_safe_free(d->xvalue);
     util_safe_free(d->yvalue);
     util_safe_free(d);
}

void plot_dataset_set_data(plot_dataset *d, double *x, double *y, int len, plot_color c, plot_style s) {
     if (!d) {
	  fprintf(stderr, "Error: you need to allocate the new dataset first\n");
	  return;
     }

     d->xvalue = x;
     d->yvalue = y;
     d->length = len;
     d->color = c;
     d->style = s;
}

int plot_dataset_add(plot *item, plot_dataset *d) {
     if (!d || !item) {
	  fprintf(stderr, "Error: you need to allocate a new dataset or plot-item.\n");
	  return false;
     }

     if (!d->xvalue || !d->yvalue || !d->length) {
	  fprintf(stderr, "Error: you need to set the data first\n");
	  return false;
     }
     
     printf("%s: Adding dataset %p to list with length %d\n", __func__, d, d->length);
     list_append_ref(item->datasets, d);

     return true;
}


