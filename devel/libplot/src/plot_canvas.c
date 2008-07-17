#include <plot.h>
#include <plot_dataset.h>
#include <plot_canvas.h>

gboolean plot_canvas_data_join(gpointer data)
{

    plot_type *item = data;
    list_node_type *node, *next_node;
    int len;
    double *x;
    double *y;
    int step;
    int flag = true;

    node = list_get_head(plot_get_datasets(item));
    while (node != NULL) {
        plot_dataset_type *tmp;
        next_node = list_node_get_next(node);
        tmp = list_node_value_ptr(node);
        len = plot_datset_get_length(tmp);

        if (plot_dataset_is_finished(tmp)) {
            node = next_node;
            continue;
        }
        flag = false;

        if (plot_dataset_get_step(tmp) == len) {
            printf("ID[%d] Plotted last node, skipping..\n",
                   plot_get_stream(item));
            plot_dataset_finished(tmp, true);
            node = next_node;
            continue;
        }

        step = plot_dataset_step_next(tmp);

        x = plot_datset_get_vector_x(tmp);
        y = plot_datset_get_vector_y(tmp);
        plplot_canvas_col0(plot_get_canvas(item), plot_datset_get_color(tmp));

        plplot_canvas_join(plot_get_canvas(item), x[step - 1], y[step - 1],
                           x[step], y[step]);

        /* This is where the hog is it seems */
        plplot_canvas_adv(plot_get_canvas(item), 0);

        printf("ID[%d] Plotting step %d -> %d of total %d\n",
               plot_get_stream(item), step - 1, step, len);
        node = next_node;
    }

    if (!flag)
        return true;

    return false;
}

