class Plotter:
    def __init__(self):
        pass

    def plot(self, axes, plot_config, x, y):
        line = axes.plot(x,
                         y,
                         plot_config.style,
                         color=plot_config.color,
                         alpha=plot_config.alpha,
                         zorder=plot_config.z_order,
                         picker = plot_config.picker,
                         visible = plot_config.is_visible and plot_config.hasStyle())

        return line[0]

    def plot_date(self, axes, plot_config, x, y):
        line = axes.plot_date(x,
                              y,
                              plot_config.style,
                              color=plot_config.color,
                              alpha=plot_config.alpha,
                              zorder=plot_config.z_order,
                              picker = plot_config.picker,
                              visible = plot_config.is_visible and plot_config.hasStyle())

        return line[0]

    def plot_errorbar(self, axes, plot_config, x, y, std_x, std_y):
        axes.errorbar(x,
                      y,
                      yerr = std_y,
                      xerr = std_x,
                      fmt=None,
                      ecolor=plot_config.color,
                      alpha=plot_config.alpha,
                      zorder=plot_config.z_order,
                      visible = plot_config.is_visible)
