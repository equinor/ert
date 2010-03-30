def resolveHelpLabel(label):
    glbs = globals()
    label = "help_" + label
    if label in glbs:
        return glbs[label]

    return ""

help = "Not defined."
help_plot_path = "The plotting engine creates 'files' with plots, they are stored in a directory. You can tell what that directory should be.\nObserve that the current 'casename' will automatically be appended to the plot path."
help_plot_width = "When the PLPLOT driver creates a plot file, it will have the width (in pixels) given by the PLOT_WIDTH keyword. The default value for PLOT_WIDTH is 1024 pixels."