# Math function example

This example contains some ways to use EVEREST to evaluate a simply objective function: the squared distance between two 3D points.

The purpose of the configuration files are meant to showcase the following:

- `config_minimal.yml`: A minimally functional configuration to run EVEREST with the distance3 job. It minimizes the distance between `point`, starting at (0,0,0), to the target of (0.5, 0.5, 0.5).

All other configuration files are extensions of the above configuration. These are the most important ones.
- `config_multiobj.yml`: Adds an additional objective function that checks the distance of the controls to a second target point (-1.5, -1.5, 0.5).
- `config_advanced.yml`: Adds both in- and output constraints to the configuration. Also adds realizations with differing weights to the model.
