## A reservoir example based on SPE1

The motivation of this example is to have a plain and simple reservoir setup
that highlights the features of ert3 in a reservoir context.

#### Requirements

 - ert3: In Equinor you can fetch an installation via Komodo, but
   in general a `pip install ert` should suffice.
 - flow: A functioning installation of OPM Flow with a binary named `flow`.
 - ecl2df: In addition to the ERT dependencies the job `summary2json` utilises
   ecl2df. It can be installed via `pip install ecl2df`.
 - ert-storage: To store results, ert3 utilizes a service called ert-storage.
   It can be installed via `pip install ert[storage]`.

### Overview of the workspace

#### Parameters
The available parameterizations of the model resides within
[parameters.yml](parameters.yml). In general a parameter group is given a name,
which can be referenced later in experiments, a distribution (`uniform` or
`gauss`) and a list of variables that are drawn from the distribution.

#### Stages
The forward model in ert3 is a stage and is described in
[stages.yml](stages.yml). Currently the only available stage is a unix step,
which gets input data as files on disk, runs a script and then is to produce
output data in files on disk again.

Each stage is given a name and is in addition to:
 - direct input records into files,
 - point to files which are to be loaded as output,
 - give commands that can be transported to the compute side and
 - a script that is to be executed.
When the script starts it can assume the input data to be present in the
specified files and when it finishes it should have produced the expected data
in the specified output files.

Note that the motivation for keeping both stages and parameters in the global workspace
is so that experiments can share them.

#### The datafile
The datafile is based on an example from [OPM test
data](https://github.com/OPM/opm-tests/blob/master/spe1/SPE1CASE2.DATA). It is
written as a Jinja2 template and resides in `resources/SPE1CASE2.DATA.jinja2`.

The datafile aims at simulating a reservoir for 10 years with two wells named
`PROD` and `INJ`.

The template is templated by the following parameters:
 - `field_properties.porosity`: The uniform porosity of the model
 - `field_properties.x_mid_permability`: The permeability of the middle layer of
   cells in X direction and
 - `wells.delay`: The delay (must be between 1 and 365 days) before the wells
   are opened.

### The experiments

#### Ert-storage
The ert-storage service needs to be available. To launch a local instance use the
command `ert3 service start storage` in a separate terminal. If you are unsure
whether you have an instance available, you can verify by running
`ert3 service check storage`.

#### Layout
Each experiment is a folder within within the `experiments` folder. The name of
the folder is the name of the experiment and each folder is expected to contain
two files, namely `experiment.yml` and `ensemble.yml`.

#### Initialising
A workspace is only a collection of files until it is initialised as a
workspace. This is a one-time operation carried out from the root of the
workspace running the command `ert3 init`.

#### Evaluation
The first experiment of the workspace is the evaluation experiment. It runs an
evaluation of the model described in the respective `ensembles.yml`. The
`experiment.yml` is the file indicating that the experiment is indeed an
evaluation. The `ensemble.yml` describes the size of the ensemble, maps data
sources to input records and specifies the stage that is the forward model
together with the queue system that is to be used. Notice in particular that by
`stochastic.field_properties` one is pointing at the parameter group named
`field_properties` in `parameters.yml`.

The experiment can be executed by `ert3 run evaluation`. And the data can
afterwards be exported using `ert3 export evaluation`, that will put the data
in the file `experiments/evaluation/data.json`.

#### Design of experiment
The next experiment in the workspace carries out a design of experiment. The
important differences to notice is that there are two files named
`field_properties.json` and `wells.json` containing the parameter values chosen
for the experiment. Both of these can be loaded via the `ert3 record load`
command. Use `ert3 record load --help` for assistance or look at the `run_demo`
script in the workspace root. Two important things to notice is that the length
of the list of records in each of the files must align with the size of the
ensemble configured. Furthermore, the input records are now fetched via
`storage.designed_field_properties`. Indicating that the records come from
storage. The name of the record given when loading it must be the same as the
one referred to in the ensemble.

#### Sensitivity analysis
The last experiment is a sensitivity study. Again it refers to distributions
from the `parameters.yml` and uses the distributions to design the different
realisations. An important difference is that the ensemble size is no longer
configured by the user, but dictated by the algorithm. There are also some
differences in the `experiment.yml`, but those are hopefully self-explaining.

#### Clean already ran experiments
An experiment can only be run once. If you wish re-run your experiment, the
results must be deleted first. To do this, execute `ert3 clean <experiment_name>`.
To see the status of all the experiments in the workspace, run `ert3 status`.

#### Visualize experiments
After experiments have been evaluated. The records labled as output in the
ensemble-config can be visualized through the visualization solution, `webviz-ert`.
To launch the visualization tool run `ert vis`.

### Some additional notes
ERT3 is highly experimental software in the sense that we aim at moving fast.
Natural features that are lacking, are among others:
 - inspect records and experiments in the cli,
 - delete workspace records,
 - monitor ongoing experiments,
 - fetch experiment results via an API and
 - cross configuration file validation (each file is validated in isolation, but not
   towards each other).

The second point are currently resolved by deleting the `.ert` folder in
the workspace and starting over. Monitoring is currently via extensive output
in the terminal. Results can fetched via the export command.

Notice that the `.ert` folder is to contain data that the user should not
inspect, assume to be present nor change directly. It currently contains the
runpath, but remember that this will not be available over time. So we should
mature other means for debugging, etc.
