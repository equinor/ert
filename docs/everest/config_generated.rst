model (optional)
----------------
Type: *Optional[ModelConfig]*

Configuration of the Everest model

**realizations (optional)**
    Type: *List[NonNegativeInt]*

    List of realizations to use in optimization ensemble.

    Typically, this is a list [0, 1, ..., n-1] of all realizations in the ensemble.


**data_file (optional)**
    Type: *Optional[str]*

    Path to the eclipse data file used for optimization.
    The path can contain r{{geo_id}}.

    NOTE: Without a data file no well or group specific summary data will be exported.


**realizations_weights (optional)**
    Type: *Optional[List[float]]*

    List of weights, one per realization.

    If specified, it must be a list of numeric values, one per realization.


**report_steps (optional)**
    Type: *Optional[List[str]]*

    List of dates allowed in the summary file.



controls (required)
-------------------
Type: *List[ControlConfig]*

Defines a list of controls.
Controls should have unique names each control defines
a group of control variables


**name (required)**
    Type: *str*

    Control name


**type (required)**
    Type: *Literal['well_control', 'generic_control']*


    Only two allowed control types are accepted

    * **well_control**: Standard built-in Everest control type designed for field optimization

    * **generic_control**: Enables the user to define controls types to be employed for customized optimization jobs.


**variables (required)**
    Type: *Union[List[ControlVariableConfig], List[ControlVariableGuessListConfig]]*

    List of control variables

    **name (required)**
        Type: *str*

        Control variable name


    **control_type (optional)**
        Type: *Optional[Literal['real', 'integer']]*


        The type of control. Set to "integer" for discrete optimization. This may be
        ignored if the algorithm that is used does not support different control types.


    **enabled (optional)**
        Type: *Optional[bool]*


        If `True`, the variable will be optimized, otherwise it will be fixed to the
        initial value.


    **auto_scale (optional)**
        Type: *Optional[bool]*


        Can be set to true to re-scale variable from the range
        defined by [min, max] to the range defined by scaled_range (default [0, 1])


    **scaled_range (optional)**
        Type: *Optional[Tuple[float, float]]*


        Can be used to set the range of the variable values
        after scaling (default = [0, 1]).

        This option has no effect if auto_scale is not set.


    **min (optional)**
        Type: *Optional[float]*


        Minimal value allowed for the variable

        initial_guess is required to be greater than this value.


    **max (optional)**
        Type: *Optional[float]*


        Max value allowed for the variable

        initial_guess is required to be less than this value.



    **perturbation_magnitude (optional)**
        Type: *Optional[PositiveFloat]*


        Specifies the perturbation magnitude for this particular variable.
        This feature adds flexibility to combine controls into more logical
        of structures at the same time allowing the variable to contain time (
        how long rate applies for) & value (the actual rate).

        NOTE: In most cases this should not be configured, and the default value should be used.


    **sampler (optional)**
        Type: *Optional[SamplerConfig]*

        The backend used by Everest for sampling points

        **backend (optional)**
            Type: *str*

            The backend used by Everest for sampling points.

            The sampler backend provides the methods for sampling the points used to
            estimate the gradient. The default is the built-in 'scipy' backend.



        **options (optional)**
            Type: *Optional[Mapping[str, Any]]*


            Specifies a dict of optional parameters for the sampler backend.

            This dict of values is passed unchanged to the selected method in the backend.



        **method (optional)**
            Type: *str*

            The sampling method or distribution used by the sampler backend.


        **shared (optional)**
            Type: *Optional[bool]*

            Whether to share perturbations between realizations.



    **initial_guess (optional)**
        Type: *List[float]*

        List of Starting values for the control variable



**initial_guess (optional)**
    Type: *Optional[float]*


    Initial guess for the control group all control variables with initial_guess not
    defined will be assigned this value. Individual initial_guess values in the control
    variables will overwrite this value.


**control_type (optional)**
    Type: *Literal['real', 'integer']*


    The type of the controls for the control group. Individual control types in the
    control variables will override this value. Set to "integer" for discrete
    optimization. This may be ignored if the algorithm that is used does not support
    different control types.


**enabled (optional)**
    Type: *Optional[bool]*


    If `True`, all variables in this control group will be optimized. If set to `False`
    the value of the variables will remain fixed.


**auto_scale (optional)**
    Type: *bool*


    Can be set to true to re-scale controls from the range
    defined by [min, max] to the range defined by
    scaled_range (default [0, 1]).



**min (optional)**
    Type: *Optional[float]*


    Defines left-side value in the control group range [min, max].
    This value will be overwritten by the control variable min value if given.

    The initial guess for both the group and the individual variables needs to be contained
    in the resulting [min, max] range


**max (optional)**
    Type: *Optional[float]*


    Defines right-side value in the control group range [min, max].
    This value will be overwritten by the control variable max value if given.

    The initial guess for both the group and the individual variables needs to be contained
    in the resulting [min, max] range


**perturbation_type (optional)**
    Type: *Literal['absolute', 'relative']*


    Example: absolute or relative
    Specifies the perturbation type for a set of controls of a certain type. The
    perturbation type keyword defines whether the perturbation magnitude
    (perturbation_magnitude) should be considered as an absolute value or relative
    to the dynamic range of the controls.

    NOTE: currently the dynamic range is computed with respect to all controls, so
    defining relative perturbation type for control types with different dynamic
    ranges might have unintended effects.



**perturbation_magnitude (optional)**
    Type: *Optional[float]*


    Specifies the perturbation magnitude for a set of controls of a certain type.

    This controls the size of perturbations (standard deviation of a
    normal distribution) of controls used to approximate the gradient.
    The value depends on the type of control and magnitude of the variables.
    For continuous controls smaller values should give a better gradient,
    whilst for more discrete controls larger values should give a better
    result. However, this is a balance as too large or too small
    of values also cause issues.

    NOTE: In most cases this should not be configured, and the default value should be used.



**scaled_range (optional)**
    Type: *Optional[Tuple[float, float]]*


    Can be used to set the range of the control values
    after scaling (default = [0, 1]).

    This option has no effect if auto_scale is not set.



**sampler (optional)**
    Type: *Optional[SamplerConfig]*


    A sampler specification section applies to a group of controls, or to an
    individual control. Sampler specifications are not required, with the
    following behavior, if no sampler sections are provided, a normal
    distribution is used.

    If at least one control group or variable has a sampler specification, only
    the groups or variables with a sampler specification are perturbed.
    Controls/variables that do not have a sampler section will not be perturbed
    at all. If that is not desired, make sure to specify a sampler for each
    control group and/or variable (or none at all to use a normal distribution
    for each control).

    Within the sampler section, the *shared* keyword can be used to direct the
    sampler to use the same perturbations for each realization.


    **backend (optional)**
        Type: *str*

        The backend used by Everest for sampling points.

        The sampler backend provides the methods for sampling the points used to
        estimate the gradient. The default is the built-in 'scipy' backend.



    **options (optional)**
        Type: *Optional[Mapping[str, Any]]*


        Specifies a dict of optional parameters for the sampler backend.

        This dict of values is passed unchanged to the selected method in the backend.



    **method (optional)**
        Type: *str*

        The sampling method or distribution used by the sampler backend.


    **shared (optional)**
        Type: *Optional[bool]*

        Whether to share perturbations between realizations.




optimization (optional)
-----------------------
Type: *Optional[OptimizationConfig]*

Optimizer options

**algorithm (optional)**
    Type: *Optional[str]*

    Algorithm used by Everest. Defaults to
    optpp_q_newton, a quasi-Newton algorithm in Dakota's OPT PP library.


**convergence_tolerance (optional)**
    Type: *Optional[float]*

    Defines the threshold value on relative change
    in the objective function that indicates convergence.

    The convergence_tolerance specification provides a real value for controlling
    the termination of iteration. In most cases, it is a relative convergence tolerance
    for the objective function; i.e., if the change in the objective function between
    successive iterations divided by the previous objective function is less than
    the amount specified by convergence_tolerance, then this convergence criterion is
    satisfied on the current iteration.

    Since no progress may be made on one iteration followed by significant progress
    on a subsequent iteration, some libraries require that the convergence tolerance
    be satisfied on two or more consecutive iterations prior to termination of
    iteration.

    (From the Dakota Manual.)


**backend (optional)**
    Type: *Optional[str]*

    The optimization backend used. Defaults to "dakota".

    Currently, backends are included to use Dakota or SciPy ("dakota" and "scipy").
    The Dakota backend is the default, and can be assumed to be installed. The SciPy
    backend is optional, and will only be available if SciPy is installed on the
    system.


**backend_options (optional)**
    Type: *Optional[Mapping[str, Any]]*

    Dict of optional parameters for the optimizer backend.
    This dict of values is passed unchanged to the selected algorithm in the backend.

    Note that the default Dakota backend ignores this option, because it requires a
    list of strings rather than a dictionary. For setting Dakota backend options, see
    the 'option' keyword.


**constraint_tolerance (optional)**
    Type: *Optional[float]*

    Determines the maximum allowable value of
    infeasibility that any constraint in an optimization problem may possess and
    still be considered to be satisfied.

    It is specified as a positive real value. If a constraint function is greater
    than this value then it is considered to be violated by the optimization
    algorithm. This specification gives some control over how tightly the
    constraints will be satisfied at convergence of the algorithm. However, if the
    value is set too small the algorithm may terminate with one or more constraints
    being violated.

    (From the Dakota Manual.)


**cvar (optional)**
    Type: *Optional[CVaRConfig]*

    Directs the optimizer to use CVaR estimation.

    When this section is present Everest will use Conditional Value at Risk (CVaR)
    to minimize risk. Effectively this means that at each iteration the objective
    and constraint functions will be calculated as the mean over the sub-set of the
    realizations that perform worst. The size of this set is specified as an
    absolute number or as a percentile value. These options are selected by setting
    either the **number_of_realizations** option, or the **percentile** option,
    which are mutually exclusive.

    **number_of_realizations (optional)**
        Type: *Optional[int]*

        The number of realizations used for CVaR estimation.

        Sets the number of realizations that is used to calculate the total objective.

        This option is exclusive with the **percentile** option.


    **percentile (optional)**
        Type: *Optional[float]*

        The percentile used for CVaR estimation.

        Sets the percentile of distribution of the objective over the realizations that
        is used to calculate the total objective.

        This option is exclusive with the **number_of_realizations** option.





**max_batch_num (optional)**
    Type: *Optional[int]*

    Limits the number of batches of simulations
    during optimization, where 0 represents unlimited simulation batches.
    When max_batch_num is specified and the current batch index is greater than
    max_batch_num an exception is raised.


**max_function_evaluations (optional)**
    Type: *Optional[int]*

    Limits the maximum number of function evaluations.

    The max_function_evaluations controls the number of control update steps the optimizer
    will allow before convergence is obtained.

    See max_iterations for a description.


**max_iterations (optional)**
    Type: *Optional[int]*

    Limits the maximum number of iterations.

    The difference between an iteration and a batch is that an iteration corresponds to
    a complete accepted batch (i.e., a batch that provides an improvement in the
    objective function while satisfying all constraints).


**min_pert_success (optional)**
    Type: *Optional[int]*

    specifies the minimum number of successfully completed
    evaluations of perturbed controls required to compute a gradient. The optimization
    process will stop if this minimum is not reached, and otherwise a gradient will be
    computed based on the set of successful perturbation runs. The minimum is checked for
    each realization individually.

    A special case is robust optimization with `perturbation_num: 1`. In that case the
    minimum applies to all realizations combined. In other words, a robust gradient may then
    still be computed based on a subset of the realizations.

    The user-provided value is reset to perturbation_num if it is larger than this number
    and a message is produced. In the special case of robust optimization case with
    `perturbation_num: 1` the maximum allowed value is the number of realizations specified
    by realizations instead.


**min_realizations_success (optional)**
    Type: *Optional[int]*

    Minimum number of realizations

    The minimum number of realizations that should be available for the computation
    of either expected function values (both objective function and constraint
    functions) or of the expected gradient. Note that this keyword does not apply
    to gradient computation in the robust case with 1 perturbation in which the
    expected gradient is computed directly.

    The optimization process will stop if this minimum is not reached, and otherwise
    the expected objective function value (and expected gradient/constraint function
    values) will be computed based on the set of successful contributions. In other
    words, a robust objective function, a robust gradient and robust constraint
    functions may then still be computed based on a subset of the realizations.

    The user-provided value is reset to the number of realizations specified by
    realizations if it is larger than this number and a message is produced.

    Note that it is possible to set the minimum number of successful realizations equal
    to zero. Some optimization algorithms are able to handle this and will proceed even
    if all realizations failed. Most algorithms are not capable of this and will adjust
    the value to be equal to one.


**options (optional)**
    Type: *Optional[List[str]]*

    specifies non-validated, optional
    passthrough parameters for the optimizer

    | Examples used are
    | - max_repetitions = 300
    | - retry_if_fail
    | - classical_search 1


**perturbation_num (optional)**
    Type: *Optional[int]*

    The number of perturbed control vectors per realization.

    The number of simulation runs used for estimating the gradient is equal to the
    the product of perturbation_num and model.realizations.


**speculative (optional)**
    Type: *Optional[bool]*

    specifies whether to enable speculative computation.

    The speculative specification enables speculative computation of gradient and/or
    Hessian information, where applicable, for parallel optimization studies. By
    speculating that the derivative information at the current point will be used
    later, the complete data set (all available gradient/Hessian information) can be
    computed on every function evaluation. While some of these computations will be
    wasted, the positive effects are a consistent parallel load balance and usually
    shorter wall clock time. The speculative specification is applicable only when
    parallelism in the gradient calculations can be exploited by Dakota (it will be
    ignored for vendor numerical gradients). (From the Dakota Manual.)


**parallel (optional)**
    Type: *Optional[bool]*

    whether to allow parallel function evaluation.

    By default Everest will evaluate a single function and gradient evaluation at
    a time. In case of gradient-free optimizer this can be highly inefficient,
    since these tend to need many independent function evaluations at each
    iteration. By setting parallel to True, multiple functions may be evaluated in
    parallel, if supported by the optimization algorithm.

    The default is to use parallel evaluation if supported.



objective_functions (required)
------------------------------
Type: *List[ObjectiveFunctionConfig]*

List of objective function specifications

**name (required)**
    Type: *str*




**alias (optional)**
    Type: *Optional[str]*


    alias can be set to the name of another objective function, directing everest
    to copy the value of that objective into the current objective. This is useful
    when used together with the **type** option, for instance to construct an objective
    function that consist of the sum of the mean and standard-deviation over the
    realizations of the same objective. In such a case, add a second objective with
    **type** equal to "stddev" and set **alias** to the name of the first objective to make
    sure that the standard deviation is calculated over the values of that objective.


**weight (optional)**
    Type: *Optional[PositiveFloat]*


    weight determines the importance of an objective function relative to the other
    objective functions.

    Ultimately, the weighted sum of all the objectives is what Everest tries to optimize.
    Note that, in case the weights do not sum up to 1, they are normalized before being
    used in the optimization process.


**normalization (optional)**
    Type: *Optional[float]*


    normalization is a multiplication factor defined per objective function.

    The value of each objective function is multiplied by the related normalization value.
    When optimizing with respect to multiple objective functions, it is important
    that the normalization is set so that all the normalized objectives have the same order
    of magnitude. Ultimately, the normalized objectives are used in computing
    the weighted sum that Everest tries to optimize.


**auto_normalize (optional)**
    Type: *Optional[bool]*


    auto_normalize can be set to true to automatically
    determine the normalization factor from the objective value in batch 0.

    If normalization is also set, the automatic value is multiplied by its value.


**type (optional)**
    Type: *Optional[str]*


    type can be set to the name of a method that should be applied to calculate a
    total objective function from the objectives obtained for all realizations.
    Currently, the only values supported are "mean" and "stddev", which calculate
    the mean and the negative of the standard deviation over the realizations,
    respectively. The negative of the standard deviation is used, since in general
    the aim is to minimize the standard deviation as opposed to the mean, which is
    preferred to be maximized.




environment (optional)
----------------------
Type: *Optional[EnvironmentConfig]*

The environment of Everest, specifies which folders are used for simulation and output, as well as the level of detail in Everest-logs

**simulation_folder (optional)**
    Type: *Optional[str]*

    Folder used for simulation by Everest


**output_folder (optional)**
    Type: *Optional[str]*

    Folder for outputs of Everest


**log_level (optional)**
    Type: *Optional[Literal['debug', 'info', 'warning', 'error', 'critical']]*

    Defines the verbosity of logs output by Everest.

    The default log level is `info`. All supported log levels are:

    debug: Detailed information, typically of interest only when diagnosing
    problems.

    info: Confirmation that things are working as expected.

    warning: An indication that something unexpected happened, or indicative of some
    problem in the near future (e.g. `disk space low`). The software is still
    working as expected.

    error: Due to a more serious problem, the software has not been able to perform
    some function.

    critical: A serious error, indicating that the program itself may be unable to
    continue running.


**random_seed (optional)**
    Type: *Optional[int]*

    Random seed (must be positive)



wells (optional)
----------------
Type: *List[WellConfig]*

A list of well configurations, all with unique names.

**name (required)**
    Type: *str*

    The unique name of the well


**drill_date (optional)**
    Type: *Optional[str]*

    Ideal date to drill a well.

    The interpretation of this is up to the forward model. The standard tooling will
    consider this as the earliest possible drill date.


**drill_time (optional)**
    Type: *Optional[float]*

    specifies the time it takes
    to drill the well under consideration.



input_constraints (optional)
----------------------------
Type: *Optional[List[InputConstraintConfig]]*

List of input constraints

**weights (required)**
    Type: *Mapping[str, float]*

    **Example**
    If we are trying to constrain only one control (i.e the z control) value:
    | input_constraints:
    | - weights:
    |   point_3D.x-0: 0
    |   point_3D.y-1: 0
    |   point_3D.z-2: 1
    | upper_bound: 0.2

    Only control values (x, y, z) that satisfy the following equation will be allowed:
    `x-0 * 0 + y-1 * 0 + z-2 * 1 > 0.2`


**target (optional)**
    Type: *Optional[float]*

    **Example**
    | input_constraints:
    | - weights:
    |   point_3D.x-0: 1
    |   point_3D.y-1: 2
    |   point_3D.z-2: 3
    | target: 4

    Only control values (x, y, z) that satisfy the following equation will be allowed:
    `x-0 * 1 + y-1 * 2 + z-2 * 3 = 4`


**lower_bound (optional)**
    Type: *Optional[float]*

    **Example**
    | input_constraints:
    | - weights:
    |   point_3D.x-0: 1
    |   point_3D.y-1: 2
    |   point_3D.z-2: 3
    | lower_bound: 4

    Only control values (x, y, z) that satisfy the following
    equation will be allowed:
    `x-0 * 1 + y-1 * 2 + z-2 * 3 >= 4`


**upper_bound (optional)**
    Type: *Optional[float]*

    **Example**
    | input_constraints:
    | - weights:
    |   point_3D.x-0: 1
    |   point_3D.y-1: 2
    |   point_3D.z-2: 3
    | upper_bound: 4

    Only control values (x, y, z) that satisfy the following equation will be allowed:
    `x-0 * 1 + y-1 * 2 + z-2 * 3 <= 4`



output_constraints (optional)
-----------------------------
Type: *Optional[List[OutputConstraintConfig]]*

A list of output constraints with unique names.

**name (required)**
    Type: *str*

    The unique name of the output constraint.


**target (optional)**
    Type: *Optional[float]*

    Defines the equality constraint

    (f(x) - b) / c = 0,

    where b is the target, f is a function of the control vector x, and c is the
    scale (scale).



**auto_scale (optional)**
    Type: *Optional[bool]*

    If set to true, Everest will automatically
    determine the scaling factor from the constraint value in batch 0.

    If scale is also set, the automatic value is multiplied by its value.


**lower_bound (optional)**
    Type: *Optional[float]*

    Defines the lower bound
    (greater than or equal) constraint

    (f(x) - b) / c >= 0,

    where b is the lower bound, f is a function of the control vector x, and c is
    the scale (scale).


**upper_bound (optional)**
    Type: *Optional[float]*

    Defines the upper bound (less than or equal) constraint:

    (f(x) - b) / c <= 0,

    where b is the upper bound, f is a function of the control vector x, and c is
    the scale (scale).


**scale (optional)**
    Type: *Optional[float]*

    Scaling of constraints (scale).

    scale is a normalization factor which can be used to scale the constraint
    to control its importance relative to the (singular) objective and the controls.

    Both the upper_bound and the function evaluation value will be scaled with this number.
    That means that if, e.g., the upper_bound is 0.5 and the scaling is 10, then the
    function evaluation value will be divided by 10 and bounded from above by 0.05.




simulator (optional)
--------------------
Type: *Optional[SimulatorConfig]*

Simulation settings

**name (optional)**
    Type: *Optional[str]*

    Specifies which queue to use


**cores (optional)**
    Type: *Optional[PositiveInt]*

    Defines the number of simultaneously running forward models.

    When using queue system lsf, this corresponds to number of nodes used at one
    time, whereas when using the local queue system, cores refers to the number of
    cores you want to use on your system.

    This number is specified in Ert as MAX_RUNNING.



**cores_per_node (optional)**
    Type: *Optional[PositiveInt]*

    defines the number of CPUs when running
    the forward models. This can for example be used in conjunction with the Eclipse
    parallel keyword for multiple CPU simulation runs. This keyword has no effect
    when running with the local queue.

    This number is specified in Ert as NUM_CPU.


**delete_run_path (optional)**
    Type: *Optional[bool]*

    Whether the batch folder for a successful simulation needs to be deleted.


**exclude_host (optional)**
    Type: *Optional[str]*

    Comma separated list of nodes that should be
    excluded from the slurm run.


**include_host (optional)**
    Type: *Optional[str]*

    Comma separated list of nodes that
    should be included in the slurm run


**max_memory (optional)**
    Type: *Optional[str]*

    Maximum memory usage for a slurm job.


**max_memory_cpu (optional)**
    Type: *Optional[str]*

    Maximum memory usage per cpu for a slurm job.


**max_runtime (optional)**
    Type: *Optional[NonNegativeInt]*

    Maximum allowed running time of a forward model. When
    set, a job is only allowed to run for max_runtime seconds.
    A value of 0 means unlimited runtime.



**options (optional)**
    Type: *Optional[str]*

    Used to specify options to LSF.
    Examples to set memory requirement is:
    * rusage[mem=1000]


**queue_system (optional)**
    Type: *Optional[Literal['lsf', 'local', 'slurm', 'torque']]*

    Defines which queue system the everest server runs on.


**resubmit_limit (optional)**
    Type: *Optional[NonNegativeInt]*


    Defines how many times should the queue system retry a forward model.

    A forward model may fail for reasons that are not due to the forward model
    itself, like a node in the cluster crashing, network issues, etc. Therefore, it
    might make sense to resubmit a forward model in case it fails.
    resumbit_limit defines the number of times we will resubmit a failing forward model.
    If not specified, a default value of 1 will be used.


**sbatch (optional)**
    Type: *Optional[str]*

    sbatch executable to be used by the slurm queue interface.


**scancel (optional)**
    Type: *Optional[str]*

    scancel executable to be used by the slurm queue interface.


**scontrol (optional)**
    Type: *Optional[str]*

    scontrol executable to be used by the slurm queue interface.


**squeue (optional)**
    Type: *Optional[str]*

    squeue executable to be used by the slurm queue interface.


**server (optional)**
    Type: *Optional[str]*

    Name of LSF server to use


**slurm_timeout (optional)**
    Type: *Optional[int]*

    Timeout for cached status used by the slurm queue interface


**squeue_timeout (optional)**
    Type: *Optional[int]*

    Timeout for cached status used by the slurm queue interface.


**enable_cache (optional)**
    Type: *bool*

    Enable forward model result caching.

    If enabled, objective and constraint function results are cached for
    each realization. If the optimizer requests an evaluation that has
    already been done before, these cached values will be re-used without
    running the forward model again.

    This option is disabled by default, since it will not be necessary for
    the most common use of a standard optimization with a continuous
    optimizer.


**qsub_cmd (optional)**
    Type: *Optional[str]*

    The submit command


**qstat_cmd (optional)**
    Type: *Optional[str]*

    The query command


**qdel_cmd (optional)**
    Type: *Optional[str]*

    The kill command


**qstat_options (optional)**
    Type: *Optional[str]*

    Options to be supplied to the qstat command. This defaults to -x, which tells the qstat command to include exited processes.


**cluster_label (optional)**
    Type: *Optional[str]*

    The name of the cluster you are running simulations in.


**memory_per_job (optional)**
    Type: *Optional[str]*

    You can specify the amount of memory you will need for running your job. This will ensure that not too many jobs will run on a single shared memory node at once, possibly crashing the compute node if it runs out of memory.
    You can get an indication of the memory requirement by watching the course of a local run using the htop utility. Whether you should set the peak memory usage as your requirement or a lower figure depends on how simultaneously each job will run.
    The option to be supplied will be used as a string in the qsub argument. You must specify the unit, either gb or mb.



**keep_qsub_output (optional)**
    Type: *Optional[int]*

    Set to 1 to keep error messages from qsub. Usually only to be used if somethign is seriously wrong with the queue environment/setup.


**submit_sleep (optional)**
    Type: *Optional[float]*

    To avoid stressing the TORQUE/PBS system you can instruct the driver to sleep for every submit request. The argument to the SUBMIT_SLEEP is the number of seconds to sleep for every submit, which can be a fraction like 0.5


**queue_query_timeout (optional)**
    Type: *Optional[int]*


    The driver allows the backend TORQUE/PBS system to be flaky, i.e. it may intermittently not respond and give error messages when submitting jobs or asking for job statuses. The timeout (in seconds) determines how long ERT will wait before it will give up. Applies to job submission (qsub) and job status queries (qstat). Default is 126 seconds.
    ERT will do exponential sleeps, starting at 2 seconds, and the provided timeout is a maximum. Let the timeout be sums of series like 2+4+8+16+32+64 in order to be explicit about the number of retries. Set to zero to disallow flakyness, setting it to 2 will allow for one re-attempt, and 6 will give two re-attempts. Example allowing six retries:



**project_code (optional)**
    Type: *Optional[str]*

    String identifier used to map hardware resource usage to a project or account. The project or account does not have to exist.



install_jobs (optional)
-----------------------
Type: *Optional[List[InstallJobConfig]]*

A list of jobs to install

**name (required)**
    Type: *str*

    name of the installed job


**source (required)**
    Type: *str*

    source file of the ert job



install_workflow_jobs (optional)
--------------------------------
Type: *Optional[List[InstallJobConfig]]*

A list of workflow jobs to install

**name (required)**
    Type: *str*

    name of the installed job


**source (required)**
    Type: *str*

    source file of the ert job



install_data (optional)
-----------------------
Type: *Optional[List[InstallDataConfig]]*

A list of install data elements from the install_data config
section. Each item marks what folders or paths need to be copied or linked
in order for the evaluation jobs to run.

**source (required)**
    Type: *str*


    Path to file or directory that needs to be copied or linked in the evaluation
    execution context.



**target (required)**
    Type: *str*


    Relative path to place the copy or link for the given source.



**link (optional)**
    Type: *Optional[bool]*


    If set to true will create a link to the given source at the given target,
    if not set the source will be copied at the given target.




install_templates (optional)
----------------------------
Type: *Optional[List[InstallTemplateConfig]]*

Allow the user to define the workflow establishing the model
chain for the purpose of sensitivity analysis, enabling the relationship
between sensitivity input variables and quantities of interests to be
evaluated.

**template (required)**
    Type: *str*




**output_file (required)**
    Type: *str*




**extra_data (optional)**
    Type: *Optional[str]*





forward_model (optional)
------------------------
Type: *Optional[List[str]]*

List of jobs to run


workflows (optional)
--------------------
Type: *Optional[WorkflowConfig]*

Workflows to run during optimization

**pre_simulation (optional)**
    Type: *Optional[List[str]]*

    List of workflow jobs triggered pre-simulation


**post_simulation (optional)**
    Type: *Optional[List[str]]*

    List of workflow jobs triggered post-simulation



server (optional)
-----------------
Type: *Optional[ServerConfig]*

Defines Everest server settings, i.e., which queue system,
queue name and queue options are used for the everest server.
The main reason for changing this section is situations where everest
times out because it can not add the server to the queue.
This makes it possible to reduce the resource requirements as they tend to
be low compared with the forward model.

Queue system and queue name defaults to the same as simulator, and the
server should not need to be configured by most users.
This is also true for the --include-host and --exclude-host options
that are used by the SLURM driver.

Note that changing values in this section has no impact on the resource
requirements of the forward models.

**name (optional)**
    Type: *Optional[str]*

    Specifies which queue to use.

    Examples are
    * mr
    * bigmem

    The everest server generally has lower resource requirements than forward models such
    as RMS and Eclipse.



**exclude_host (optional)**
    Type: *Optional[str]*

    Comma separated list of nodes that should be
    excluded from the slurm run


**include_host (optional)**
    Type: *Optional[str]*

    Comma separated list of nodes that
    should be included in the slurm run


**options (optional)**
    Type: *Optional[str]*

    Used to specify options to LSF.
    Examples to set memory requirement is:
    * rusage[mem=1000]


**queue_system (optional)**
    Type: *Optional[Literal['lsf', 'local', 'slurm']]*

    Defines which queue system the everest server runs on.



export (optional)
-----------------
Type: *Optional[ExportConfig]*

Settings to control the exports of a optimization run by everest.

**csv_output_filepath (optional)**
    Type: *Optional[str]*

    Specifies which file to write the export to.
    Defaults to <config_file_name>.csv in output folder.


**discard_gradient (optional)**
    Type: *Optional[bool]*

    If set to True, Everest export will not contain gradient simulation data.


**discard_rejected (optional)**
    Type: *Optional[bool]*

    If set to True, Everest export will contain only simulations
    that have the increase_merit flag set to true.


**keywords (optional)**
    Type: *Optional[List[str]]*

    List of eclipse keywords to be exported into csv.


**batches (optional)**
    Type: *Optional[List[int]]*

    list of batches to be exported, default is all batches.


**skip_export (optional)**
    Type: *Optional[bool]*

    set to True if export should not
    be run after the optimization case.
    Default value is False.



definitions (optional)
----------------------
Type: *Optional[dict]*

Section for specifying variables.

Used to specify variables that will be replaced in the file when encountered.

| scratch: /scratch/ert/
| num_reals: 10
| min_success: 13
| fixed_wells: [Prod1, Inj3]

Some keywords are pre-defined by Everest,

| realization: <GEO_ID>
| configpath: <CONFIG_PATH>
| runpath_file: <RUNPATH_FILE>
| eclbase: <ECLBASE>

and environment variables are exposed in the form 'os.NAME', for example:

| os.USER: $USER
| os.HOSTNAME: $HOSTNAME
| ...
