
.. ert_forward_models::

    In the context of uncertainty estimation and data assimilation,
    a forward model refers to a predictive model that simulates how a system evolves
    over time given certain inputs or intial conditions.
    The model is called "forward" because it predicts the future state of the system based 
    on the current state and a set of input parameters.
    The predictive model may include pre-processing and post-processing steps in addition
    to the physics simulator itself.
    In ERT, we think of a forward model as a sequence of jobs such as making directories,
    copying files, executing simulators etc.
    
    In order to include a job in the forward model within ERT, utilize the keyword
    :code:`FORWARD_MODEL`.
    The subsequent list documents pre-configured jobs in ERT that can be used to define
    your forward models.
