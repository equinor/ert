# `poly_template` 

This is a template for instantiating a forward model that generates a variant of the `poly_example` model of a required size. See the documentation of that example for the basics of how it functions.


## Usage

To instantiate a version of the template, run the `make_poly_example()` function
in `tests/performance_tests/performance_utils.py`, or the main function in there.

It takes several parameters:

* `gen_data_count` &mdash; Number of gen_data vectors to generate
* `gen_data_entries` &mdash; Number of entries in each gen_data vector
* `summary_data_entries` &mdash; Number of entries in each summary vector
* `reals` &mdash; Number of realizations to run
* `summary_data_count` &mdash; Number of summary vectors to put in the summary file
* `sum_obs_count` &mdash; Attach observations to this many gen_data vectors, must be less that gen_data_count
* `gen_obs_count` &mdash; Attach observations to this many gen_data vectors, must be less that gen_data_count
* `sum_obs_every` &mdash; Put an observation point for every n summary_data entry, for the summary vectors that have 
    observations
* `gen_obs_every` &mdash; Put an observation point for every n gen_data entry, for the gen_data vectors that have 
    observations
* `parameter_entries` &mdash; Number of parameters in each group
* `parameter_count` &mdash; Numbar of groups of parameters will be generated
* `ministeps` &mdash; how many ministeps are in each report step for summary data, should probably be 1
