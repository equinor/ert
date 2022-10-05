## Template 
This is a template for instantiating a forward model that generates data of a
required size. It is based on the poly_example model. See the documentation of
that for the basics of how it functions.

### Usage
To instantiate a version of the template, run the make_poly_example() function
in tests/performance/performance_utils.py, or the main function in there.

It takes several parameters:

* gen_data_count
  - Number of gen_data vectors to generate
* gen_data_entries
  - Number of entries in each gen_data vector
* summary_data_entries
  - Number of entries in each summary vector
* reals
  - Number of realizations to run
* summary_data_count
  - Number of summary vectors to put in the summary file
* sum_obs_count
  - Attach observations to this many gen_data vectors, must be less that gen_data_count
* gen_obs_count
  - Attach observations to this many gen_data vectors, must be less that gen_data_count
* sum_obs_every
  - Put an observation point for every n summary_data entry, for the summary vectors that have 
    observations
* gen_obs_every
  - Put an observation point for every n gen_data entry, for the gen_data vectors that have 
    observations
* parameter_entries
  - Number of parameters in each group
* parameter_count
  - Numbar of groups of parameters will be generated
* ministeps
  - how many ministeps are in each report step for summary data, should probably be 1
