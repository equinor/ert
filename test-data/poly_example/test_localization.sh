#!/bin/bash

ert ensemble_smoother poly_no_loc.ert --target ES_no_loc

ert ensemble_smoother poly_zero_loc.ert --target ES_zero_loc

ert ensemble_smoother poly_loc.ert --target ES_loc

ert ensemble_smoother poly_loc_1.ert --target ES_loc_1
