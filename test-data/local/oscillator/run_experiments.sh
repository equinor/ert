#!/bin/bash

# ert ensemble_smoother oscillator-ES.ert --target-case ES

ert iterative_ensemble_smoother oscillator-IES.ert --target-case IES-%d --num-iterations 4 

ert es_mda oscillator-MDA.ert --target-case MDA_%d --weights '1,1,1,1'

# ert iterative_ensemble_smoother oscillator-IES-low-error.ert --target-case IES-low-error-%d --num-iterations 16

# ert es_mda oscillator-MDA-low-error.ert --target-case MDA-low-error-%d --weights '1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1'
