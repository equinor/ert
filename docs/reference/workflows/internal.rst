Internal workflows jobs
=======================

These jobs invoke a function in the address space of the ERT program
itself; i.e. they are run as part of the running ERT process - and can
in principle do anything that ERT can do itself. There are two
varieties of the internal workflow jobs:

Invoke a pre exported function
------------------------------

This is the simplest, where you can invoke a predefined ERT
function. The function must already have been marked as *exported* in
the ert code base. The list of predefined workflow jobs based on this
method can be found here: :ref:`built_in_workflow_jobs`.
Feel free to reach out if you have suggestions for new predefined workflow jobs.

.. _ert_script:

Run a Python script
-------------------

If you are using one of the Python based frontends, *GUI* or
*CLI*, you can write your own Python script which is run as part
of the existing process. By using the full ERT Python API you get
access to powerful customization/automation features. Below is an
example of :code:`ErtScript` which calculates the misfit for all
observations and prints the result to a text file. All Python scripts
of this kind must:

  1. Be implemented as a class which inherits from :code:`ErtScript`
  2. Have a method :code:`run(self)` which does the actual job

.. code:: python

    from ert.util import DoubleVector
    from res.enkf import ErtScript

    class ExportMisfit(ErtScript):

        def run(self):
            # Get a handle to running ert instance
            ert = self.ert()


            # Get a handle to the case / filesystem we are interested in;
            # this should ideally come as an argument - not just use current.
            fs = ert.storage_manager.current_case


            # How many realisations:
            ens_size = ert.getEnsembleSize( )


            # Get a handle to all the observations
            observations = ert.getObservations()


            # Iterate through all the observations; each element in this
            # iteration corresponds to one key in the observations file.
            for obs in observations:
                misfit = DoubleVector()
                for iens in range(ens_size):
                    chi2 = obs.getTotalChi2( fs , iens )
                    misfit[iens] = chi2

                permutation = misfit.permutationSort( )

                print " #      Realisation     Misfit:%s" % obs.observation_key
                print "-----------------------------------"
                for index in range(len(misfit)):
                    iens = permutation[index]
                    print "%2d     %2d            %10.5f" % (index , iens , misfit[iens])

                print "-----------------------------------\n"
