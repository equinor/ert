Running the forward model
#########################

.. code-block:: bash

    FORWARD_MODEL RMS(<IENS> <RMS_PROJECT> <RMS_WORKFLOW>)

RMS script documentation
########################

The script must be invoked with minimum three positional arguments:

Positional arguments:
  IENS
        Realization number
  RMS_PROJECT
        The RMS project we are running
  RMS_WORKFLOW
        The rms workflow we intend to run

Optional arguments:
  -r, --run-path RUN_PATH
                        The directory which will be used as cwd when running
                        rms
  -t, --target-file TARGET_FILE
                        name of file which should be created/updated by rms
  -i, --import-path IMPORT_PATH
                        the prefix of all relative paths when rms is importing
  -e, --export-path EXPORT_PATH
                        the prefix of all relative paths when rms is exporting
  -v, --version VERSION
                        the prefix of all relative paths when rms is exporting
  -a, --allow-no-env
                        Boolean flag to allow RMS to run without a site configured
                        environment. WARNING: This is typically a sign of using an
                        unsupported version of RMS and the configuration of the
                        environment is now entirely on the shoulders of the user.
                        Consider contacting your system administrators.
