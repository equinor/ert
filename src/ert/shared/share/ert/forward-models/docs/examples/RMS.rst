Running the forward model
#########################

RMS is usually incorporated in ERT configurations using statements like

.. code-block:: bash

    DEFINE  <RMS_PROJECT>         reek.rms11.0.1
    DEFINE  <RMS_VERSION>         11.0.1
    DEFINE  <RMS_WORKFLOW_NAME>   MAIN_WORKFLOW
    FORWARD_MODEL RMS(<IENS>=<IENS>, <RMS_VERSION>=<RMS_VERSION>, <RMS_PROJECT>=<CONFIG_PATH>/../../rms/model/<RMS_NAME>)

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
                        The version of rms to use
  -a, --allow-no-env
                        Boolean flag to allow RMS to run without a site configured
                        environment. WARNING: This is typically a sign of using an
                        unsupported version of RMS and the configuration of the
                        environment is now entirely on the shoulders of the user.
                        Consider contacting your system administrators.
