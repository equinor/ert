# ERT GUI Hackathon

### Starting the experiment server:
```bash
fastapi run src/ert/experiment_server/main.py
```

You can now see the documentation be visiting: http://127.0.0.1:8000/docs

Note that the websocket endpoint (`"ws://127.0.0.1:8000/experiments/{experiment_id}/events"`) will not show up in the documentation.

### Running the ert GUI
The GUI can be started as normal by running for example:

```bash
ert gui poly.ert
```

The ert GUI has been modified to submit experiments to the server instead, and then stream statuses from the server to view the status. The restriction to only run one experiment at the time has also been lifted, and you can submit more than one experiment by focusing the main window and selecting another experiment to run.

### Submitting experiments from cli
The ert cli commands have been modified to output a valid json configuration for the given config file, instead of running the experiment. This can be piped directly into `curl` to submit a new experiment to the server:

```bash
ert test_run poly.ert | curl -s -XPOST -H "Content-type: application/json" http://127.0.0.1:8000/experiments/ -d @-
ert ensemble_experiment poly.ert --realizations 0-99 | curl -s -XPOST -H "Content-type: application/json" http://127.0.0.1:8000/experiments/ -d @-
ert ensemble_smoother poly.ert --realizations 0-99 | curl -s -XPOST -H "Content-type: application/json" http://127.0.0.1:8000/experiments/ -d @-
ert es_mda poly.ert --realizations 0-99 --target-ensemble ensemble_%d | curl -s -XPOST -H "Content-type: application/json" http://127.0.0.1:8000/experiments/ -d @-
ert iterative_ensemble_smoother poly.ert --realizations 0-99 --target-ensemble ensemble_%d --num-iterations 4 | curl -s -XPOST -H "Content-type: application/json" http://127.0.0.1:8000/experiments/ -d @-
```

### ert status command
A new command has been added that will open the status window (run dialog) for a given experiment and connect to that experiment id:

```bash
ert status <experiment_id>
```

This can be combined with the commands from the previous section to directly submit and open a status window by doing the following:

```bash
ert status $(<command from previous section> | jq -r .experiment_id)
```

