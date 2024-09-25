# ERT GUI Hackathon

## Event Details

### Program

```text
08:45 - 09:15 - Intro and selection of groups
09:15 - 11:00 - Hack
11:00 - 11:30 - Lunch
11:30 - 14:30 - Hack
14:30 - 15:00 - Demo
```

### Task suggestions

- Create a GUI for ERT using Svelte, React or the web-framework of your choice
- Add Workflow status to the GUI (either in existing GUI, or something new)
- Make a mobile app for ERT showing the experiment status
- Add support for multiple users to experiment server
- Create a visualization of an experiment config (what is sent to the server), for example showing the jobs, settings etc.
- Create a new configuration for ERT
- ... or anything cool you can think of related to ERT, experiments, GUI, events, etc. :tada:

## Technical details

:warning: This is a higly experimental and hacky branch, expect things to break!

### Starting the experiment server:
If you are on a mac you first need to increase the ulimit, as running multiple experiments can open a lot of files:
```bash
ulimit -n 4096
```

Then you can start the experiment server in production mode:

```bash
fastapi run src/ert/experiment_server/main.py
```

You can now see the documentation by visiting: http://127.0.0.1:8000/docs

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
ert status $(<command from previous section> | jq -r .id)
```

This requires `jq` to be installed (`brew install jq` or `sudo apt install jq`).

### Event types
The events the you receive from the websocket endpoint will be one of the following:

```python
StatusEvents = Union[
    FullSnapshotEvent,
    SnapshotUpdateEvent,
    EndEvent,
    AnalysisStatusEvent,
    AnalysisTimeEvent,
    AnalysisReportEvent,
    RunModelErrorEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelDataEvent,
    RunModelUpdateEndEvent,
]
```

### ert next.js frontent

A very small next.js frontend has been set up in: frontend/ert-gui. First make sure you have node.js >= v18.17.0 installed.
To run it, first start the experiment server, then navigate to:

```bash
cd frontend/ert-gui
```

install deps:

```bash
npm install
```

and run:

```bash
npm run dev
```
