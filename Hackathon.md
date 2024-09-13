# ERT GUI Hackathon

```bash
ert test_run poly.ert | curl -s -XPOST -H "Content-type: application/json" http://127.0.0.1:8000/experiments/ -d @-
ert ensemble_experiment poly.ert --realizations 0-99 | curl -s -XPOST -H "Content-type: application/json" http://127.0.0.1:8000/experiments/ -d @-
ert ensemble_smoother poly.ert --realizations 0-99 | curl -s -XPOST -H "Content-type: application/json" http://127.0.0.1:8000/experiments/ -d @-
ert es_mda poly.ert --realizations 0-99 --target-ensemble ensemble_%d | curl -s -XPOST -H "Content-type: application/json" http://127.0.0.1:8000/experiments/ -d @-
ert iterative_ensemble_smoother poly.ert --realizations 0-99 --target-ensemble ensemble_%d --num-iterations 4 | curl -s -XPOST -H "Content-type: application/json" http://127.0.0.1:8000/experiments/ -d @-
```

```bash
ert status <experiment_id>
ert status $(<command from previous section> | jq -r .experiment_id)

ert status $(x | jq -r .experiment_id)

```

