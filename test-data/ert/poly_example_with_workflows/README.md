## Polynomial curve fitting with all workflow hooks

This is the minimal polynomial model updating case (see `poly_example`) extended
with one workflow hooked to each of the seven runtime points that ERT exposes
through `HOOK_WORKFLOW`.

All seven workflows reuse a single workflow job, `PRINT_MESSAGE`, which simply
prints the hook it was triggered from. This makes the example useful for
demonstrating and verifying when each hook fires during an experiment.

### The seven hooks

| Workflow file                | Hook runtime       | When it runs                                   |
| ---------------------------- | ------------------ | ---------------------------------------------- |
| `workflows/pre_experiment`   | `PRE_EXPERIMENT`   | Before the experiment starts                   |
| `workflows/pre_simulation`   | `PRE_SIMULATION`   | Before the simulations of each iteration start |
| `workflows/post_simulation`  | `POST_SIMULATION`  | After all simulations of an iteration complete |
| `workflows/pre_first_update` | `PRE_FIRST_UPDATE` | Only before the first update                   |
| `workflows/pre_update`       | `PRE_UPDATE`       | Before every update step                       |
| `workflows/post_update`      | `POST_UPDATE`      | After every update step                        |
| `workflows/post_experiment`  | `POST_EXPERIMENT`  | After the experiment has completed             |

For non-iterative algorithms `PRE_FIRST_UPDATE` is equal to `PRE_UPDATE`.

### Layout

- `poly.ert` - the configuration, loading the workflow job, the seven workflows
  and hooking each workflow to its runtime.
- `PRINT_MESSAGE` / `print_message.py` - the shared workflow job and its script.
- `workflows/` - one workflow file per hook, each calling `PRINT_MESSAGE` with
  the name of the hook.

The rest of the configuration (parameters, forward model, observations) is
identical to the `poly_example` case.
