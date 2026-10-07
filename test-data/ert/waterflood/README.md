# A waterflood you can score: what localization is for, and what a misfit cannot tell you

A two-phase (water/oil) reservoir simulation, run with
[minires](https://github.com/patnr/MiniRes) — a two-point flux approximation simulator,
pure Python, on PyPI — so the case needs **no simulator binary, no licence and no
deck**, and runs anywhere ert does. The 50 x 50 flood simulates in 0.35 s, so a
realization costs about a second and the whole case is a couple of minutes.

It is posed in metric units (m, day, bar, mD, cP) and reports its wells as ordinary
`WWCT`/`WOPR`/`WBHP` summary vectors, so it exercises the machinery a real reservoir
case does.

- **definition.py:** the numbers that define the case: grid, rock, fluids, wells, schedule.
- **waterflood.py:** forward model. Samples the permeability prior (at the first
  iteration), runs the flood, writes `WATERFLOOD.SMSPEC`/`.UNSMRY`.
- **generate_files.py:** run once to regenerate `CASE.EGRID`, `observations.txt` and the
  truth the case is scored against (`truth_permx.bgrdecl`, `truth_forecast.txt`).
- **evaluate.py:** score one experiment. **compare.py:** run the update schemes against
  each other and print the table below.

## The case

A 1000 x 1000 x 20 m reservoir, discretized areally into 50 x 50 cells, initially at
300 bar, produced by an ordinary **five-spot**: one injector at the centre taking
1000 m³/day, four producers towards the corners held at 250 bar, so their rates and
their water breakthrough are outcomes of how the permeability steers the flood.
40 report steps of 60 days inject about 0.6 pore volumes.

Uncertain: the `PERMX` field (log-normal about 200 mD, ~300 m correlation length,
`FIELD` + `FORWARD_INIT`, updated in log-space), the range of its variogram, and the
residual oil saturation of the Corey curves (both `GEN_KW`).

**The wells are observed over the first 16 report steps only** (960 days) and simulated
to the end. In the truth, PROD4 and PROD2 water during that history; **PROD3 breaks
through after it ends** (step 18); and **PROD1 never waters at all**, its corner being
poorly connected. `observations.txt` holds the 36 observations that history affords —
water cut and oil rate at each producer, and the injector's bottom-hole pressure, at
every fourth step — and `truth_permx.bgrdecl` / `truth_forecast.txt` carry the answers
the update is scored against.

## What the case is for

Four breakthrough times cannot determine 2500 cells. That is not a defect of the case:
it is the situation every real history match is in, and it is what localization exists
for. Because the truth is shipped, an update can be scored on where it *put* the field
and on what it predicts, not only on how well it fits. `compare.py` on the shipped
configuration, 50 realizations:

```
                           misfit  RMSE logK  corr logK   spread  RMSE forecast
                   prior    255.6      0.868     -0.093    0.782          0.335

     ES, no localization     21.1      0.908      0.364    0.509          0.146
            ES, adaptive     27.5      0.778      0.450    0.633          0.160
            ES, distance     18.1      0.779      0.401    0.697          0.078

 ES-MDA, no localization      1.7      1.036      0.085    0.452          0.113
        ES-MDA, adaptive      3.9      0.794      0.413    0.600          0.037
        ES-MDA, distance        (crashes -- see below)

                    EnIF     32.9      0.695      0.572    0.707          0.109
```

- **The best fit to the data has the worst field.** ES-MDA without localization reaches
  a misfit of 1.7, the lowest of any row, and an RMSE logK of 1.036, the highest of any
  row — *further from the truth than the prior's 0.868* — while its spread contracts
  from 0.790 to 0.452. Confidently wrong, and no misfit would say so.
- **Localization works.** RMSE logK after one ES: 0.908 unlocalized against 0.778
  (adaptive) and 0.779 (distance). Under ES-MDA: 1.036 against 0.794 (adaptive).
- **EnIF gets the best field**: RMSE logK 0.695 and correlation 0.572, with the spread
  largely intact, from a misfit an order above ES-MDA's.
- **Iterating buys the forecast, not the field.** Against plain ES with the same
  localization, ES-MDA cuts the forecast error 0.160 → 0.037 on the 24 report steps
  nobody assimilated, while RMSE logK goes 0.778 → 0.794, i.e. nowhere.

## On distance based localization

Both strategies are configured: the observations carry `LOCALIZATION` blocks with their
well's position and a radius, so `ANALYSIS_SET_VAR PARAMETERS FIELD DISTANCE` is the
whole switch. Two things to know.

**It crashes under ES-MDA.** `ert es_mda` with `FIELD DISTANCE` dies at the third
assimilation in `_update_strategies/_distance.py`, at `K *= rho`:

```
ValueError: operands could not be broadcast together with shapes (2500,31) (2500,33) (2500,31)
```

All 36 observations carry positions, and ES (a single assimilation) is unaffected: the
taper and the assimilated set stop agreeing in number once observations are dropped
between iterations. `test_waterflood.py` carries this as a non-strict `xfail`.

**Its radius has an optimum.** Sweeping it at 20 realizations (RMSE logK after one ES,
against a prior of 0.885): 400 m → 0.901, **250 m → 0.840**, 150 m → 0.841,
100 m → 0.854, 60 m → 0.868 — by which point the taper is narrow enough that little
survives it. The observations ship with the 250 m optimum.

Neither the case nor its tests claims a winner between the two tapers, because tuned
they are hard to separate here, and untuned the comparison says more about the tuning
than about the method (RMSE logK after one ES):

| realizations | no localization | adaptive | distance | prior |
|---|---|---|---|---|
| 20 | 1.490 | 0.945 | **0.840** | 0.885 |
| 50 | 0.908 | **0.778** | 0.779 | 0.868 |

At 50 that is a dead heat; at 20 the radius wins comfortably, because an adaptive
threshold has to track the noise floor of a correlation, ~1/sqrt(N), and the 0.3 in
`config.ert` suits 50 realizations and is too permissive at 20 — a radius has no such
dependence. What both rows agree on, and what the tests assert, is that either taper
beats none.

## Running it

```
ert es_mda config.ert      # the shipped configuration, ~2 minutes
python evaluate.py         # score it
python compare.py 50       # or score the schemes against each other
```
