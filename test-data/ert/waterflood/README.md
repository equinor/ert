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

**The wells are observed over the first 18 report steps only** (1080 days) and simulated
to the end. In the truth, PROD4 and PROD2 water well within that history; **PROD3 only
breaks through right at its end** (step 18); and **PROD1 never waters at all**, its
corner being poorly connected. `observations.txt` holds the 27 observations that history affords —
oil rate at each producer and the injector's bottom-hole pressure, every fourth step;
PROD1's water cut, which never rises; and PROD2, PROD3 and PROD4's water cut
breakthrough, a crossing of 20%, 1% and 50% respectively, each dated from the truth at
full (every report step) resolution — and `truth_permx.bgrdecl` / `truth_forecast.txt`
carry the answers the update is scored against.

## What the case is for

Four breakthrough times cannot determine 2500 cells. That is not a defect of the case:
it is the situation every real history match is in, and it is what localization exists
for. Because the truth is shipped, an update can be scored on where it *put* the field
and on what it predicts, not only on how well it fits.


20 realizations, assimilating the first 18 report steps:
```
                           misfit  RMSE logK  corr logK   spread  RMSE forecast
                   prior    302.6      0.859     -0.027    0.790          0.335
     ES, no localization     18.6      0.724      0.552    0.599          0.083
            ES, adaptive     28.1      0.780      0.407    0.644          0.150
            ES, distance     22.3      0.845      0.250    0.706          0.130
 ES-MDA, no localization      1.7      1.467     -0.001    0.222          0.194
        ES-MDA, adaptive      4.6      0.724      0.509    0.660          0.046
        ES-MDA, distance     13.7      0.834      0.298    0.651          0.093
                    EnIF     41.8      0.708      0.548    0.712          0.137
```


50 realizations, assimilating the first 18 report steps
```
                           misfit  RMSE logK  corr logK   spread  RMSE forecast
                   prior    327.3      0.868     -0.093    0.782          0.336
     ES, no localization     16.5      0.869      0.309    0.530          0.054
            ES, adaptive     33.8      0.892      0.261    0.524          0.224
            ES, distance     22.3      0.845      0.250    0.706          0.130
 ES-MDA, no localization      1.7      1.467     -0.001    0.222          0.194
        ES-MDA, adaptive      4.0      0.774      0.434    0.603          0.023
        ES-MDA, distance     13.7      0.834      0.298    0.651          0.093
                    EnIF     41.8      0.708      0.548    0.712          0.137
```

100 realizations, assimilating the first 18 report steps
```

                           misfit  RMSE logK  corr logK   spread  RMSE forecast
                   prior    327.3      0.868     -0.093    0.782          0.336
     ES, no localization     16.5      0.869      0.309    0.530          0.054
            ES, adaptive     33.8      0.892      0.261    0.524          0.224
            ES, distance     22.3      0.845      0.250    0.706          0.130
 ES-MDA, no localization      1.7      1.467     -0.001    0.222          0.194
        ES-MDA, adaptive      4.6      0.724      0.509    0.660          0.046
        ES-MDA, distance     17.7      0.793      0.367    0.692          0.043
                    EnIF     40.1      0.721      0.548    0.745          0.126
```


## Running it

```
ert es_mda config.ert      # the shipped configuration, ~2 minutes
python evaluate.py         # score it
python compare.py 50       # or score the schemes against each other
```
