# Sensitivity Analysis Integration Plan

## Purpose

Sensitivity analysis should become an ERT-owned workflow instead of a workflow
that depends on externally generated `DESIGN_MATRIX` files.

Today users typically author a sensitivity design in Excel, generate a design
matrix with `fmudesign`, point ERT at the generated matrix, and then lose the
ability to validate, refresh, preview, or quality-control the design inside ERT
before running simulations.

The first integration phase should deliver a GUI-centered Sensitivity Analysis
run model where users can:

1. Select an Excel sensitivity-design workbook.
2. Validate the workbook in ERT.
3. Refresh validation and preview after editing the workbook without restarting
   ERT.
4. Generate and inspect draft design data before running simulations.
5. Run a sensitivity study from the accepted draft design.

`DESIGN_MATRIX` remains a compatibility path for already generated physical
realization tables. It should not be the execution model for new updateable
sensitivity-design factors.

## Phase 1 Principles

- The GUI run model is the primary Phase 1 deliverable.
- ERT should own parsing, validation, design generation, preview, QC metadata,
  and execution setup.
- The existing `fmudesign` Excel workbook style is the initial supported input
  format.
- Normal execution should not write an intermediate Excel file.
- Generated physical design tables may still be exported explicitly for
  compatibility with legacy tools.
- Continuous sensitivity factors that can participate in history matching should
  become native scalar `GenKwConfig` parameters with real distribution settings.
- Updateable scalar values should be stored in ERT's latent standard-normal
  scalar storage convention, with physical values derived for preview, runpath
  export, and reporting.
- Discrete and categorical factors from sensitivity-design workflows are
  sensitivity-analysis-only at first. They may be generated and exported, but
  must not become updateable history-matching parameters.
- `SENSITIVITY_DESIGN <xlsx>` is optional for Phase 1. It is useful for headless
  and scripted workflows, but it should not block the GUI workflow.

## Target User Workflow

1. Open ERT and choose the Sensitivity Analysis run model.
2. Select or reference an Excel sensitivity-design workbook.
3. Validate workbook structure, factors, distributions, defaults, cases, active
   realizations, and supported correlations.
4. Generate a draft design in memory.
5. Preview cases, active realizations, factor values, distributions, warnings,
   and QC summaries.
6. Refresh the draft after workbook edits without mutating existing experiments.
7. Optionally adjust parameter updateability in an ERT-owned parameter view.
8. Create and run an experiment from the accepted draft design.
9. Preserve metadata that links realizations, factors, cases, workbook source,
   and generated values.
10. Optionally export a generated physical design matrix for legacy tooling.

## Architecture

### Sensitivity Study Model

Add small ERT-owned model types under a new sensitivity-design package. Keep the
models minimal at first and extend them only when needed by implemented slices.

- `DesignSpec`: parsed and validated workbook definition, including workbook
  path, sheet names, generation settings, seed, active-realization settings,
  defaults, cases, factors, and correlation metadata.
- `DesignFactor`: one workbook factor, including name, source context,
  distribution or constant value, default value, optional description/units,
  correlation group, and whether it can become an ERT scalar parameter.
- `DesignCase`: one reference, scenario, seed, background, or other supported
  case, including `SENSNAME`, `SENSCASE`, case type, realization count, and
  factor overrides.
- `DesignMatrixData`: generated realization-indexed data, including physical
  preview values, latent scalar values for updateable continuous factors, active
  realizations, and realization-to-case metadata.
- `SensitivityStudy`: the object passed between preview, GUI, storage, and run
  setup. It ties together the parsed spec, generated data, metadata, active
  realizations, and generated ERT parameter configs.

Do not extend the legacy `DESIGN_MATRIX` class into the main sensitivity-design
model. Refactor it only where useful to validate or export in-memory physical
tables for compatibility.

### Sampling And Storage

ERT currently has two scalar paths: sampled `GEN_KW` values and copied raw
`DESIGN_MATRIX` values. Sensitivity design should converge on one sampling model
for distribution-defined scalar parameters.

The target sampling contract is:

1. Generate deterministic quantiles `u` for each sampled scalar factor.
2. Preserve stable variable identity so generated values can be mapped back to
   workbook factors and ERT parameters.
3. Convert quantiles to latent values with `z = norm.ppf(u)` for scalar storage.
4. Derive physical values from the same quantiles through each factor's
   distribution transform for preview, runpath export, and reporting.
5. Use correlated quantile generation for correlation groups once correlation
   support is implemented.

This likely requires a `probabilit` API that exposes LHS/correlated quantiles
before physical distribution transforms. The API should be generic, not
ERT-specific, and should keep probabilit's existing physical sampling behavior
backward compatible.

### Distribution Handling

The first implementation should support only distribution forms with clear ERT
semantics. Unsupported or ambiguous workbook forms should fail validation with an
actionable message rather than being silently approximated.

Initial supported candidates:

- Normal and normal p10/p90 forms that can be converted to ERT `NORMAL`.
- Bounded normal only if the current ERT truncated-normal clipping semantics are
  explicitly documented to users.
- Uniform and uniform p10/p90 forms that can be converted to ERT `UNIFORM`.
- Triangular where workbook semantics match ERT `TRIANGULAR` exactly.
- Log-uniform where workbook semantics match ERT `LOGUNIF` exactly.
- Constants as non-updateable values or metadata.

Reject initially unless explicit ERT support and tests are added:

- PERT.
- Beta.
- Bounded lognormal.
- Triangular p10/p90 variants with non-ERT semantics.
- Weighted categorical values as updateable history-matching parameters.

Discrete and categorical values may still be generated for sensitivity-analysis
runs, preview, reporting, and runpath use once that slice is implemented.

### Parameter Merge Semantics

Generated continuous factors should be registered before experiment creation as
native scalar parameters, not overlaid late as raw design-matrix columns.

Rules:

- Parse ordinary ERT config parameters first.
- Parse and validate the sensitivity design.
- Convert supported continuous factors into generated `GenKwConfig` entries with
  real `DistributionSettings` and source metadata.
- Reject duplicate names between config-authored parameters and generated
  sensitivity factors until a deliberate override syntax exists.
- Default generated continuous factors to updateable where appropriate.
- Let users disable updates through an ERT-owned parameter control surface, not
  through an ERT-specific Excel workbook column.
- Store generated latent scalar values in `SCALAR.parquet`; do not execute
  updateable factors through raw `DESIGN_MATRIX` semantics.

### GUI, CLI, And Config

The GUI workflow is the Phase 1 product. CLI and config entry points should reuse
the same parser, validation, preview, generation, and run setup APIs.

Optional config syntax:

```text
SENSITIVITY_DESIGN <path-to-design-specification.xlsx>
```

If added, the keyword means ERT imports design definitions and owns sampling,
metadata, latent storage, and physical realization-table generation. It must be
mutually exclusive with `DESIGN_MATRIX`, which imports already generated physical
values with raw, non-updateable semantics.

## Implementation Roadmap

The work should be delivered in small reviewable slices. The roadmap below is a
dependency graph, not a strict single-team sequence.

### 0. Low-Risk Seams

Goal: create internal seams without changing current behavior.

- Refactor `DESIGN_MATRIX` handling so Excel reading is separate from validation
  and conversion of physical design tables. Add an in-memory constructor such as
  `DesignMatrix.from_dataframe()` if useful.
- Extract current scalar-prior sampling behind a small helper or service while
  preserving existing sampled values.
- Add a parameter-summary model for the GUI parameter viewer while preserving
  current viewer behavior.
- Clarify current parameter source metadata and how a future sensitivity-design
  source should fit.

Acceptance criteria:

- Existing `DESIGN_MATRIX` and `GEN_KW` behavior is unchanged.
- Tests prove Excel-loaded and in-memory physical design tables are equivalent.
- No sensitivity-design execution behavior is introduced yet.

### 1. Workbook Parsing And Validation

Goal: parse the supported `fmudesign` workbook shape into ERT-owned models.

- Add the minimal sensitivity-design package and domain models.
- Parse the initial supported workbook sheets, starting with `general_input`,
  `designinput`, and `defaultvalues`.
- Capture source context for sheet, row, column, and cell where practical.
- Validate duplicate names, missing required values, invalid sheet structure,
  invalid paths, invalid repeats/seeds, unsupported distributions, and unsupported
  workbook constructs.
- Provide a reusable validation API that the GUI and tests can call.

Acceptance criteria:

- Representative and minimal workbooks parse into ERT models.
- Validation errors are actionable and include workbook context when available.
- No realization values or experiments are generated yet.

### 2. Draft Design Generation And Preview API

Goal: generate in-memory draft design data without writing intermediate Excel
files.

- Add `DesignMatrixData` for generated physical rows, optional latent rows,
  active realizations, and realization-to-case metadata.
- Generate deterministic rows for reference, default, scenario, seed, and
  supported background cases.
- Generate independent continuous preview values for the initial supported
  distribution subset.
- Return parsed spec, validation results, preview tables, warnings, and metadata
  from one reusable preview API.
- Keep existing `DESIGN_MATRIX` behavior unchanged.

Acceptance criteria:

- Preview is deterministic for a given workbook, seed, and realization count.
- Preview API can be called repeatedly after workbook edits.
- Refresh-like calls do not mutate existing experiments.

### 3. Native Scalar Execution

Goal: run a minimal sensitivity study while preserving ERT's native scalar
storage semantics.

- Add source metadata for generated sensitivity-design scalar parameters.
- Convert supported continuous factors into native `GenKwConfig` entries.
- Add a generated-latent scalar input path to prior initialization.
- Assemble a minimal `SensitivityStudy` object for run setup.
- Run an end-to-end minimal sensitivity study from accepted draft data.
- Persist enough metadata to connect realizations to factors and cases after
  execution.

Acceptance criteria:

- Continuous factors store latent standard-normal values in `SCALAR.parquet`.
- Runpath export and loading transform values like ordinary scalar `GEN_KW`.
- Duplicate names with config-authored parameters are rejected.
- Existing `DESIGN_MATRIX` workflows still pass.

### 4. GUI Workflow

Goal: expose the tested backend workflow as the user-visible Phase 1 feature.

- Add a Sensitivity Analysis run model or experiment panel.
- Let users select a workbook path.
- Wire workbook validation and refresh.
- Add preview tables for cases, active realizations, factors, physical values,
  warnings, and available latent values.
- Extend the parameter viewer to show parameter source, group, distribution, and
  updateability state.
- Let users change updateability in draft ERT state before experiment creation.
- Run the accepted draft sensitivity study from the GUI.

Acceptance criteria:

- Users can validate, refresh, preview, and run without restarting ERT.
- Refresh updates draft state only; it does not mutate created or running
  experiments.
- Generated sensitivity metadata links realizations to factors and cases.

### 5. Optional Headless And Compatibility Entry Points

Goal: add scriptable workflows after the core GUI path is working.

- Add `SENSITIVITY_DESIGN <xlsx>` if needed for automation or config-driven use.
- Add a headless validate/preview command using the shared API.
- Add explicit export of generated physical design tables for legacy tools.
- Document the distinction between `SENSITIVITY_DESIGN` and `DESIGN_MATRIX`.

Acceptance criteria:

- `SENSITIVITY_DESIGN` imports design definitions, not generated physical tables.
- `SENSITIVITY_DESIGN` and `DESIGN_MATRIX` are mutually exclusive.
- Exported physical tables can be consumed by the existing `DESIGN_MATRIX` path.

### 6. Quantile Sampling And Correlations

Goal: add the machinery needed for important `fmudesign` parity while protecting
existing scalar behavior.

- Specify and implement the required `probabilit` quantile sampling API.
- Integrate independent quantile-based latent sampling in ERT.
- Parse and validate correlation metadata.
- Generate correlated continuous latent values through the unified sampler.
- Add correlation QC summaries in the preview API and GUI.

Acceptance criteria:

- Existing physical `probabilit.sample()` behavior remains backward compatible.
- Correlation groups are sampled jointly.
- QC reports requested and observed latent/physical correlations.
- Correlated scalar factors remain compatible with ensemble-smoother updates.

### 7. Full Workbook Parity And External Analysis Handoff

Goal: move from MVP to broader `fmudesign` parity and external analysis support.

- Add selected support for external designs, background parameters, dependencies,
  derived parameters, discrete/categorical factors, and remaining distribution
  forms.
- Add parity tests against representative `fmudesign` examples.
- Persist complete sensitivity metadata and physical preview data.
- Provide a stable export or documented storage access path that connects
  realizations, factors, cases, and responses for external analysis.

Acceptance criteria:

- Supported parity examples parse, preview, and run, or are explicitly rejected
  with documented limitations.
- External tools can reliably join generated design data with response data.
- ERT does not need to own sensitivity-result methods such as tornado plots,
  rank correlations, regression coefficients, or Sobol metrics in this phase.

## Open Decisions

- What is the exact Phase 1 MVP workbook feature set?
- Which distribution forms are supported in the first end-to-end workflow?
- Should invalid correlation matrices be repaired with warnings or rejected?
- Does Phase 1 require `SENSITIVITY_DESIGN`, or is the GUI draft workflow enough?
- What is the minimal persistent metadata schema needed after a run?
- How should parameter updateability overrides be represented in draft GUI state
  and persisted experiment metadata?
