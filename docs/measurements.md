# Measurement Artifacts

## Residential Study

`results/` contains the complete measurements for the implementation
described in the methods documentation. It includes both
capacities, both split protocols, the small-model equal-weight ablation, both
membership probes, and all 205 target/shadow checkpoints from the main suite.
A separate isolated runtime measurement adds five checkpoints, for 210 in total.
Generated figures are stored separately in `figures/`.

Verify the snapshot, its checkpoints, and its correspondence to local source and
datasets from the repository root:

```bash
python scripts/verify_artifact.py
```

This verification uses Python's standard library and does not retrain models.
Fetch the datasets with `git lfs pull` first. Checkpoint checksums establish file
integrity, not a privacy guarantee or proof of experimental validity.

| Purpose | Path within `results/` |
|---|---|
| Small-model table values | `paper_numbers.json` |
| Medium-model table values | `paper_numbers_baseline_16_medium.json` |
| Primary summaries | `baseline_16/summary.json`, `baseline_16_medium/summary.json` |
| Equal-weight ablation | `baseline_16/summary_equal_weight.json` |
| Temporal summaries | `baseline_16/temporal/summary.json`, `baseline_16_medium/temporal/summary.json` |
| Temporal reports | `temporal_split_check.json`, `temporal_split_check_medium.json` |
| Loss-based probes | `{baseline_16,baseline_16_medium}/mia/` |
| Shadow-model probes and their models | `{baseline_16,baseline_16_medium}/shadow_mia/` |
| Execution record and environment | `provenance/` |
| Isolated timing inputs and summary | `runtime_isolated/baseline_16/seed_42/`, `runtime_isolated/runtime_summary.json` |
| Isolated timing launch conditions and completion | `runtime_isolated/provenance/` |

At repository root, `figures/{baseline_16,baseline_16_medium}/` contains the PDF
and PNG plots. `repro_manifest.json` lists checksums for the result and figure
files, along with source/data identities and execution metadata.

Maintainers can regenerate the manifest after assembling measurement outputs
with `python scripts/build_manifest.py`. This hashes the existing files and
checks source/data correspondence; it does not generate scientific results.
Run the verifier before modifying a packaged snapshot to check its integrity,
and preserve an independently verified copy. Publishing completed measurements
in a separate subdirectory is a packaging operation, not a training run.
Verify copied inputs against their execution archive before rebuilding the
manifest; rebuilding hashes alone does not establish that the inputs are valid.
The main five-seed results must not be overwritten with single-seed timing runs.

Each target JSON includes its full configuration, data/source hashes, partition
hashes, environment, per-round metrics, and checkpoint checksum. Shadow and attack
outputs retain their target identities. The stored paths in execution records
describe the original host; they are provenance, not paths that need to exist on
another machine.

## Recompute Tables Without Training

Copy the immutable artifact to a separate working directory before running
analysis. This preserves the recorded checksums:

```bash
# Choose a destination that does not already exist.
mkdir -p runs
test ! -e runs/artifact_analysis && cp -R results runs/artifact_analysis
python scripts/extract_paper_numbers.py --results-dir runs/artifact_analysis \
    --feature-set baseline_16 --model-size small --seeds 42 123 456 789 1024
python scripts/extract_paper_numbers.py --results-dir runs/artifact_analysis \
    --feature-set baseline_16 --model-size medium --seeds 42 123 456 789 1024
```

For new measurements, follow the full-run commands in the top-level README and
use a fresh directory under `runs/`. Exact target reuse is deliberately rejected
when the runtime environment or protocol differs. Analysis of saved JSONs does
not require reproducing the original execution environment.

## Runtime Scope

The recorded full suite completed in approximately 9 hours 20 minutes using ten
concurrent CPU workers, one training thread each, on an AMD EPYC 7702 host. The
container CPU quota was 15.3 CPU equivalents and its memory limit about 49.4 GiB.
This is an observed execution duration, not a promise for another host.

The main suite's `time_s` fields and runtime tables in `paper_numbers*.json`
reflect that parallel schedule. They must not be presented as isolated costs.
Contention is workload-dependent; do not apply a blanket percentage correction.

`runtime_isolated/` records a separate sequential measurement of the five
small-model configurations, seed 42, on the same AMD EPYC 7702 CPU host with one
training thread. All targets use the full 20-round protocol. Each round includes
both homes' training, aggregation, and evaluation. The original launch manifest,
script, process log, and zero exit status are retained in `provenance/` under
that directory. The invocation ran from 17:54 to 20:32 UTC on 15 September 2026,
approximately 2 hours 38 minutes including setup between configurations.

The runtime summary uses the median of 20 round durations per configuration.
Estimated run minutes equal that median times 20 divided by 60; these are not
measured end-to-end latencies or multi-seed uncertainty estimates. FedDPA has
two five-epoch stages plus Fisher estimation, unlike the other configurations.
These are simulation-host costs, not measurements on residential gateways.

| Configuration | Median round (s) | Estimated run (min) |
|---|---:|---:|
| Baseline FL | 54.6 | 18.2 |
| FS-mild | 54.7 | 18.2 |
| FS-aggressive | 54.7 | 18.2 |
| DP-SGD | 184.7 | 61.6 |
| FedDPA | 124.4 | 41.5 |

DP-SGD's median-round ratio to FS-mild is approximately 3.38. Recompute the
unrounded values with a read-only command:

```bash
python scripts/extract_runtime_numbers.py
```

The extractor checks completion, launcher and checkpoint hashes, all 20 rounds,
and exact protocol agreement with the main suite's corresponding seed-42 targets.
It prints JSON to standard output and never changes the stored snapshot.
`scripts/verify_artifact.py` also recomputes this summary when checking the
artifact. Do not substitute the timing runs' utility values for the five-seed
results or use their checkpoints for the main membership probes.

## Protocol Identity

The scientific pipeline source hash covers `fl_pipeline/*.py`; it does not
identify documentation or artifact-packaging changes. The job
launcher hash and execution arguments are preserved in `results/provenance/execution.json`.

Publishing seeded checkpoints and datasets is not a differentially private
release. See `docs/methods.md` and `data/README.md` for privacy scope and dataset
disclosure limitations. In particular, the shadow probe is a controlled
target-training-pool diagnostic, and FedDPA uses a different privacy unit from
DP-SGD.
