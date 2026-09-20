# Feature Suppression and Differential Privacy for Residential Traffic Classification: A Two-Home Federated Study

Research artifact accompanying the paper of the same title.

The pipeline studies traffic classification on two residential gateways, with fixed feature suppression, DP-SGD, and a Fisher-personalized update-noise baseline. Utility and attack results are measurements in this setting, not interchangeable privacy guarantees.

## Included Measurements

The complete measurement snapshot is in [results](results/), with metrics, both capacities and split protocols, attack outputs, and 205 target/shadow checkpoints from the main suite. Five additional checkpoints belong to the separate [isolated timing measurement](results/runtime_isolated/). Generated plots are in [figures](figures/). See the [measurement guide](docs/measurements.md) for table inputs, provenance, verification, and analysis without retraining.

```bash
python scripts/verify_artifact.py
```

The full suite completed in **9 hours 20 minutes** on an AMD EPYC 7702 host using ten concurrent single-threaded CPU workers. These measurements are not isolated runtime benchmarks. [repro_manifest.json](repro_manifest.json) records the measurement files, checksums, source identity, and execution scope.

The paper's runtime table instead uses a sequential, single-threaded CPU measurement of all five small-model configurations at seed 42. Median round times are 54.6 s (baseline), 54.7 s (FS-mild), 54.7 s (FS-aggressive), 184.7 s (DP-SGD), and 124.4 s (FedDPA). DP-SGD is approximately **3.4 times** FS-mild in this measurement; FedDPA performs two local stages plus Fisher estimation. Recompute the timing summary without writing to the artifact:

```bash
python scripts/extract_runtime_numbers.py
```

## Experiment Matrix

| Configuration | Features | Training |
|---|---:|---|
| Baseline FL | 16 | FedAvg |
| FS-mild | 12 | FedAvg, omitting bidirectional inter-arrival statistics |
| FS-aggressive | 8 | FedAvg, also omitting directional counts |
| DP-SGD | 16 | Per-example gradient clipping and Gaussian noise |
| FedDPA | 16 | Fisher personalization, two-stage optimization, full-update Gaussian release |

Small models use fixed hidden widths **16-16**; medium models use **128-64**. Full-feature parameter counts are 646 and 10,822. Input dimensions, but not hidden widths, change with feature suppression. Preprocessing is a fixed per-record logarithmic transform, with no fitted population statistics. See [method details and privacy scope](docs/methods.md).

The default full run includes:

| Experiment | Configurations | Seeds | Target trainings |
|---|---|---:|---:|
| Small, stratified split | All five | 5 | 25 |
| Small, equal FedAvg weights | Baseline FL, FS-mild, DP-SGD | 5 | 15 |
| Medium, stratified split | Baseline FL, FS-mild, DP-SGD | 5 | 15 |
| Small, temporal split | Baseline FL, FS-mild, DP-SGD | 5 | 15 |
| Medium, temporal split | Baseline FL, FS-mild, DP-SGD | 5 | 15 |

Both membership probes evaluate the same 30 primary, size-proportional, stratified-split target checkpoints across the two capacities. Four shadows per target add **120 shadow trainings**. No medium-model equal-weight or FedDPA run, and no temporal MIA, is included in this matrix.

## Setup

Use Python 3.11.14 and fetch the two Parquet files using Git LFS:

```bash
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
python -m unittest discover -s tests -v
```

For CUDA, install a PyTorch 2.10 build compatible with the machine's driver using the [official PyTorch installer](https://pytorch.org/get-started/locally/). Verify `torch.cuda.is_available()` before selecting `--device cuda`. CPU is the default; MPS is not used. Do not assume that GPU acceleration makes this small, sequential workload fit a particular time budget.

The [dataset documentation](data/README.md) describes the release. Removing IP addresses does **not** make the data anonymous or differentially private; MAC addresses and other linkable metadata remain.

## Full Run

From the repository root, using the activated environment:

```bash
# Inspect the complete command sequence without training or writing outputs.
bash scripts/reproduce.sh --device cpu --results-dir runs/residential_v1 --dry-run

# Execute the complete matrix, including FedDPA, temporal checks, and both attacks.
bash scripts/reproduce.sh --device cpu --results-dir runs/residential_v1

# Same matrix on a CUDA machine; use a separate directory for each environment.
bash scripts/reproduce.sh --device cuda --results-dir runs/residential_cuda
```

`PYTHON=/path/to/python bash scripts/reproduce.sh ...` selects a specific interpreter. The script does not install packages, download data, rent hardware, or push results.

Each stage can be run separately. Preserve the same device, thread count, data, code, and training arguments across stages:

```bash
python scripts/run_suite.py --stage training --device cuda --results-dir runs/residential_cuda
python scripts/run_suite.py --stage temporal --device cuda --results-dir runs/residential_cuda
python scripts/run_suite.py --stage mia --device cuda --results-dir runs/residential_cuda
python scripts/run_suite.py --stage shadow --device cuda --results-dir runs/residential_cuda
python scripts/run_suite.py --stage analysis --device cuda --results-dir runs/residential_cuda
python scripts/run_suite.py --stage figures --device cuda --results-dir runs/residential_cuda
```

`training` must finish before `mia` and `shadow`. `temporal` is independent of the attacks. `analysis` and `figures` should follow the required measurements. `--capacity small` and `--capacity medium` select disjoint jobs, suitable for separate machines with separate output directories. Do not concurrently run the same job in the same directory.

## Runtime and Resume

For multi-core CPU hosts, the dependency-aware queue runs independent jobs concurrently:

```bash
python scripts/run_parallel_suite.py --device cpu --num-threads 1 --workers 10 \
    --seeds 42 123 456 789 1024 --n-shadows 4 --max-hours 36 \
    --results-dir runs/residential_cpu --dry-run
python scripts/run_parallel_suite.py --device cpu --num-threads 1 --workers 10 \
    --seeds 42 123 456 789 1024 --n-shadows 4 --max-hours 36 \
    --results-dir runs/residential_cpu
```

Choose the worker count from measured throughput and container CPU/memory limits, not the host's CPU count. The queue continuously replaces finished workers and prioritizes long jobs and targets needed by shadow attacks. Optional `--timing-reference runs/benchmark_cpu/benchmark.json` supplies measured scheduling estimates. Each shadow-attack job trains its four shadows sequentially. Workers write distinct artifacts; shared summaries, extraction and figures run serially after all measurement jobs finish. A worker failure stops the invocation rather than generating partial final summaries.

The launcher records its schedule, per-job logs and `progress.json` under `parallel_*` in the run directory. It locks that directory against a second parallel launcher; do not start separate manual commands there during execution. `--max-hours` bounds the invocation, including finalization, but does not stop a cloud pod or its billing. Repeat the identical command to reuse completed checkpoints after interruption. Parallel-run timings include contention and must be labeled accordingly; measure isolated per-method runtime separately when needed.

Benchmark the intended host before scheduling the full run:

```bash
python scripts/benchmark.py --device cuda --results-dir runs/benchmark_cuda
```

The benchmark measures short full-data targets and representative half-size shadow workloads. Its extrapolation is a scheduling estimate, not a result or a guarantee. Training, evaluation, accounting, Fisher computation, model loading, and attack inference have different costs; CPU/GPU timings are not interchangeable. Keep the machine awake throughout measurement.

Completed targets have JSON logs and `.pt` checkpoints. Rerunning an identical command reuses them. Both MIA probes load those exact checkpoints. Completed individual shadows are also reusable. An interrupted target or shadow restarts that model from round one; there is no mid-round resume.

Cache reuse checks data hashes, partition hashes, source hash, hyperparameters, execution environment, and checkpoint checksum. A mismatch fails rather than silently mixing runs. Use a new output directory for a different protocol or environment. Deterministic PyTorch execution is requested, but numerical identity across platforms or dependency versions is not promised.

## Outputs

New runs default to `runs/residential_v1/` (or `FL_RESULTS_DIR`). This directory is gitignored:

```text
runs/residential_v1/
  baseline_16/
    seed_42/{config}.json + .pt
    equal_weight_seed_42/{config}.json + .pt
    temporal/seed_42/{config}.json + .pt
    mia/seed_42_{config}.json
    shadow_mia/seed_42_{config}.json
    shadow_mia/models/                 # individual shadow checkpoints
    summary.json
    summary_equal_weight.json
  baseline_16_medium/                  # primary + temporal + both probes
  temporal_split_check.json
  temporal_split_check_medium.json
  paper_numbers.json
  paper_numbers_baseline_16_medium.json
  figures/{baseline_16,baseline_16_medium}/
  suite_*.json                         # actual per-command elapsed times
```

The utility extractor recomputes summaries from per-seed logs and validates attack-target correspondence. It reports sample standard deviations (`ddof=1`), paired intervals, and directional counts, not pass/fail decisions. Its runtime fields retain the main suite's concurrent measurement context. Use `scripts/extract_runtime_numbers.py` for the separate isolated timings. Estimated run time (median round time times round count) and observed elapsed run time are separate fields.

`results/` and `figures/` contain the versioned measurement snapshot. Treat them as read-only and copy `results/` to a fresh directory under `runs/` before regenerating tables or plots, as shown in the measurement guide. The pipeline rejects `results/` as a writable destination. Use each run's saved specification to establish which protocol generated it; do not combine different protocols or environments into one table.

## Privacy and Interpretation

FS is data minimization, **not DP**. DP-SGD's accounting concerns individual training flows, conditional on fixed partitions and public protocol metadata. The FedDPA implementation accounts for independently released full client updates under client-dataset replacement; its unit and sensitivity differ from DP-SGD. Matching epsilon alone does not equate these guarantees.

Public datasets, private evaluation labels, personalized FedDPA models/metrics, and the joint release of all experimental runs are not covered by a single reported training epsilon. Experiments use seeded pseudorandomness for reproducibility, not a production private release. Shadow MIA resamples the target's training partition; it is a controlled diagnostic, not an independent auxiliary-data attack. Near-chance attacks do not establish absence of leakage. Details and primary sources are in [docs/methods.md](docs/methods.md).

## Citation and License

See `CITATION.cff` and `LICENSE` (MIT).
