# Method Specification

## Data and Preprocessing

The two Parquet files are filtered in this order: at least two bidirectional packets, `confidence == "DPI"`, and membership in the fixed six-class list in `config.py`. Categories are encoded in alphabetical order. No class reweighting or oversampling is applied.

Each home is split independently. Main experiments use seeded stratified 80/20 partitions. Temporal experiments sort by `bidirectional_first_seen_ms` within each home (stable tie ordering), taking the first 80% of retained records for training. Test support is recorded per class. Temporal boundaries can separate records with identical timestamps and do not establish independence between related sessions or devices.

The transform `flow_log_units_v1` is applied independently to each record:

| Feature family | Transform |
|---|---|
| Duration and inter-arrival times in milliseconds | `log1p(value / 1000)` |
| Byte counts and packet-size statistics in bytes | `log1p(value / 1024)` |
| Packet counts | `log1p(value)` |
| IP protocol number | `value / 255` |

These constants express seconds, KiB, counts, and the protocol field range; they are not estimated from either home. Non-finite values, negative values, and out-of-range protocol numbers cause an error. No exact private mean, variance, quantile, or cross-record scale is released or used. Feature selection slices the same per-record representation in every configuration.

## Training and Evaluation

Both homes participate in every round. FedAvg uses training-size weights, with an equal-weight variant for the three primary small-model configurations. Models use ReLU and cross-entropy. Hidden widths are 16-16 or 128-64 and do not depend on the input feature count. Each round uses a fresh local Adam optimizer. Defaults are 20 rounds, five local epochs, batch size 256, and learning rate 0.001. No accuracy or privacy ordering is assumed by the tests.

Combined macro-F1 is the equal-home average of the two macro-F1 values, regardless of FedAvg weights. Worst-group F1 is the minimum over all home/class pairs **within each seed**, followed by averaging across seeds. Paired intervals use seed-aligned differences and Student's t distribution with sample standard deviations. Five seeds estimate run variability, not uncertainty across the population of households.

CUDA timings are synchronized. Per-round elapsed times include local training, aggregation and evaluation; accounting setup contributes to total target elapsed time. The extraction reports observed run times separately from estimates based on the median round. Sleep, throttling and contention can contaminate wall-clock measurements; no timing samples are silently replaced.

Independent CPU experiments may use a concurrent worker schedule with one training thread per process. The launcher records the worker count and job timings separately from each model's specification. Targets precede their membership probes, and aggregate outputs are generated only after all measurement jobs finish. Concurrent timings are not isolated execution costs; neither absolute times nor relative runtime ratios should be assumed invariant to the schedule.

## DP-SGD

Opacus uses flat per-example gradient clipping at norm 1, Gaussian gradient noise and Poisson sampling. For a home with N training records and nominal batch size B, the actual sampling rate is `1 / ceil(N / B)`, matching the wrapped loader. Expected batch size is therefore `N / ceil(N / B)`, not necessarily B. Empty batches still perform a noise-only optimizer step.

One RDP accountant per home persists across all rounds. Calibration covers `rounds * local_epochs * ceil(N / B)` steps. Each home's target is epsilon 8 with `delta = min(1e-5, 1/(10*N))`. Sampling rates, step counts, deltas, noise multipliers and cumulative epsilons are recorded. Privacy is per training flow, with add/remove adjacency and fixed/public partition sizes, roster and aggregation weights as protocol metadata. The guarantee is conditional on the supplied training partition; the data-dependent selection and split procedure is not itself claimed to be DP.

Because preprocessing is record-local, changing one training record does not transform all other records. Aggregation and subsequent processing of already privatized updates preserve the mechanism's guarantee. Exact private test losses and labels, released datasets, and the composition of multiple experimental runs are not included in a single run's training budget.

Primary references: [Abadi et al., 2016](https://arxiv.org/abs/1607.00133), [Mironov, 2017](https://arxiv.org/abs/1702.07476), and the [Opacus privacy engine](https://opacus.ai/api/privacy_engine.html).

## Fisher-Personalized Update Baseline (FedDPA)

The baseline follows the dynamic personalization structure of [Yang, Huang and Ye, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/e4724af0e2a0d52ce5a0a4e084b87f59-Paper-Conference.pdf). Its implementation choices are specified here rather than assumed to be identical to every setting in the paper or [author code](https://github.com/xiyuanyang45/DynamicPFL).

1. Estimate diagonal empirical Fisher values on up to 5,000 local examples using the **mean of squared per-example** log-likelihood gradients.
2. Normalize each parameter tensor by its local minimum and maximum. Values at least 0.4 are personal; the rest are shared. Constant tensors produce zero normalized values.
3. Initialize personal coordinates from the previous local model and shared coordinates from the current global model.
4. Train the personal coordinates for five local epochs, then shared coordinates for five local epochs. Each stage has a fresh Adam optimizer; inactive gradients are masked out.
5. Relative to the personalized initialization, use penalties `lambda/2 * ||d_personal||_2` and `lambda/2 * abs(||d_shared||_2 - C)`, with lambda 0.05 and C 1. The norms are taken over the concatenated active parameter vector.
6. Clip the **entire** local-minus-global update to norm C and add independent Gaussian noise to every coordinate before transmission. The personal/shared support is not transmitted. Aggregate the noisy full updates with FedAvg weights.
7. Evaluate each home on its local post-training model. Only the global aggregate is saved as a target checkpoint; the local personalized models remain outside the release.

The original paper reports dataset-specific grid search over tau, lambda_1 and lambda_2 in {0.05, 0.1, 0.3, 0.5}; its separate threshold ablation also evaluates tau 0.4. The author code defaults to tau 0.4, lambda_1 0.1 and lambda_2 0.05. Here we fix tau at 0.4 and both stage coefficients at 0.05 through the shared `lambda_reg` parameter, without dataset-specific tuning. These are fixed experimental settings, not a claim to reproduce an optimally tuned FedDPA baseline.

These two training stages mean ten local data passes per round with the default five epochs per stage. Their cost must not be described as identical training work to five-epoch DP-SGD.

### Release and Accounting

Each client release is independently observable; secure aggregation and amplification from client subsampling are **not** assumed. Replacing a client's entire training dataset/state can move two clipped vectors by at most **2C**. The mechanism therefore adds per-coordinate noise with standard deviation `2*C*sigma`, calibrating sigma for full participation (`q=1`) over all rounds. This is a client-dataset replacement accounting convention, unlike DP-SGD's flow-level convention. The selected delta follows the recorded per-home formula but must be interpreted with this privacy unit. The author implementation's distributed-noise scaling is not assumed for these independently observed client releases.

The budget applies to the noisy update transcript under fixed roster and weights, not to personalized models, their reported private-data metrics, masks, or raw data. Persistent local personalization is allowed inside the bounded local update function; support-dependent noiseless coordinates are not released. Test metric publication is an experimental evaluation on the released research data, not an additional DP mechanism.

A common numeric epsilon does not make this baseline privacy-equivalent to DP-SGD. Its performance cannot establish that all update-level or adaptive DP mechanisms fail in small federations. It is a scoped exploratory baseline with the above release model and adaptation choices.

## Membership Probes

Both probes load the exact primary stratified target checkpoints, not independently retrained utility surrogates. Per-home, per-class evaluation uses equal member/nonmember counts up to 2,000 each, with deterministic seed-based indices. Class metrics are macro-averaged, then homes receive equal weight. AUC, TPR at 1% and 5% FPR, and maximum empirical ROC advantage are descriptive attack metrics. Thresholds read from the target ROC are not independently calibrated operational attack thresholds; maximum advantage is an in-sample threshold maximum.

The loss probe uses negative cross-entropy as the membership score. The shadow probe fits home/class-specific logistic regressions to loss, confidence, entropy, true-class probability and margin. Each of four shadows trains on a random half of **each target training partition**; the remaining half supplies that shadow's known nonmembers. It uses the corresponding architecture, features, FL schedule and training mechanism; DP calibration uses the shadow's actual size. Target test records are excluded from shadow training and attack fitting.

This resampling construction is a controlled diagnostic with training-pool access, **not** a realistic independently sourced auxiliary-data attacker. Shadow/target training-size mismatch and related data can influence attack transfer. Neither near-chance AUC nor an interval spanning zero proves privacy equivalence. Cross-run paired intervals are descriptive, with multiple comparisons and small sample size to be considered when interpreting isolated differences.

## Artifacts and Reproducibility

Run specifications include dataset and partition hashes, feature transforms, class order, architecture, seed, hyperparameters, source hash, library versions and execution environment. Checkpoints have checksums; caches with a different specification fail. Source and environment changes require separate run directories. Completed shadows are individually resumable; incomplete training jobs restart from round one.

All runs use seeded pseudorandomness and request deterministic PyTorch algorithms. These are reproducible **research simulations** of noisy mechanisms on released data, not production releases using secret cryptographic randomness. A public noise seed must not be used to claim a private model release. The reported accountant values describe the mathematical mechanisms under their randomness assumptions, not end-to-end privacy of this public artifact or joint protection across its many runs.
