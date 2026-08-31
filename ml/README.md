# Machine Learning

This area contains data and evaluation infrastructure for offline recommendation
experiments. It still contains no recommender, training loop, negative sampler,
learned feature, or inference implementation.

- `data/`: frozen-snapshot loading, observed Event aggregation, time splits,
  task states, relevance/seen/candidate sets, validation, export, and audit
- `models/`: future model definitions
- `training/`: future training entry points and experiment configuration
- `evaluation/`: model-independent Top-K metric helpers
- `baselines/`: deterministic non-personalized controls
- `representations/`: observed user-item signal transforms for future experiments
- `experiments/`: reproducible offline experiment entry points and reports
- `inference/`: future model loading and backend adapters

Build the deterministic v1 snapshot from the repository root:

```bash
python -m ml.data.build_recommendation_dataset \
  --dataset-version recommendation_dataset_v1 \
  --output-dir data/recommendation_v1
```

The builder reads only observed Product identity/status, User identity, and
Event facts. Synthetic user true preferences, hidden product attributes,
preference-match values, prototypes, and future Events are not model features.

Implicit Feedback has no reliable true negative. The task datasets preserve
`positive`, `observed_non_conversion`, and `unknown` as distinct, mutually
exclusive states. Observed non-conversion is an opportunity without the target
action, not dislike. Unknown stays unknown. Likewise, any sampled negative in a
future experiment would not be a true negative.

No event weights are fixed in v1. This keeps View+, Favorite+, Purchase-only,
and future weighted-implicit experiments comparable without rewriting the raw
facts. Model selection and implementation remain a later collaborative phase.

Run Popularity Baseline v1 after building the Recommendation Dataset:

```bash
python -m ml.experiments.run_popularity_baselines \
  --dataset-dir data/recommendation_v1 \
  --output-dir results/popularity_v1
```

This experiment is deliberately global rather than personalized. Scores use
Train interactions only, then every eligible User starts with the same ranking
before task-specific candidate and seen policies are applied. The configurable
weighted v1 coefficients are an experimental hypothesis, not an optimized
label. Use `--include-seen` only with a separate output directory when comparing
the optional non-exclusion policy.

Matrix Factorization v1 is the first deliberately limited personalized model:

```bash
python -m pip install -r ml/requirements.txt
python -m ml.experiments.run_matrix_factorization_v1
```

Both BCE and BPR use the same Train Binary View+ positives and the same four
seeded Random Unknown samples per positive. Observed Non-conversion is excluded.
For BCE, target zero means sampled training non-positive, not true dislike; for
BPR, the learned ordering is positive above sampled Unknown, not a true-negative
claim. Validation Purchase NDCG@10 alone controls early stopping. Test never
selects a checkpoint or parameter.

MF Signal Representation v1 keeps that BCE architecture and optimization fixed
while comparing five Train signals:

```bash
python -m ml.experiments.run_mf_signal_representation_v1
```

Log View uses `1 + log1p(view_count)` positive confidence. Weighted uses the
fixed 1/3/5/8 strength followed by `1 + log1p(weighted_strength)`. Targets stay
binary, confidence is not mean-normalized, and every sampled Unknown remains a
training non-positive rather than a true negative.

MF Negative Sampling v1 freezes the Weighted BCE representation and compares
Random Unknown, Exposed Non-conversion with Unknown backfill, and a fixed 2/2
Mixed sampler:

```bash
python -m ml.experiments.run_mf_negative_sampling_v1
```

Observed Non-conversion means Train impression without View. It is used only as
a contrast label and is not interpreted as true dislike.

MF Bias v1 freezes Weighted/Exposed training and compares No Bias, Item Bias,
and User+Item Bias. Global bias is omitted because it cannot alter Top-K order.
Run `python -m ml.experiments.run_mf_bias_v1`.

MF Cart Signal v1 keeps the selected Item-Bias architecture and Exposed
sampling fixed while comparing Existing Weighted, Cart+, Favorite+Cart+, and
Cart-centered Weighted positive definitions. Exposed non-conversion remains a
Train-only contrast, not a true negative. Run
`python -m ml.experiments.run_mf_cart_signal_v1`; interpretation is recorded in
`docs/mf_cart_signal_v1_quality.md`.

MF Latent Dimension v1 changes only capacity across 8, 16, 32, and 64 hidden
coordinates. It shares the same positive pairs and sampled contrasts, reuses
the frozen dimension-16 checkpoint, and selects every new checkpoint only with
Validation Purchase NDCG@10. Run `python -m ml.experiments.run_mf_latent_dim_v1`.

MF Objective v2 freezes Existing Weighted confidence, the four-per-positive
Exposed/Unknown comparison triples, Item Bias, latent dimension 8, optimizer,
candidates, and seen exclusion. It compares the reused pointwise BCE checkpoint
with confidence-weighted pairwise BPR. An exposed non-conversion is only a
training contrast, never dislike or a true negative. Run
`python -m ml.experiments.run_mf_objective_v2`.
