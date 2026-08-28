# Recommendation Dataset v1

`recommendation_dataset_v1` is the deterministic observed-data layer between
the frozen Synthetic World and future recommendation experiments. It contains
no recommendation model, popularity baseline, negative sampling strategy,
event weighting, API, or UI.

## Input boundary

The builder consumes seed-42 snapshots for 200 Products, 1,000 Users, Session /
Impression / View Events, and Favorite / Cart / Purchase Events. Only Product
identity and lifecycle status, User identity, and observed Event fields are
read. Synthetic User preference vectors, Product hidden ground-truth
attributes, `true preference_match`, archetype prototypes, and future Events are
excluded from experiment features.

The implementation is under `ml/data/`, rather than `synthetic_data/`, because
it transforms a frozen world into experiment-specific views without changing
the generator or its frozen coefficients.

## Three-state feedback contract

Implicit Feedback has no reliable true negative.

| task | positive | observed non-conversion | unknown |
| --- | --- | --- | --- |
| View+ | View observed | Impression observed, no View | no Impression or View |
| Favorite+ | Favorite, Cart, or Purchase observed | View observed, none of those conversions | no View or stronger event |
| Purchase-only | Purchase observed | View observed, no Purchase | no View or Purchase |

The states are mutually exclusive. Observed Non-conversion is not a True
Negative: it says that an opportunity was observed without a target action, not
that the user disliked the Product. Unknown is retained as Unknown and is never
silently relabeled. Similarly, Sampled Negative != True Negative. Any future
sampling policy belongs to an individual model experiment.

No numeric strength such as `View=1, Favorite=2, Cart=3, Purchase=5` is stored.
The master retains raw counts and booleans, allowing binary View+, Favorite+,
Purchase-only, and later weighted-implicit definitions to be compared safely.

## Master table

`master_interactions.csv` has one row for each of the 1,000 x 200 user-item
pairs. It stores the five Event counts, `was_*` booleans, first timestamps for
all five types, last timestamps for all five types, and overall first/last
interaction timestamps. A row with no observations contains zero counts, false
flags, and blank timestamps; it does not mean negative.

The three `train_*.csv` files use only Train-period facts and add `state`,
`is_positive`, `is_observed_non_conversion`, and `is_unknown`. The three flags
sum to exactly one for every row.

## Temporal split and leakage control

All timestamps are UTC and intervals are half-open:

| split | start inclusive | end exclusive |
| --- | --- | --- |
| Train | `2026-01-01T00:00:00Z` | `2026-01-21T00:00:00Z` |
| Validation | `2026-01-21T00:00:00Z` | `2026-01-26T00:00:00Z` |
| Test | `2026-01-26T00:00:00Z` | `2026-01-31T00:00:00Z` |

Each Event belongs to exactly one split. Facts are aggregated independently per
split. Therefore a Jan 28 Purchase cannot enter Train facts for a pair viewed on
Jan 10. Validation and Test relevance are formed only from positive Event types
inside their own period.

## Evaluation artifacts and policies

- Relevance files are JSON mappings from every `user_id` to a sorted list of
  relevant Product IDs. An eligible evaluation User has at least one relevant
  item in that task and split; metrics with an empty relevance set are not
  computed.
- Task-specific `train_seen_items` are the same-task Train positives. The
  documented v1 default is `exclude_seen=true`, but the lists remain separate
  so an experiment can choose `false` or a different cross-task policy. In
  particular, a Train View followed by a Test Purchase stays visible in the raw
  facts and can be treated differently by a Purchase experiment.
- View+ and Favorite+ all-item candidates include all 200 catalog Products,
  including `coming_soon`, `sold_out`, and `unavailable`, because those tasks can
  measure interest independent of immediate purchaseability.
- Purchase-only candidates include the 170 Products whose status is
  `available`; `coming_soon`, `sold_out`, and `unavailable` are excluded.
- `impressed_candidates_{split}.json` preserves per-User exposed items for
  future exposed-item ranking, without treating non-clicked items as true
  negatives.

Metric helpers in `ml/evaluation/metrics.py` implement Recall@K, NDCG@K,
HitRate@K, and Precision@K for any positive K, including 5, 10, and 20. Recall
and NDCG reject empty relevance sets, enforcing the eligible-User definition.

## Build and tracking

```bash
python -m ml.data.build_recommendation_dataset \
  --dataset-version recommendation_dataset_v1 \
  --output-dir data/recommendation_v1
```

Generated CSV/JSON artifacts are deterministic and Git-ignored because several
are large. `data/recommendation_v1/manifest.json` is tracked and records source
versions and hashes, split boundaries, policies, artifact row counts, sizes,
and hashes. `docs/recommendation_dataset_v1_quality.md` is also tracked. The
snapshot can therefore be regenerated and checked without committing bulky
tables.
