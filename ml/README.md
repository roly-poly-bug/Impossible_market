# Machine Learning

This area contains data and evaluation infrastructure for offline recommendation
experiments. It still contains no recommender, training loop, negative sampler,
learned feature, or inference implementation.

- `data/`: frozen-snapshot loading, observed Event aggregation, time splits,
  task states, relevance/seen/candidate sets, validation, export, and audit
- `models/`: future model definitions
- `training/`: future training entry points and experiment configuration
- `evaluation/`: model-independent Top-K metric helpers
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
