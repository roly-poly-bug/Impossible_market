# Machine Learning

This directory is intentionally limited to structure and documentation. No
recommendation model or inference logic has been implemented.

- `data/`: local or generated ML inputs (large artifacts are ignored by Git)
- `models/`: serialized model artifacts (ignored by Git)
- `training/`: future training code
- `evaluation/`: offline metrics and experiment evaluation
- `inference/`: model-loading and prediction adapters called by the backend

ML code remains separate from the web application and will be added only after
its design and learning goals are agreed upon.
