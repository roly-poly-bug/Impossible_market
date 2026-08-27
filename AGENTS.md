# Project Guidelines

## Project
Impossible Market is a fictional e-commerce platform designed
to study recommendation systems and ML integration.

## Responsibilities

Codex may implement:
- frontend
- backend
- database
- APIs
- tests
- infrastructure

Recommendation-system models should not be implemented
without explicit instruction.

## Git

- Keep main stable.
- Create feature branches for substantial changes.
- Make small, meaningful commits.
- Do not commit secrets or .env files.
- Check git diff before committing.
- Do not force push.
- Do not merge into main without explicit instruction.

## ML

ML code should remain separated from the web application.
Prefer a structure such as:

ml/
  data/
  models/
  training/
  evaluation/
  inference/