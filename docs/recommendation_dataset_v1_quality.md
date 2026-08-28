# Recommendation Dataset v1 Quality Report

This report audits observed-fact dataset construction. It does not train or evaluate a recommendation model.

## Frozen inputs

- Dataset: `recommendation_dataset_v1`, seed `42`
- Products: `synthetic_product_v1`, seed `42`
- Users: `synthetic_user_v1`, seed `42`
- Session/Exposure/View: `synthetic_session_event_v1`, seed `42`
- Engagement: `synthetic_engagement_v1`, seed `42`

## Overall

- Users: 1,000
- Items: 200
- Raw Events: 131,772
- Observed user-item pairs: 71,683 / 200,000

## Temporal split Event counts

| split | events | impressions | views | favorites | carts | purchases | unique pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 85,711 | 61,134 | 19,890 | 2,792 | 1,164 | 731 | 51,109 |
| validation | 22,690 | 16,080 | 5,293 | 773 | 323 | 221 | 15,473 |
| test | 23,371 | 16,496 | 5,354 | 820 | 394 | 307 | 15,746 |

## Task positives and eligible users

| task | split | positive pairs | eligible users |
| --- | --- | --- | --- |
| viewplus | train | 18,591 | 971 |
| viewplus | validation | 5,211 | 591 |
| viewplus | test | 5,234 | 611 |
| favoriteplus | train | 3,944 | 873 |
| favoriteplus | validation | 1,133 | 467 |
| favoriteplus | test | 1,277 | 485 |
| purchase | train | 731 | 431 |
| purchase | validation | 221 | 173 |
| purchase | test | 307 | 212 |

## Three-state distribution

| split | task | positive | observed non-conversion | unknown |
| --- | --- | --- | --- | --- |
| train | viewplus | 18,591 (9.30%) | 32,518 (16.26%) | 148,891 (74.45%) |
| train | favoriteplus | 3,944 (1.97%) | 14,647 (7.32%) | 181,409 (90.70%) |
| train | purchase | 731 (0.37%) | 17,860 (8.93%) | 181,409 (90.70%) |
| validation | viewplus | 5,211 (2.61%) | 10,136 (5.07%) | 184,653 (92.33%) |
| validation | favoriteplus | 1,133 (0.57%) | 4,215 (2.11%) | 194,652 (97.33%) |
| validation | purchase | 221 (0.11%) | 5,031 (2.52%) | 194,748 (97.37%) |
| test | viewplus | 5,234 (2.62%) | 10,385 (5.19%) | 184,381 (92.19%) |
| test | favoriteplus | 1,277 (0.64%) | 4,094 (2.05%) | 194,629 (97.31%) |
| test | purchase | 307 (0.15%) | 4,975 (2.49%) | 194,718 (97.36%) |

## Positives per user

| split | task | mean | std | min | median | max | zero count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | viewplus | 18.591 | 12.059 | 0 | 17.0 | 63 | 29 |
| train | favoriteplus | 3.944 | 3.254 | 0 | 3.0 | 18 | 127 |
| train | purchase | 0.731 | 1.060 | 0 | 0.0 | 6 | 569 |
| validation | viewplus | 5.211 | 6.008 | 0 | 4.0 | 40 | 409 |
| validation | favoriteplus | 1.133 | 1.592 | 0 | 0.0 | 12 | 533 |
| validation | purchase | 0.221 | 0.526 | 0 | 0.0 | 3 | 827 |
| test | viewplus | 5.234 | 6.212 | 0 | 4.0 | 46 | 389 |
| test | favoriteplus | 1.277 | 1.848 | 0 | 0.0 | 15 | 515 |
| test | purchase | 0.307 | 0.698 | 0 | 0.0 | 5 | 788 |

## Positives per item

| split | task | mean | std | min | median | max | zero count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | viewplus | 92.955 | 27.856 | 41 | 86.0 | 168 | 0 |
| train | favoriteplus | 19.720 | 9.522 | 5 | 17.0 | 60 | 0 |
| train | purchase | 3.655 | 3.206 | 0 | 3.0 | 15 | 38 |
| validation | viewplus | 26.055 | 9.086 | 9 | 25.0 | 51 | 0 |
| validation | favoriteplus | 5.665 | 3.517 | 0 | 5.0 | 21 | 4 |
| validation | purchase | 1.105 | 1.362 | 0 | 1.0 | 8 | 84 |
| test | viewplus | 26.170 | 10.263 | 6 | 24.0 | 65 | 0 |
| test | favoriteplus | 6.385 | 3.786 | 0 | 6.0 | 20 | 4 |
| test | purchase | 1.535 | 1.578 | 0 | 1.0 | 7 | 63 |

Purchase-only train items with 1–2 positives (cold-ish): 45.

## Time leakage audit

- Every Event belongs to exactly one half-open UTC split.
- Aggregated first/last timestamps remain inside their split.
- Validation/Test conversions are not added to Train facts.
- Future-conversion pairs with prior Train history checked: 931.
- Violations: **0**

## Interpretation

- Implicit Feedback has no reliable true negative.
- Positive means the task event was observed; observed non-conversion means the prerequisite opportunity was observed without conversion.
- Unknown means the prerequisite opportunity was not observed. Unknown is never converted to a negative label.
- A sampled negative in a later experiment would not be a true negative.
- No event weights, hidden user preferences, product ground-truth attributes, or future Events are included as training features.
