# MF Negative Sampling v1 Quality Report

## Scope and fixed MF conditions

This experiment changes only the sampled non-positive source for the frozen Weighted Implicit BCE MF. The score is a bias-free 16-dimensional User/Item dot product. Positive confidence remains `1 + log1p(log1p(view_count) + 3·favorite + 5·cart + 8·purchase)`.

Adam, learning rate 0.001, weight decay 0.0001, batch size 1,024, four samples per positive, seed 42, maximum 100 epochs, patience 5, Validation Purchase NDCG@10 selection, full task candidate sets, and task-specific seen exclusion are unchanged. Sampled target zero is a training contrast label, not evidence of dislike.

## Sampling definitions and backfill

- **Random Unknown:** four View+ Unknown items per positive. The frozen Weighted v1 result is reused without another Test pass.
- **Exposed Non-conversion:** up to four User-specific Train impression-without-View pairs per positive. Any shortage is filled only with View+ Unknown.
- **Mixed:** fixed two Exposed Non-conversion plus two Random Unknown items per positive. Any Exposed shortage is filled with Unknown.

Sampling never uses Validation or Test interactions. A positive pair is never sampled, and samples are unique within each positive's four-item set. Observed Non-conversion is more informative than Unknown but is not a true negative.

## Sampling statistics and concentration

| strategy | samples | unique user-item | exposed | unknown | backfill | backfill users | sampled items | Gini | Top10 share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Unknown | 74,364 | 51,320 | 0 | 74,364 | 0 | 0 | 200 | 0.0642 | 5.9962% |
| Exposed | 74,364 | 28,820 | 74,332 | 32 | 32 (0.0430%) | 4 | 200 | 0.0862 | 6.8810% |
| Mixed | 74,364 | 52,050 | 37,182 | 37,182 | 0 | 0 | 200 | 0.0380 | 5.8267% |

Exposed uses a smaller set of repeated User–Item contrasts and is modestly more concentrated, but all 200 items are sampled and concentration remains low.

## Hardness diagnostics

| source | mean item Cart count | median Cart count | mean Train exposure | median Train exposure |
| --- | ---: | ---: | ---: | ---: |
| Random Unknown | 5.6683 | 5 | 299.2217 | 289 |
| Exposed Non-conversion | 5.8649 | 5 | 313.0734 | 306 |
| Mixed overall | 5.7683 | 5 | 306.7301 | 298 |

Exposed samples come from items with 3.47% higher mean Cart popularity and 4.63% higher historical exposure than Random Unknown. This supports the operational interpretation that they are less trivial, more visible comparison items. Hidden preferences and Product attributes were not used.

## Training and Validation

| strategy | epochs | best epoch | first loss | best-epoch loss | Val Purchase Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Unknown | 12 | 7 | 0.8055 | 0.7732 | 17.1484% | 8.0559% |
| Exposed | 29 | 24 | 0.8052 | 0.6411 | 17.4374% | 9.7957% |
| Mixed | 15 | 10 | 0.8056 | 0.7285 | 15.3179% | 7.2179% |

All loss, embedding, and score values are finite. Test is not present in training, early stopping, or checkpoint selection.

## Test results at K=10

| strategy | Purchase Recall | Purchase NDCG | View+ Recall | View+ NDCG | Favorite+ Recall | Favorite+ NDCG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Unknown | 11.0613% | 5.7687% | 8.4657% | 8.4802% | 9.4165% | 5.6934% |
| Exposed | 12.2406% | 6.5507% | 6.9087% | 7.1798% | 8.6052% | 4.8637% |
| Mixed | 11.9654% | 5.6809% | 8.2287% | 8.5913% | 8.8400% | 5.3052% |
| Cart Popularity | 14.4969% | 7.6119% | — | — | — | — |

Exposed improves Random Purchase Recall@10 by 1.1792 percentage points and NDCG@10 by 0.7820 points. Mixed improves Recall by 0.9041 points but reduces NDCG by 0.0879 points. Neither beats Cart Popularity.

## Validation/Test stability

- Random: Recall -6.0870 points; NDCG -2.2872 points.
- Exposed: Recall -5.1968 points; NDCG -3.2451 points.
- Mixed: Recall -3.3525 points; NDCG -1.5370 points.

Mixed has the smallest temporal gap, while Exposed has the strongest absolute Test Purchase result but a larger NDCG gap than Random.

## Personalization and popularity overlap

| strategy | unique Top10 / 212 | User-to-User overlap | Cart Top10 overlap | recommended Cart mean |
| --- | ---: | ---: | ---: | ---: |
| Random | 206 | 53.2907% | 37.4528% | 13.4146 |
| Exposed | 206 | 26.4048% | 34.1509% | 12.5910 |
| Mixed | 206 | 71.5443% | 45.0472% | 14.5953 |

Exposed produces substantially more distinct rankings across Users and is less aligned with Cart Popularity. Mixed instead becomes more shared and popularity-oriented.

## Backfill User performance

Only four training Users require Exposed backfill. Among Test Purchase eligible Users, one is in that group and has no hit at K=10; the remaining 211 Users achieve Recall@10 12.2986% and NDCG@10 6.5817%. The one-User group is too small for a general performance conclusion. Mixed requires no backfill.

## Embedding and score diagnostics

| strategy | User norm mean | Item norm mean | score mean | score std | score min | score max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.3299 | 0.6520 | -0.1130 | 0.0971 | -0.7226 | 0.2635 |
| Exposed | 0.7894 | 1.7354 | -0.5100 | 0.3687 | -2.5432 | 1.4921 |
| Mixed | 0.4823 | 0.9875 | -0.3406 | 0.2115 | -1.4118 | 0.2780 |

Exposed learns larger norms and a wider score distribution, consistent with stronger contrasts, but values remain finite without evidence of explosion.

## Hypothesis, recommendation, and next phase

The main hypothesis is supported in this fixed experiment: Exposed comparisons are observably harder and improve both primary Purchase metrics over Random Unknown. This does not make an impression-without-View a negative truth; missed attention, position, session intent, and noise remain alternative causes.

Use **Exposed Non-conversion with Unknown backfill** as the next main sampling strategy. It has the strongest Test Purchase metrics, negligible backfill dependence, full sampled-item coverage, and stronger personalization. Mixed is the stability-oriented secondary result.

The next controlled experiment should compare Exposed/Unknown mixing ratios around this boundary without changing model capacity or confidence. Later isolated candidates are Cart-centered positives, bias terms, latent dimension, confidence adjustment, and BPR with harder comparisons. No setting was adjusted after Test.
