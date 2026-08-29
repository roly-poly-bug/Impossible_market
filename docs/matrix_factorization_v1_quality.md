# Matrix Factorization v1 Quality Report

## Scope and semantics

BCE MF learns target 1 for Train Binary View+ and target 0 for sampled Unknown using `BCEWithLogitsLoss`. Target 0 is a sampled training non-positive, not true dislike.

BPR MF learns `score(u, positive) > score(u, sampled unknown)` with `-logsigmoid(score_positive-score_unknown)`. This is a pairwise training assumption, not proof that the Unknown item is disliked.

Observed Non-conversion pairs are excluded from v1 sampling. Hidden user preferences, Product attributes, archetypes, preference_match, Validation events, and Test events are not training features.

## Fixed training configuration

- Architecture: bias-free `user_embedding · item_embedding`; latent dimension `16`
- Positive signal: Train Binary View+ (18,591 pairs)
- Sampling: `4` distinct Random Unknown items per positive, seed `42` (74,364 BPR triples)
- Adam: learning rate `0.001`, weight decay `0.0001`
- Batch size `1024`, max epochs `100`, patience `5`
- Early stopping and checkpoint selection: Validation Purchase NDCG@10 only
- Test: one final full-ranking pass after both best checkpoints were fixed
- Candidate and task-specific seen policies: unchanged from Recommendation Dataset v1

## Training behavior

| model | epochs run | best epoch | first loss | best-epoch loss | best Val Purchase NDCG@10 |
| --- | --- | --- | --- | --- | --- |
| bce | 11 | 6 | 0.692869 | 0.595317 | 6.9228% |
| bpr | 26 | 21 | 0.692172 | 0.474904 | 8.8957% |

## Validation and Test results at K=10

| model | split | task | Recall@10 | NDCG@10 | HitRate@10 | Precision@10 | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bce | validation | purchase | 16.6667% | 6.9228% | 19.6532% | 2.0231% | 173 |
| bce | validation | viewplus | 7.4218% | 7.7262% | 47.0389% | 6.5482% | 591 |
| bce | validation | favoriteplus | 9.2142% | 5.3492% | 20.3426% | 2.4197% | 467 |
| bce | test | purchase | 11.0377% | 5.3432% | 16.0377% | 1.6981% | 212 |
| bce | test | viewplus | 8.4946% | 8.8146% | 48.6088% | 7.2177% | 611 |
| bce | test | favoriteplus | 9.5143% | 5.3748% | 21.4433% | 2.3299% | 485 |
| bpr | validation | purchase | 16.4740% | 8.8957% | 20.8092% | 2.1387% | 173 |
| bpr | validation | viewplus | 6.1745% | 6.3001% | 42.1320% | 5.4146% | 591 |
| bpr | validation | favoriteplus | 8.4320% | 5.5001% | 20.1285% | 2.1413% | 467 |
| bpr | test | purchase | 9.0409% | 5.1212% | 12.2642% | 1.4623% | 212 |
| bpr | test | viewplus | 7.0280% | 7.1166% | 41.4075% | 5.9902% | 611 |
| bpr | test | favoriteplus | 7.7470% | 4.8170% | 17.7320% | 1.9588% | 485 |

## Purchase Test comparison

| model | K | Recall | NDCG | HitRate | Precision | eligible |
| --- | --- | --- | --- | --- | --- | --- |
| cart_popularity | 5 | 8.1132% | 5.2753% | 12.2642% | 2.4528% | 212 |
| bce_mf | 5 | 4.8742% | 3.0291% | 6.1321% | 1.2264% | 212 |
| bpr_mf | 5 | 5.5818% | 3.9101% | 8.4906% | 1.9811% | 212 |
| cart_popularity | 10 | 14.4969% | 7.6119% | 21.6981% | 2.2642% | 212 |
| bce_mf | 10 | 11.0377% | 5.3432% | 16.0377% | 1.6981% | 212 |
| bpr_mf | 10 | 9.0409% | 5.1212% | 12.2642% | 1.4623% | 212 |
| cart_popularity | 20 | 23.3962% | 10.0015% | 32.5472% | 1.7453% | 212 |
| bce_mf | 20 | 20.0393% | 7.7385% | 26.8868% | 1.4387% | 212 |
| bpr_mf | 20 | 19.2925% | 7.9097% | 24.5283% | 1.4151% | 212 |

## Coverage and diagnostics

- Train View+ zero-interaction Users: `29`. They use the Train-only Cart popularity fallback rather than untrained random user embeddings.
- Train View+ zero-interaction Items: `0`.
- All evaluation scores use full task candidate sets, never sampled ranking.

### BCE diagnostics

- User embedding norm mean/std/min/median/max: 0.4947 / 0.1964 / 0.0000 / 0.5061 / 0.9842
- Item embedding norm mean/std/min/median/max: 0.9484 / 0.1950 / 0.4160 / 0.9843 / 1.3484
- Score mean/std/min/median/max: -0.4043 / 0.1940 / -1.2332 / -0.3976 / 0.0040
- Unique Purchase Top10 lists: 206/212; average pairwise overlap: 76.5533%
- Mean overlap with Cart Popularity Top10: 29.3396%
- All embedding and score values finite: `True`

### BPR diagnostics

- User embedding norm mean/std/min/median/max: 0.8652 / 0.2797 / 0.0000 / 0.8818 / 1.6239
- Item embedding norm mean/std/min/median/max: 1.8130 / 0.3131 / 0.9579 / 1.8009 / 2.6062
- Score mean/std/min/median/max: -0.0092 / 0.4680 / -2.7319 / -0.0066 / 2.8543
- Unique Purchase Top10 lists: 206/212; average pairwise overlap: 13.1968%
- Mean overlap with Cart Popularity Top10: 20.9906%
- All embedding and score values finite: `True`

## Observed issues

- Neither fixed-config MF objective beats Cart Popularity on Purchase Test Recall@10 or NDCG@10.
- BPR has the stronger Validation Purchase NDCG@10 but drops more on Test, so the Validation advantage is not stable in this single temporal split.
- BCE recommendations are personalized by list identity, but their high average pairwise Top10 overlap shows a strong shared-ranking component.
- The 29 Users without Train View+ positives require the documented Train-only popularity fallback.

## Interpretation and next phase

- `BCE` is the recommended main v1 objective because it has the higher Purchase Test Recall@10 and NDCG@10 of the two fixed objectives. This is a single fixed-config result, not a hyperparameter conclusion.
- Compare both MF results with Cart Popularity before claiming personalization improved ranking quality.
- The next single representation experiment should be Log View: it preserves broad View coverage while compressing repeats. Favorite+ can follow; Weighted should remain later because it changes multiple assumptions at once.
- No parameter was changed after seeing Test. Test was not used for early stopping or model selection.
