# Popularity Baseline v1 Quality Report

This report evaluates non-personalized global rankings built only from the Recommendation Dataset v1 Train split.

## Experiment contract

- Experiment: `popularity_baseline_v1`
- Dataset: `recommendation_dataset_v1`, seed `42`
- Score source: Train only; Validation/Test events never affect popularity scores.
- Personalization: none. Every eligible User starts from the same global ranking.
- Main evaluation: task-specific candidates, task-specific Train seen exclusion enabled, identical relevance and metrics across signals.
- Tie-break: score descending, then `product_id` ascending.
- Cold candidates remain with score 0.
- View+/Favorite+ candidates: all 200 Products; Purchase candidates: 170 `available` Products.

A zero interaction representation means no positive signal was observed for that representation. It is not a true negative, and Unknown is not converted to a negative label.

## Popularity signals

- `total_view_count`: sum of all Train View events, including repeated Views.
- `unique_view_users`: number of Train Users with at least one View.
- `log_view_strength`: sum of `log1p(view_count_ui)` across Users.
- `favoriteplus_unique_pairs`: unique Train user-item pairs with Favorite, Cart, or Purchase. Pair counting avoids triple-crediting the same funnel progression.
- `cart_count`: Train Add-to-Cart count.
- `purchase_count`: Train Purchase count.
- `weighted_popularity_v1`: `1.0*log1p(View) + 3.0*Favorite + 5.0*Cart + 8.0*Purchase` summed over user-item pairs.

The weighted coefficients are a v1 hypothesis, not optimized values or ground truth.

## Task-matched results

| task | split | K | Recall | NDCG | HitRate | Precision | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| viewplus | validation | 5 | 3.7599% | 7.1303% | 29.1032% | 6.7343% | 591 |
| viewplus | validation | 10 | 7.5419% | 7.9805% | 47.5465% | 6.6159% | 591 |
| viewplus | validation | 20 | 15.4173% | 11.1734% | 69.2047% | 6.7851% | 591 |
| favoriteplus | validation | 5 | 4.5111% | 3.4984% | 11.5632% | 2.4411% | 467 |
| favoriteplus | validation | 10 | 8.9490% | 5.3060% | 21.1991% | 2.3126% | 467 |
| favoriteplus | validation | 20 | 18.5447% | 8.3380% | 36.6167% | 2.3019% | 467 |
| purchase | validation | 5 | 9.3449% | 5.6081% | 12.1387% | 2.4277% | 173 |
| purchase | validation | 10 | 14.7399% | 7.3788% | 18.4971% | 1.8497% | 173 |
| purchase | validation | 20 | 26.1079% | 10.3016% | 30.6358% | 1.5607% | 173 |
| viewplus | test | 5 | 4.7876% | 8.5186% | 32.4059% | 7.8887% | 611 |
| viewplus | test | 10 | 9.3372% | 9.8029% | 50.2455% | 8.0524% | 611 |
| viewplus | test | 20 | 16.8085% | 12.4826% | 70.2128% | 7.2831% | 611 |
| favoriteplus | test | 5 | 5.2866% | 4.1552% | 11.9588% | 2.4330% | 485 |
| favoriteplus | test | 10 | 10.3590% | 6.2574% | 23.7113% | 2.6186% | 485 |
| favoriteplus | test | 20 | 20.0218% | 9.3787% | 39.7938% | 2.5258% | 485 |
| purchase | test | 5 | 7.1069% | 4.6110% | 11.3208% | 2.3585% | 212 |
| purchase | test | 10 | 14.0409% | 6.9187% | 18.8679% | 1.9811% | 212 |
| purchase | test | 20 | 25.3223% | 10.0804% | 32.5472% | 1.8632% | 212 |

## Purchase cross-signal results at K=10

| signal | split | Recall@10 | NDCG@10 | HitRate@10 | Precision@10 | eligible |
| --- | --- | --- | --- | --- | --- | --- |
| total_view_count | validation | 13.6802% | 6.3694% | 16.7630% | 1.6763% | 173 |
| unique_view_users | validation | 13.1021% | 6.1830% | 16.1850% | 1.6185% | 173 |
| log_view_strength | validation | 13.1021% | 6.0960% | 16.1850% | 1.6185% | 173 |
| favoriteplus_unique_pairs | validation | 13.1985% | 6.1197% | 15.0289% | 1.5607% | 173 |
| cart_count | validation | 15.1252% | 7.6464% | 18.4971% | 1.9075% | 173 |
| purchase_count | validation | 14.7399% | 7.3788% | 18.4971% | 1.8497% | 173 |
| weighted_popularity_v1 | validation | 17.0520% | 7.6783% | 20.8092% | 2.0809% | 173 |
| total_view_count | test | 11.9811% | 6.2045% | 17.9245% | 1.8396% | 212 |
| unique_view_users | test | 12.0597% | 6.4852% | 17.9245% | 1.8868% | 212 |
| log_view_strength | test | 12.0597% | 6.4739% | 17.9245% | 1.8868% | 212 |
| favoriteplus_unique_pairs | test | 11.9497% | 6.6075% | 16.9811% | 1.7453% | 212 |
| cart_count | test | 14.4969% | 7.6119% | 21.6981% | 2.2642% | 212 |
| purchase_count | test | 14.0409% | 6.9187% | 18.8679% | 1.9811% | 212 |
| weighted_popularity_v1 | test | 13.6714% | 7.0095% | 19.8113% | 2.0755% | 212 |

## All-task cross-signal results at K=10

| signal | task | split | Recall@10 | NDCG@10 | eligible |
| --- | --- | --- | --- | --- | --- |
| total_view_count | viewplus | validation | 7.5419% | 7.9805% | 591 |
| unique_view_users | viewplus | validation | 7.8349% | 8.2491% | 591 |
| log_view_strength | viewplus | validation | 7.5392% | 8.0902% | 591 |
| favoriteplus_unique_pairs | viewplus | validation | 7.5043% | 7.7989% | 591 |
| cart_count | viewplus | validation | 7.7229% | 7.8413% | 591 |
| purchase_count | viewplus | validation | 7.5226% | 7.7967% | 591 |
| weighted_popularity_v1 | viewplus | validation | 7.8865% | 7.9866% | 591 |
| total_view_count | favoriteplus | validation | 8.4553% | 5.0484% | 467 |
| unique_view_users | favoriteplus | validation | 8.6337% | 5.1254% | 467 |
| log_view_strength | favoriteplus | validation | 8.3482% | 5.0608% | 467 |
| favoriteplus_unique_pairs | favoriteplus | validation | 8.9490% | 5.3060% | 467 |
| cart_count | favoriteplus | validation | 8.9829% | 5.4603% | 467 |
| purchase_count | favoriteplus | validation | 10.2768% | 6.0262% | 467 |
| weighted_popularity_v1 | favoriteplus | validation | 10.3143% | 5.8517% | 467 |
| total_view_count | purchase | validation | 13.6802% | 6.3694% | 173 |
| unique_view_users | purchase | validation | 13.1021% | 6.1830% | 173 |
| log_view_strength | purchase | validation | 13.1021% | 6.0960% | 173 |
| favoriteplus_unique_pairs | purchase | validation | 13.1985% | 6.1197% | 173 |
| cart_count | purchase | validation | 15.1252% | 7.6464% | 173 |
| purchase_count | purchase | validation | 14.7399% | 7.3788% | 173 |
| weighted_popularity_v1 | purchase | validation | 17.0520% | 7.6783% | 173 |
| total_view_count | viewplus | test | 9.3372% | 9.8029% | 611 |
| unique_view_users | viewplus | test | 9.3230% | 9.8830% | 611 |
| log_view_strength | viewplus | test | 9.3665% | 9.8383% | 611 |
| favoriteplus_unique_pairs | viewplus | test | 8.8312% | 9.1499% | 611 |
| cart_count | viewplus | test | 8.6017% | 9.3789% | 611 |
| purchase_count | viewplus | test | 8.0996% | 8.6919% | 611 |
| weighted_popularity_v1 | viewplus | test | 8.7179% | 9.3643% | 611 |
| total_view_count | favoriteplus | test | 10.3558% | 6.0381% | 485 |
| unique_view_users | favoriteplus | test | 10.0003% | 6.2421% | 485 |
| log_view_strength | favoriteplus | test | 10.2698% | 6.2599% | 485 |
| favoriteplus_unique_pairs | favoriteplus | test | 10.3590% | 6.2574% | 485 |
| cart_count | favoriteplus | test | 11.3489% | 7.0490% | 485 |
| purchase_count | favoriteplus | test | 10.0410% | 6.5125% | 485 |
| weighted_popularity_v1 | favoriteplus | test | 11.0288% | 6.8875% | 485 |
| total_view_count | purchase | test | 11.9811% | 6.2045% | 212 |
| unique_view_users | purchase | test | 12.0597% | 6.4852% | 212 |
| log_view_strength | purchase | test | 12.0597% | 6.4739% | 212 |
| favoriteplus_unique_pairs | purchase | test | 11.9497% | 6.6075% | 212 |
| cart_count | purchase | test | 14.4969% | 7.6119% | 212 |
| purchase_count | purchase | test | 14.0409% | 6.9187% | 212 |
| weighted_popularity_v1 | purchase | test | 13.6714% | 7.0095% | 212 |

## Best Test baseline by task

| task | best NDCG signal | NDCG@10 | Recall@10 | best Recall signal | Recall@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| viewplus | unique_view_users | 9.8830% | 9.3230% | log_view_strength | 9.3665% | 9.8383% |
| favoriteplus | cart_count | 7.0490% | 11.3489% | cart_count | 11.3489% | 7.0490% |
| purchase | cart_count | 7.6119% | 14.4969% | cart_count | 14.4969% | 7.6119% |

## Interaction representation statistics (nonzero values)

| representation | pairs | density | mean | std | min | p25 | median | p75 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary_viewplus | 18,591 | 9.2955% | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| view_count | 18,591 | 9.2955% | 1.0699 | 0.2733 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.0000 |
| log_view_count | 18,591 | 9.2955% | 0.7209 | 0.1068 | 0.6931 | 0.6931 | 0.6931 | 0.6931 | 1.6094 |
| binary_favoriteplus | 3,944 | 1.9720% | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_purchase | 731 | 0.3655% | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| weighted_implicit_v1 | 18,591 | 9.2955% | 1.7991 | 2.6221 | 0.6931 | 0.6931 | 0.6931 | 1.0986 | 17.6094 |

## Signal richness versus future Purchase

| signal | pairs | users | items | density | Test Purchase Recall@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| view | 18,591 | 971 | 200 | 9.2955% | 11.9811% | 6.2045% |
| favoriteplus | 3,944 | 873 | 200 | 1.9720% | 11.9497% | 6.6075% |
| cart | 1,164 | 603 | 169 | 0.5820% | 14.4969% | 7.6119% |
| purchase | 731 | 431 | 162 | 0.3655% | 14.0409% | 6.9187% |
| weighted | 18,591 | 971 | 200 | 9.2955% | 13.6714% | 7.0095% |

## Heavy User and Raw-vs-Log audit

- Top 1% observed-activity Users contribute 3.4942% of raw View strength and 3.3271% after per-pair log1p compression.
- Maximum per-User strength changes from 76.0 raw to 47.7229 log strength.
- Activity deciles are computed only from observed Train Views; the hidden synthetic activity tier is not loaded.

## Event overlap audit

- View pairs: 18,591; Favorite+ pairs: 3,944; Purchase pairs: 731.
- Favorite+ outside View: 0; Purchase outside View: 0; Purchase outside Favorite+: 0.
- Maximum Favorite/Cart/Purchase count per pair: 1/1/1.

## Validation/Test stability

The largest absolute NDCG@10 shift is `total_view_count` on `viewplus`: 7.9805% Validation to 9.8029% Test (difference +0.0182). No Test result was used to tune a score or weight.
The largest absolute Recall@10 shift is `weighted_popularity_v1` on `purchase`: 17.0520% Validation to 13.6714% Test (difference -0.0338).

## Interpretation for the next phase

- Popularity is a non-personalized control: all Users receive the same global order before task-specific seen exclusion.
- View is weak but rich; Purchase is strong but sparse. Log View reduces repeat-view dominance while retaining broad coverage.
- Weighted implicit v1 should remain a candidate, not the assumed answer. Matrix Factorization v1 should compare binary View+, log View count, Favorite+, Purchase-only, and weighted implicit under the same split and evaluation policy.
- Recommended MF v1 sequence: establish binary View+ as the simplest dense control, compare log View as the count-aware primary candidate, then add Favorite+ and weighted implicit as challengers. Purchase-only is too sparse to be the sole first representation.
- Do not make weighted implicit the only initial MF input: Cart popularity beat weighted v1 on Test Purchase, and weighted Purchase Recall@10 moved more between Validation and Test. Keep the weights configurable and compare representations before tuning them.
- Hidden User preferences, Product ground-truth attributes, archetypes, preference_match, and Validation/Test Events are absent from scoring and representations.
