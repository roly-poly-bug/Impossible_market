# MF Cart Signal v1 Quality Report

## Scope and fixed conditions

This experiment changes only the Train positive pool and positive confidence. All models use BCE Matrix Factorization with `user_embedding · item_embedding + item_bias`, latent dimension 16, Train-only Exposed Non-conversion with Random Unknown backfill at ratio 4, Adam, learning rate 0.001, weight decay 0.0001, batch size 1,024, seed 42, maximum 100 epochs, patience 5, Purchase candidates, seen exclusion, and Validation Purchase NDCG@10 checkpoint selection. A zero training contrast remains an observed non-conversion or Unknown sample, not dislike or a true negative.

The Cart-centered coefficients were fixed before Test as `0.5·log1p(view_count) + 2·favorite + 6·cart + 10·purchase`. No weight or hyperparameter was changed after Test.

## Signal definitions

- **Existing Weighted:** all View+ pairs; confidence `1 + log1p(log1p(view_count) + 3F + 5C + 8P)`. This is the frozen Item Bias control.
- **Cart+:** Cart or Purchase pairs only; confidence `1 + log1p(5C + 8P)`. View and Favorite do not enter confidence.
- **Favorite+Cart+:** Favorite, Cart, or Purchase pairs; confidence `1 + log1p(3F + 5C + 8P)`. View does not enter confidence.
- **Cart-centered Weighted:** all View+ pairs; confidence `1 + log1p(0.5log1p(view_count) + 2F + 6C + 10P)`.

For Cart+ and Favorite+Cart+, a Train View/Favorite opportunity without the target action is eligible for the same Exposed Non-conversion sampling mechanism. This is a task-relative contrast boundary, not a negative-truth claim. Unknown backfill is still drawn from Train-only unexposed pairs.

## Coverage and Future Purchase alignment

| signal | positive pairs | positive Users | items | density | zero-positive Users | Test fallback | exact future-item continuity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing Weighted | 18,591 | 971 | 200 | 9.2955% | 29 | 7/212 (3.30%) | 25.47% |
| Cart+ | 1,572 | 674 | 170 | 0.7860% | 326 | 50/212 (23.58%) | 6.13% |
| Favorite+Cart+ | 3,944 | 873 | 200 | 1.9720% | 127 | 20/212 (9.43%) | 11.79% |
| Cart-centered Weighted | 18,591 | 971 | 200 | 9.2955% | 29 | 7/212 (3.30%) | 25.47% |

Narrower intent signals sharply reduce coverage and do not show stronger exact User–Item continuity into the later Test window. Cart+ loses 91.5% of the positive pairs and leaves almost one quarter of Test Purchase Users on fallback.

## Training, Validation, and Test

| signal | epochs | best | best loss | Val Recall@10 | Val NDCG@10 | Test Recall@10 | Test NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing Weighted | 14 | 9 | 0.7185 | 19.4605% | 10.2934% | **13.9701%** | **7.3146%** |
| Cart+ | 17 | 12 | 0.9530 | 12.1387% | 5.9430% | 9.2925% | 5.6766% |
| Favorite+Cart+ | 16 | 11 | 0.8937 | 13.4875% | 6.3865% | 9.9214% | 5.5755% |
| Cart-centered Weighted | 12 | 7 | 0.6945 | 16.0886% | 7.4265% | 12.1777% | 6.4100% |
| Cart Popularity | — | — | — | — | — | 14.4969% | 7.6119% |

No NaN, Inf, or divergence was observed. Cart-centered is the strongest new signal, but it remains below Existing Weighted by 1.7925 Recall points and 0.9047 NDCG points. None exceeds Cart Popularity.

Secondary Test Recall/NDCG@10 is 6.0607%/6.3471% View+ and 6.6766%/4.3221% Favorite+ for Cart+; 5.5781%/5.9677% and 7.1766%/4.2304% for Favorite+Cart+; and 8.9758%/9.2298% and 9.7524%/5.8869% for Cart-centered. Existing Weighted remains 8.0509%/8.4423% and 9.3470%/5.9924%.

## Learned group and fallback

| signal | learned Users | learned Recall/NDCG@10 | fallback Users | fallback Recall/NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Existing Weighted | 205 | 14.4472% / 7.5644% | 7 | 0 / 0 |
| Cart+ | 162 | 9.8971% / 5.9724% | 50 | 7.3333% / 4.7183% |
| Favorite+Cart+ | 192 | 10.1736% / 5.7248% | 20 | 7.5000% / 4.1422% |
| Cart-centered Weighted | 205 | 12.5935% / 6.6289% | 7 | 0 / 0 |

Cart+ is not being rescued into a strong aggregate by fallback: its learned group also trails Existing Weighted substantially. The Cart popularity fallback helps its aggregate modestly, but does not explain away the weak learned embeddings.

## Personalization and popularity overlap

| signal | unique Top10 | User overlap | Cart overlap | recommended Cart mean |
| --- | ---: | ---: | ---: | ---: |
| Existing Weighted | 206 | 50.27% | 45.42% | 14.45 |
| Cart+ | 163 | 24.05% | 39.48% | 12.69 |
| Favorite+Cart+ | 193 | 33.12% | 32.22% | 12.90 |
| Cart-centered Weighted | 206 | 77.24% | 56.65% | 15.64 |

Cart+ does not merely reproduce Cart Popularity; it is more diverse between learned Users, while its 50 fallback Users share the Cart ranking policy. Cart-centered moves strongly toward popularity: accuracy is below Existing Weighted despite much higher User overlap and Cart overlap, so the extra popularity concentration did not improve Future Purchase ranking.

## Item Bias and model-scale diagnostics

| signal | bias mean/std | Cart Pearson/Spearman | Purchase Pearson/Spearman | User norm | Item norm | score mean/std |
| --- | --- | --- | --- | ---: | ---: | --- |
| Existing Weighted | -0.3728 / 0.0977 | .6731 / .5719 | .6706 / .5696 | .3386 | .6969 | -.4008 / .1279 |
| Cart+ | -0.0233 / 0.0424 | .7708 / .8049 | .7625 / .7962 | .3095 | .4171 | -.0233 / .0552 |
| Favorite+Cart+ | -0.0709 / 0.0560 | .7256 / .7249 | .6607 / .6777 | .3124 | .4703 | -.0711 / .0696 |
| Cart-centered Weighted | -0.3409 / 0.0920 | .6859 / .5739 | .6735 / .5586 | .2517 | .5070 | -.3904 / .1179 |

Cart+ bias is highly rank-correlated with Cart/Purchase counts, but its magnitude is small and its total ranking quality is weak. Cart-centered bias variance is about 4.50 times personal-component variance, consistent with its high cross-User overlap. All scales remain finite and bounded.

## Stability, hypothesis, and recommendation

Validation-to-Test Recall/NDCG changes are -5.4904/-2.9788 points for Existing Weighted, -2.8463/-0.2663 for Cart+, -3.5661/-0.8110 for Favorite+Cart+, and -3.9110/-1.0165 for Cart-centered. Narrow signals have smaller gaps mainly because their Validation level is already lower; this is not evidence of superior absolute generalization.

The main hypothesis is **not supported in this fixed experiment**. Moving the positive definition toward immediate Cart intent loses too much User–Item coverage and temporal continuity. Retaining broad View coverage but increasing Cart/Purchase weights also over-concentrates rankings toward popularity without improving the primary metrics.

Keep **Existing Weighted** as the main MF signal. The next isolated comparison should be latent dimension capacity or BCE versus BPR while retaining the broad signal. Later candidates are confidence redesign, bias regularization, sampling mix, and a simple MF-plus-Cart-popularity hybrid.

## Test protocol note

The first pipeline invocation completed the fixed training and entered the batched Test evaluator, then failed in post-evaluation fallback assembly before metrics were inspected or artifacts were written. The same deterministic configuration was rerun after correcting only that assembly path. No signal coefficient, model setting, sampling rule, checkpoint rule, or hyperparameter was changed based on Test. The operational retry is recorded in the root manifest rather than hidden.
