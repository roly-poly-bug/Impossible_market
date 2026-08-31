# MF Signal Representation v1 Quality Report

## Scope and fixed conditions

This experiment changes only the Train interaction representation used by the fixed Pointwise BCE Matrix Factorization. The frozen Binary View control is reused without another Test evaluation.

- Score: bias-free `user_embedding · item_embedding`; latent dimension 16
- Adam: learning rate 0.001, weight decay 0.0001
- Four distinct Random Unknown samples per positive; seed 42
- Batch size 1,024; maximum 100 epochs; patience 5
- Selection: Validation Purchase NDCG@10 only
- Test: one final pass after each new best checkpoint was fixed
- Candidate and task-specific seen policy: Recommendation Dataset v1 unchanged
- Cold users: Train-only Cart Popularity fallback

Sampled Unknown is a training non-positive, not a true dislike. Observed Non-conversion is excluded. Hidden preferences, Product attributes, future interactions, Validation, and Test events are not training features.

## Representation definitions

| representation | positive definition | positive confidence | Unknown task |
| --- | --- | --- | --- |
| Binary View | Train View+ | 1 | View+ |
| Log View | Train View+ | `1 + log1p(view_count)`; alpha=1 | View+ |
| Favorite+ | Favorite, Cart, or Purchase | 1 | Favorite+ |
| Weighted | Train View+ | `1 + log1p(log1p(view_count) + 3·favorite + 5·cart + 8·purchase)` | View+ |
| Purchase-only | Train Purchase | 1 | Purchase |

Targets remain binary: positive=1 and sampled Unknown=0. Confidence changes loss contribution, not the target. No mean normalization is applied because the transforms keep weights modest.

## Coverage

| representation | positive pairs | users | items | density | zero-positive users | zero-positive items |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary View | 18,591 | 971 | 200 | 9.2955% | 29 | 0 |
| Log View | 18,591 | 971 | 200 | 9.2955% | 29 | 0 |
| Favorite+ | 3,944 | 873 | 200 | 1.9720% | 127 | 0 |
| Weighted | 18,591 | 971 | 200 | 9.2955% | 29 | 0 |
| Purchase-only | 731 | 431 | 162 | 0.3655% | 569 | 38 |

## Confidence distribution

| representation | mean | std | min | p25 | median | p75 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Log View | 1.7209 | 0.1068 | 1.6931 | 1.6931 | 1.6931 | 1.6931 | 2.6094 |
| Weighted | 1.8113 | 0.5608 | 1.5266 | 1.5266 | 1.5266 | 1.7413 | 3.9237 |

Neither distribution has an extreme maximum. Loss values, embeddings, and scores remained finite.

## Training and Validation

| representation | epochs | best epoch | first loss | best-epoch loss | Val Purchase Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary View | 11 | 6 | 0.6929 | 0.5953 | 16.6667% | 6.9228% |
| Log View | 12 | 7 | 0.7929 | 0.7541 | 16.3776% | 7.2454% |
| Favorite+ | 29 | 24 | 0.6934 | 0.6181 | 18.2081% | 8.6739% |
| Weighted | 12 | 7 | 0.8055 | 0.7732 | 17.1484% | 8.0559% |
| Purchase-only | 14 | 9 | 0.6932 | 0.6866 | 10.7900% | 4.8169% |

All checkpoints use Validation Purchase NDCG@10. Test is absent from training and early stopping.

## Test results at K=10

| representation | Purchase Recall | Purchase NDCG | View+ Recall | View+ NDCG | Favorite+ Recall | Favorite+ NDCG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary View | 11.0377% | 5.3432% | 8.4946% | 8.8146% | 9.5143% | 5.3748% |
| Log View | 10.7469% | 5.4777% | 8.3132% | 8.6738% | 8.3787% | 5.2987% |
| Favorite+ | 9.9057% | 5.9654% | 7.7042% | 8.0712% | 7.2545% | 4.8622% |
| Weighted | 11.0613% | 5.7687% | 8.4657% | 8.4802% | 9.4165% | 5.6934% |
| Purchase-only | 11.7846% | 6.0376% | 7.5377% | 7.7489% | 9.9638% | 6.0346% |
| Cart Popularity | 14.4969% | 7.6119% | — | — | — | — |

No representation beats Cart Popularity on Purchase Recall@10 or NDCG@10.

## Hypothesis assessment

**Log View:** relative to Binary View, Purchase Recall@10 changes by -0.2909 percentage points and NDCG@10 by +0.1345 points. Repeated-interest confidence slightly improves ordering quality but not retrieval coverage, so the hypothesis is only weakly supported.

**Favorite+:** it produces the strongest Validation result but loses 8.3024 Recall points and 2.7086 NDCG points on Test. Stronger intent does not offset lower coverage and temporal instability here.

**Weighted:** it is the strongest broadly covered representation. It retains 971 users and 200 items and improves Binary View Purchase NDCG@10 by 0.4255 points while Recall stays nearly flat (+0.0236 points). Fixed 1/3/5/8 weights were not adjusted.

**Purchase-only:** it has the highest aggregate learned-experiment Test result, but 569/1,000 users have no Train Purchase. Among 212 eligible Test Purchase users, 103 use Cart fallback. Fallback users achieve Recall@10 12.9450% and NDCG@10 7.1277%; 109 learned users achieve 10.6881% and 5.0076%. Its aggregate advantage is substantially driven by fallback, so it remains a reference result.

## Personalization and popularity

| representation | unique Top10 / 212 | user-to-user overlap | Cart Top10 overlap | mean recommended Cart count |
| --- | ---: | ---: | ---: | ---: |
| Binary View | 206 | 76.5533% | 29.3396% | 12.5840 |
| Log View | 206 | 60.0858% | 33.9151% | 13.0250 |
| Favorite+ | 193 | 31.0181% | 31.4151% | 11.8887 |
| Weighted | 206 | 53.2907% | 37.4528% | 13.4146 |
| Purchase-only | 110 | 28.2625% | 51.6509% | 12.0009 |

Weighted is more personalized than Binary View by pairwise overlap, while its higher Cart overlap shows that deep-engagement confidence also pulls toward popular/cart-heavy items. Purchase-only has only 110 unique lists because fallback is used extensively.

## Validation/Test stability

- Binary View: Recall -5.6289 points; NDCG -1.5796 points.
- Log View: Recall -5.6308 points; NDCG -1.7676 points.
- Favorite+: Recall -8.3024 points; NDCG -2.7086 points.
- Weighted: Recall -6.0870 points; NDCG -2.2872 points.
- Purchase-only: Recall +0.9946 points; NDCG +1.2207 points, confounded by fallback.

## Recommendation and next phase

Use **Weighted Implicit v1** as the next main MF representation. It offers broad View coverage, all-item coverage, stronger Purchase NDCG than Binary/Log/Favorite+, more personalization than Binary View, and simpler interpretation than a Purchase-only result dominated by fallback.

The next controlled experiment should compare Random Unknown with Exposed Non-conversion sampling while keeping Weighted MF fixed. Mixed negative sampling can follow. Confidence tuning, latent-dimension comparison, bias terms, and Cart/Favorite-centered objectives should remain separate later experiments. No parameter was changed after Test.
