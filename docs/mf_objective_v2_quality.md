# MF Objective v2 Quality Report

## Decision

Keep **BCE** as the main Matrix Factorization objective. Under the frozen best setup, confidence-weighted BPR did not support the ranking hypothesis: at Test Purchase@10 it reached Recall `12.2170%` and NDCG `6.3001%`, below BCE (`15.1258%`, `8.3149%`) and Cart Popularity (`14.4969%`, `7.6119%`). No parameter was changed after Test.

## Frozen comparison contract

Both objectives use Existing Weighted strength `log1p(view_count) + 3*favorite + 5*cart + 8*purchase`, positive confidence `1 + log1p(strength)`, the same 18,591 Train positive pairs, and the same 74,364 pre-generated comparison triples (four per positive). Each comparison prefers a Train Exposed Non-conversion and uses seeded Unknown only as backfill. Exposed Non-conversion is neither dislike nor a true negative; BPR only assumes the observed positive should rank above that comparison item.

The identical score is `score(u,i) = user_embedding[u] dot item_embedding[i] + item_bias[i]`. The remaining fixed settings are latent dimension 8, Adam, learning rate `1e-3`, weight decay `1e-4`, batch size 1024, seed 42, at most 100 epochs, patience 5, Purchase candidates, seen exclusion, and Validation Purchase NDCG@10 as the sole selection metric. Hidden simulator dimensions and future events are not used.

## What BCE and BPR learn

BCE learns whether each User-Item pair looks positive as an individual binary classification example. Existing positive confidence weights its positive loss. BPR directly learns to place a positive item above a comparison item, using the stable loss `mean(confidence_ui * -logsigmoid(score_ui - score_uj))`. Item bias participates in both sides of the BPR difference, so the only intended change is objective.

## Validation and training behavior

| Objective | Best epoch | Epochs run | Loss at best | Validation Purchase Recall@10 | Validation Purchase NDCG@10 |
|---|---:|---:|---:|---:|---:|
| BCE | 23 | 28 | 0.688596 | 20.8092% | 9.9855% |
| BPR | 7 | 12 | 1.183920 | 19.7495% | 11.6505% |

BCE and BPR losses have different meanings and scales and are not compared as quality scores. BPR selected a higher Validation NDCG but did not preserve it on Test.

## Final Test

| Model | Purchase Recall@10 | Purchase NDCG@10 | HitRate@10 | Precision@10 |
|---|---:|---:|---:|---:|
| Cart Popularity | 14.4969% | 7.6119% | 21.6981% | 2.2642% |
| BCE dim8 | **15.1258%** | **8.3149%** | **21.2264%** | **2.1698%** |
| BPR dim8 | 12.2170% | 6.3001% | 17.4528% | 1.7925% |

At K=10, BPR minus BCE is Purchase Recall `-2.9088 pp` and NDCG `-2.0148 pp`; View+ Recall `-0.4749 pp` and NDCG `-0.8620 pp`; Favorite+ Recall `-1.1913 pp` and NDCG `-0.6152 pp`. Full K=5/10/20 results for all three tasks are in the per-objective metrics artifacts.

Validation-to-Test Purchase changes were BCE Recall `-5.6835 pp`, NDCG `-1.6706 pp`, versus BPR Recall `-7.5325 pp`, NDCG `-5.3504 pp`. The earlier warning pattern therefore recurred: BPR looked strongest on Validation NDCG but generalized less stably.

## Accuracy and personalization

| Objective | Purchase Recall@10 | Purchase NDCG@10 | User Top10 overlap | Cart Top10 overlap |
|---|---:|---:|---:|---:|
| BCE | 15.1258% | 8.3149% | 59.7975% | 52.7358% |
| BPR | 12.2170% | 6.3001% | 49.6517% | 51.7453% |

Both produced 206 unique Top-10 lists among 212 Purchase-eligible users. BPR was more personalized by pairwise overlap, but accuracy fell. Its recommended items had a slightly lower mean Train Cart count (14.9146 versus 15.2769), so this is `individualization up / accuracy down`, not a better frontier.

## Item bias and score decomposition

| Objective | Bias mean / std / min / median / max | Cart Pearson / Spearman | Purchase Pearson / Spearman |
|---|---|---|---|
| BCE | -0.5661 / 0.1966 / -0.9976 / -0.5859 / 0.0124 | 0.6157 / 0.5269 | 0.6066 / 0.5185 |
| BPR | -0.0173 / 0.1667 / -0.3447 / -0.0526 / 0.3393 | 0.5954 / 0.5245 | 0.5900 / 0.5107 |

BPR retained a clear popularity component. It did not become personal-component dominated: personal variance was `0.003419`, item-bias variance `0.028597`, and their ratio `8.3637`; BCE was `0.025399`, `0.039813`, and `1.5675`. BPR's personal component was instead much weaker, despite its lower recommendation overlap.

## Embeddings and training margins

All embedding, score, and bias values are finite; no NaN, Inf, or norm explosion was found. BCE/BPR mean User norms were `0.3742/0.3036`, mean Item norms `0.8672/0.4927`, and score standard deviations `0.3030/0.1788`.

On the identical 74,364 Train comparisons, BCE margin mean/std/median/range was `0.1915 / 0.4619 / 0.1723 / [-1.7410, 2.2301]`, with `65.5142%` positive. BPR was `0.0996 / 0.2568 / 0.1027 / [-0.9557, 1.0036]`, with `64.8634%` positive. Contrary to the mechanistic hypothesis, BPR did not create a stronger observed pairwise margin than BCE.

## History, popularity, and coverage

Test Purchase NDCG@10 by Low/Medium/High Train history was BCE `8.2615% / 6.8919% / 9.7905%` and BPR `7.8676% / 4.3545% / 6.7003%`. BPR was below BCE in every group, with the largest instability in Medium/High-history users rather than only cold users.

For Low-popularity relevant items both models had zero Recall; for Medium, BCE was `1.2821%` and BPR `0%`; for High, BCE was `26.6796%` and BPR `22.1576%`. BPR did not uncover a niche-item advantage.

Coverage is objective-invariant because the same representation, candidates, seen policy, and fallback are shared: 971/1,000 users and all 200 items have a Train positive; all 212 Test Purchase users are evaluated, 205 use learned rankings and the same 7 use Train-only Cart Popularity fallback (3.3019%).

## Runtime and freeze decision

BCE's reused latent-dimension run took 67.6472 seconds over 28 epochs (2.4160 seconds/epoch), best epoch 23, checkpoint 41,587 bytes. The recorded BPR run took 31.9229 seconds over 12 epochs (2.6602 seconds/epoch), best epoch 7, checkpoint 41,898 bytes. BPR's shorter total time came from earlier stopping, not a cheaper epoch.

The main hypothesis is not supported. Freeze BCE + Existing Weighted + Exposed sampling + Item Bias + latent dimension 8 as the current main MF. The most useful next experiment is **D. MF + Cart Popularity hybrid**, because BCE now beats popularity at K=10 while still leaving complementary global-popularity signal. After that, consider E Content-Based and F Hybrid Recommendation. Confidence design remains a narrower alternative; do not reopen latent dimension from this Test.
