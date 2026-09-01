# MF Cart Hybrid v1 Quality Report

## Decision

Do not promote the post-hoc Cart Hybrid as a new recommender. Validation selected
`alpha=1.00`, so the best hybrid is exactly the frozen BCE MF and adds no Cart
Popularity contribution. Keep BCE + Existing Weighted + Exposed sampling + Item
Bias + latent dimension 8 as the main MF candidate.

## Purpose and fixed boundary

This experiment tests whether explicit Train Cart Popularity can complement the
personalized best MF score without retraining. The source checkpoint remains
byte-identical (`SHA-256 241309a4...113191`); no optimizer, backward pass,
fine-tuning, new representation, or learned fusion is involved.

```text
MF(u,i) = user_embedding[u] dot item_embedding[i] + item_bias[i]
Cart(i) = Train-period add_to_cart count
Hybrid(u,i) = alpha * z_mf(u,i) + (1-alpha) * z_cart(i)
```

MF is z-scored per User over the task candidate set. Cart is z-scored over the
same task candidate set. Both use population standard deviation; a zero or
non-finite standard deviation safely produces zeros. Candidates, deterministic
Product-ID tie-breaking, seen exclusion, and Train-only Cart fallback are the
same as the frozen evaluator.

## Pre-fixed alpha selection

Only `0.00, 0.25, 0.50, 0.75, 1.00` were evaluated. No `0.90`, fine grid, or
normalization alternative was added. Validation Purchase NDCG@10 selects alpha;
ties use Recall@10, then the higher alpha.

| Alpha | Validation Recall@10 | Validation NDCG@10 | HitRate@10 | Precision@10 |
|---:|---:|---:|---:|---:|
| 0.00 | 15.1252% | 7.6464% | 18.4971% | 1.9075% |
| 0.25 | 17.5337% | 8.4729% | 20.8092% | 2.1387% |
| 0.50 | 19.0751% | 9.1800% | 23.1214% | 2.3121% |
| 0.75 | 17.4374% | 8.6622% | 21.3873% | 2.1387% |
| **1.00** | **20.8092%** | **9.9855%** | **24.8555%** | **2.6012%** |

The pattern is not monotonic in all metrics, but every explicit Cart mixture has
lower selection NDCG than the MF endpoint. Test was run once only after selecting
`alpha=1.00`.

## Final Test and secondary tasks

| Model | Purchase Recall@10 | Purchase NDCG@10 | HitRate@10 | Precision@10 |
|---|---:|---:|---:|---:|
| Cart Popularity | 14.4969% | 7.6119% | 21.6981% | 2.2642% |
| Best MF | **15.1258%** | **8.3149%** | 21.2264% | 2.1698% |
| Best Hybrid (`alpha=1`) | **15.1258%** | **8.3149%** | 21.2264% | 2.1698% |

Best Hybrid minus Best MF is exactly zero for every K=5/10/20 Purchase metric.
Against Cart Popularity at K=10 it gains `+0.6289 pp` Recall and `+0.7030 pp`
NDCG. Because alpha is one, Test View+ Recall/NDCG@10 remains
`8.7801%/9.5438%`, and Favorite+ remains `9.7621%/6.2693%`; interest tasks are
not damaged.

## Item Bias overlap and score contribution

Frozen Item Bias versus Train Cart Count has Pearson `0.6157` and Spearman
`0.5269`. This confirms substantial, though incomplete, shared popularity
information. At selected alpha, the normalized MF contribution has standard
deviation `1.0` and variance `1.0`; Cart contribution has standard deviation and
variance `0.0`; final score equals MF z-score. Thus the selected result is not a
genuine two-source hybrid.

## Personalization

| Model | Unique Top10 | User overlap | Cart Top10 overlap | Mean Train Cart Count |
|---|---:|---:|---:|---:|
| Cart Popularity | 14 | 97.2548% | 100.0000% | 17.5514 |
| Best MF | 206 | 59.7975% | 52.7358% | 15.2769 |
| Best Hybrid | 206 | 59.7975% | 52.7358% | 15.2769 |

Personalization is fully maintained only because Cart received zero selected
weight. The experiment did not achieve the stronger target of accuracy gain
while maintaining personalization.

## History, item popularity, and fallback

Best Hybrid is identical to Best MF. Low/Medium/High-history Purchase NDCG@10 is
`8.2615% / 6.8919% / 9.7905%`. Popularity addition therefore provides no
special cold-history benefit. Relevant-item Recall@10 is `0% / 1.2821% /
26.6796%` for Low/Medium/High popularity; long-tail behavior is unchanged.

All 212 Test Purchase users are evaluated. The same 205 learned users use MF and
the same seven zero-Train-positive users use Train-only Cart fallback. Candidate
sets and `exclude_seen=true` are unchanged.

## Runtime, hypothesis, and next phase

Final best-alpha ranking evaluation took `2.3172 s`; measured additional peak
Python memory was `653,268 bytes` (about `0.62 MiB`). No training runtime or new
checkpoint is required.

The main hypothesis is not supported in this fixed z-score/alpha experiment.
The leading explanation is that the frozen MF Item Bias already captures much
of Train Cart popularity, while explicit Cart mixing suppresses useful
personalized ordering. Freeze the existing MF, not a separate Hybrid v1.

Next, move to **Content-Based Recommendation** as an independently informative
signal, followed by a later Hybrid Recommendation experiment combining
collaborative and content signals. Do not tune this Test by adding alpha points,
normalizations, per-user weights, or a learned ranker.
