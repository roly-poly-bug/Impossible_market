# MF Latent Dimension v1 Quality Report

## Meaning and fixed conditions

Latent dimension is the number of hidden coordinates used to represent each User and Item. Dimension 8 is a strongly compressed representation, 16 is the previous baseline, 32 can express more complex variation, and 64 has still more capacity with greater overfitting and compute risk. The simulator's hidden nine Product/User attributes were not read or used; 8/16/32/64 are independent capacity choices.

Only latent dimension changes. Existing Weighted confidence, all 18,591 positive pairs, all 74,364 shared Exposed/Unknown samples, Item-Bias BCE MF, Adam, learning rate 0.001, weight decay 0.0001, batch size 1,024, seed 42, maximum 100 epochs, patience 5, Validation Purchase NDCG@10 selection, candidates, seen exclusion, and Cart fallback are fixed. Dimension 16 reuses the frozen MF Bias v1 checkpoint; 8/32/64 are newly trained. All four fixed candidates enter one guarded final Test batch. No setting or dimension was added after Test.

## Capacity and runtime

| dim | total params | User emb | Item emb | bias | runtime | sec/epoch | checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 9,800 | 8,000 | 1,600 | 200 | 67.65s | 2.42s | 41,587 B |
| 16 | 19,400 | 16,000 | 3,200 | 200 | reused | reused | 80,032 B |
| 32 | 38,600 | 32,000 | 6,400 | 200 | 52.88s | 2.40s | 156,796 B |
| 64 | 77,000 | 64,000 | 12,800 | 200 | 88.81s | 2.61s | 310,396 B |

Parameter count and checkpoint size scale almost linearly with dimension. Runtime is also affected by the number of early-stopping epochs, so total runtime is not monotonic.

## Training, Validation, and Test

| dim | epochs | best | best-epoch loss | Val Recall@10 | Val NDCG@10 | Test Recall@10 | Test NDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 28 | 23 | 0.6886 | 20.8092% | 9.9855% | **15.1258%** | **8.3149%** |
| 16 | 14 | 9 | 0.7185 | 19.4605% | 10.2934% | 13.9701% | 7.3146% |
| 32 | 22 | 17 | 0.6680 | 18.1118% | 9.0501% | 14.5204% | 6.9761% |
| 64 | 34 | 29 | **0.5463** | **22.2543%** | **11.2144%** | 13.9701% | 8.1425% |
| Cart Popularity | — | — | — | — | — | 14.4969% | 7.6119% |

Dimension 8 beats Cart Popularity by 0.6289 Recall points and 0.7030 NDCG points. It improves dimension 16 by 1.1557 and 1.0002 points. Dimension 32 narrowly beats Cart Recall but trails its NDCG. Dimension 64 has the strongest Validation and lowest loss but loses 8.2842 Recall points and 3.0718 NDCG points from Validation to Test, a clear capacity/temporal-overfit warning.

Secondary Test Recall/NDCG@10 is 8.7801%/9.5438% View+ and 9.7621%/6.2693% Favorite+ for dim8; 8.0509%/8.4423% and 9.3470%/5.9924% for dim16; 9.0556%/9.0627% and 9.6689%/5.7306% for dim32; and 7.8885%/8.2838% and 8.1453%/4.9609% for dim64.

## Embedding and score diagnostics

| dim | User norm mean | /sqrt(dim) | Item norm mean | /sqrt(dim) | score mean/std | score range |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 8 | .3742 | .1323 | .8672 | .3066 | -.6691 / .3030 | -1.9964 to .6235 |
| 16 | .3386 | .0847 | .6969 | .1742 | -.4008 / .1279 | -1.0719 to .2078 |
| 32 | .5527 | .0977 | 1.2400 | .2192 | -.6374 / .3101 | -2.0534 to .8053 |
| 64 | .9779 | .1222 | 2.2588 | .2824 | -.7032 / .4642 | -3.0541 to 2.2713 |

Norms and score range grow substantially at 64 dimensions. Dimension-normalized norms remain bounded, so this is partly expected capacity scaling, but the wider score distribution accompanies the Validation/Test instability.

## Item Bias and score decomposition

| dim | bias mean/std | Cart Pearson/Spearman | Purchase Pearson/Spearman | personal var | bias var | bias/personal |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 8 | -.5661 / .1966 | .6157 / .5269 | .6066 / .5185 | .02540 | .03981 | 1.5675 |
| 16 | -.3728 / .0977 | .6731 / .5719 | .6706 / .5696 | .00479 | .01010 | 2.1087 |
| 32 | -.4680 / .1556 | .6151 / .5384 | .6085 / .5204 | .04153 | .02509 | .6041 |
| 64 | -.5488 / .1628 | .5787 / .5096 | .5695 / .4962 | .16069 | .02689 | .1673 |

Personal variance rises sharply with capacity and Item Bias becomes relatively less dominant. This confirms that high dimensions express more User-specific structure, but the 64-dimensional structure does not generalize best to the later Test period.

## Accuracy and personalization

| dim | Recall@10 | NDCG@10 | unique lists | User overlap | Cart overlap | recommended Cart mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | **15.1258%** | **8.3149%** | 206 | 59.80% | 52.74% | 15.28 |
| 16 | 13.9701% | 7.3146% | 206 | 50.27% | 45.42% | 14.45 |
| 32 | 14.5204% | 6.9761% | 206 | 50.16% | 49.34% | 14.90 |
| 64 | 13.9701% | 8.1425% | 206 | **21.11%** | **31.93%** | 12.18 |

Dimension 64 is by far the most personalized and least popularity-aligned, but not the most accurate. Dimension 8 wins while being more popularity-aligned than 16. The gain therefore comes from a useful compressed mixture of personalization and global propensity, not simply from more capacity.

## User-history and Item-popularity groups

Dimension 8 Test Recall/NDCG@10 is 14.57%/8.26% for low-history Users, 12.32%/6.89% for medium, and 18.47%/9.79% for high. Dimension 64 shifts strength toward medium/high-history Users (16.43%/8.97% and 16.83%/9.30%) but performs only 8.57%/6.13% for low-history Users. This supports higher-capacity data-hunger.

All models rely mainly on high-Cart-popularity relevant items. Dimension 64 is the only model with non-zero low-popularity Recall (1.96%) and has much better medium-popularity Recall (5.77%), showing some long-tail capacity. It pays for that with lower high-popularity Recall than dim8 (22.03% versus 26.68%) and lower aggregate Recall.

## Hypothesis, recommendation, and next phase

The hypothesis that 16 dimensions are insufficient is **not supported in the expected direction**: increasing capacity does not improve both primary metrics. Instead, reducing to 8 dimensions improves both and crosses Cart Popularity. The frozen data are small and popularity-heavy enough that compression acts as useful regularization. Dimension 64 learns a strong, personalized Validation fit but shows temporal overfitting and poor low-history robustness.

Use **latent_dim 8** as the next main MF dimension. It has the best primary Test metrics, the smallest model/checkpoint, competitive per-epoch cost, good low- and high-history performance, and remains personalized with 206 unique lists. Next isolate BCE versus BPR using this setup. Later candidates are confidence design, Item Bias regularization, Exposed/Unknown ratio, an MF-plus-Cart hybrid, and Content-Based/Hybrid recommendation.
