# MF Bias v1 Quality Report

## Fixed conditions and Bias structures

Weighted Implicit positive confidence, Exposed Non-conversion sampling with Unknown backfill, all 74,364 samples, BCE, latent dimension 16, Adam, learning rate 0.001, weight decay 0.0001, batch size 1,024, seed 42, patience 5, Validation Purchase NDCG@10 selection, candidates, seen exclusion, and Cart fallback are fixed.

- No Bias: `user_embedding · item_embedding` (frozen control).
- Item Bias: personal dot product plus `item_bias[item]`.
- User+Item Bias: personal dot product plus item and user bias.
- Biases start at zero and receive the same Adam weight decay as embeddings.
- Global bias is omitted because it cannot change a same-User Top-K order.

User bias is likewise constant across all candidate items for one User, so it has no direct ranking effect. It can still alter training calibration and indirectly change learned embeddings and item bias.

## Training, Validation, and Test

| model | epochs | best | Val Recall@10 | Val NDCG@10 | Test Recall@10 | Test NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No Bias | 29 | 24 | 17.4374% | 9.7957% | 12.2406% | 6.5507% |
| Item Bias | 14 | 9 | 19.4605% | 10.2934% | 13.9701% | 7.3146% |
| User+Item Bias | 10 | 5 | 17.4374% | 8.9060% | 12.5000% | 6.6136% |
| Cart Popularity | — | — | — | — | 14.4969% | 7.6119% |

Item Bias improves No Bias by 1.7296 Recall points and 0.7640 NDCG points. It remains 0.5267 Recall points and 0.2972 NDCG points below Cart Popularity.

## Bias diagnostics and popularity relationship

Item Bias distribution is mean -0.3728, std 0.0977, min -0.5576, median -0.3897, max -0.0619. Pearson/Spearman correlations are 0.7658/0.7162 with Train View count, 0.6731/0.5719 with Cart count, and 0.6706/0.5696 with Purchase count. The Bias therefore captures a clear global propensity component without receiving these counts as features.

For User+Item Bias, item bias mean/std is -0.2183/0.0618. User bias mean/std/min/median/max is -0.1087/0.0369/-0.1901/-0.1131/0.0000. Its correlation with Train View-pair count is Pearson -0.8613 and Spearman -0.9241; with Favorite/Cart/Purchase event volume it is -0.5432/-0.5869. The negative sign reflects BCE calibration under many sampled target-zero examples, not low activity as a semantic truth.

## Score decomposition and dominance

For Item Bias on Purchase candidates, personal variance is 0.00479 and item-bias variance is 0.01010, a ratio of 2.11. For User+Item Bias the ratio is 4.36. Item Bias is the larger score component, but it does not eliminate personalization: both models still produce 206 unique Top10 lists among 212 Users.

## Accuracy and personalization

| model | unique Top10 | User overlap | Cart Top10 overlap | recommended Cart mean |
| --- | ---: | ---: | ---: | ---: |
| No Bias | 206 | 26.4048% | 34.1509% | 12.5910 |
| Item Bias | 206 | 50.2692% | 45.4245% | 14.4519 |
| User+Item Bias | 206 | 56.1111% | 49.5283% | 14.9943 |

Item Bias moves rankings toward popularity and increases shared recommendations, but lists remain User-specific. The accuracy gain is therefore a combination of a global item effect and retained personal components, rather than a complete collapse to Cart ranking.

## Secondary Test tasks

Item Bias achieves View+ Recall/NDCG@10 8.0509%/8.4423% and Favorite+ 9.3470%/5.9924%. User+Item Bias achieves 8.0747%/8.5464% and 9.2115%/5.5708%. Both exceed No Bias on these secondary tasks.

## Stability, hypothesis, and recommendation

Validation-to-Test Recall/NDCG changes are -5.1968/-3.2451 points for No Bias, -5.4904/-2.9788 for Item Bias, and -4.9374/-2.2924 for User+Item Bias. User+Item is more stable, but its absolute Test accuracy is lower.

The main hypothesis is supported: explicit item propensity materially closes the Cart gap while retaining personalized lists. Use **Item Bias Weighted BCE MF** as the next main architecture. Cart Popularity is still narrowly ahead, so the next isolated experiment should test Cart-centered positive design. Later candidates are latent dimension, exposed/unknown ratio, BPR with exposed comparisons, confidence design, or a separately regularized popularity prior. No setting was changed after Test.
