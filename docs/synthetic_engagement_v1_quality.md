# Synthetic Engagement v1 Quality Report

This report audits synthetic behavior rules, not a recommendation model.

## Frozen world

- Products: `synthetic_product_v1`, seed `42`, 200 products
- Users: `synthetic_user_v1`, seed `42`, 1,000 users
- Exposure/View: `synthetic_session_event_v1`, seed `42`
- Engagement: `synthetic_engagement_v1`, seed `42`

## Funnel

- Impressions: 93,710
- Views: 30,537
- Favorites: 4,385
- Carts: 1,881
- Purchases: 1,259

| conversion | rate |
| --- | --- |
| impression → view | 32.59% |
| view → favorite | 14.36% |
| view → cart | 6.16% |
| view → purchase | 4.12% |
| favorite → cart | 11.70% |
| favorite → purchase | 9.92% |
| cart → purchase | 29.88% |

## User-level distribution

| event | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| favorite | 4.385 | 3.281 | 0 | 4.0 | 18 |
| add_to_cart | 1.881 | 1.865 | 0 | 1.5 | 13 |
| purchase | 1.259 | 1.535 | 0 | 1.0 | 11 |

- Users with zero Purchases: 409
- Users with at least one Purchase: 591

## Activity-tier conversion

| group | views | favorites | carts | purchases | purchase/view |
| --- | --- | --- | --- | --- | --- |
| casual | 1,625 | 215 | 92 | 53 | 3.26% |
| heavy | 6,213 | 863 | 388 | 279 | 4.49% |
| regular | 22,699 | 3,307 | 1,401 | 927 | 4.08% |

## Archetype conversion

| group | views | favorites | carts | purchases | purchase/view |
| --- | --- | --- | --- | --- | --- |
| Curious Generalist | 4,255 | 608 | 243 | 149 | 3.50% |
| Eclectic Browser | 4,602 | 628 | 282 | 198 | 4.30% |
| Fantasy Lover | 2,454 | 372 | 183 | 91 | 3.71% |
| History Collector | 1,851 | 265 | 93 | 63 | 3.40% |
| Luxury Collector | 2,013 | 279 | 110 | 79 | 3.92% |
| Nature Explorer | 2,773 | 391 | 120 | 60 | 2.16% |
| Power Seeker | 2,803 | 392 | 212 | 170 | 6.06% |
| Space Enthusiast | 2,882 | 433 | 189 | 127 | 4.41% |
| Tech Futurist | 3,592 | 518 | 202 | 127 | 3.54% |
| Thrill Seeker | 3,312 | 499 | 247 | 195 | 5.89% |

## Product-level distribution

| event | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| favorite | 21.925 | 10.083 | 6 | 19.0 | 53 |
| add_to_cart | 9.405 | 7.061 | 0 | 8.5 | 41 |
| purchase | 6.295 | 5.282 | 0 | 5.0 | 25 |

- Products with zero Purchases: 31
- Purchase Gini: 0.4548
- Top-10 Purchase share: 16.36%
- Top products: Teleportation Device (25), Luminous Enchanted Artifact (24), Luminous Impossible Ability (23), Invulnerability (21), Crimson Legendary Object (21), Golden Legendary Object (19), Silent Impossible Ability (19), Obsidian Enchanted Artifact (18), Matter Replicator (18), Crimson Impossible Device (18)

## Preference signal

- impression: mean preference match 0.5344
- view: mean preference match 0.5502
- favorite: mean preference match 0.5759
- add_to_cart: mean preference match 0.5799
- purchase: mean preference match 0.5845
- Purchased viewed pairs: 0.5845
- Non-purchased viewed pairs: 0.5431

## Price signal

| budget | views | favorites | carts | purchases |
| --- | --- | --- | --- | --- |
| within | 17,412 | 2,451 (14.08%) | 1,240 (7.12%) | 885 (5.08%) |
| over | 13,125 | 1,934 (14.74%) | 641 (4.88%) | 374 (2.85%) |

## Impulsiveness signal

| quartile | direct cart/view | direct purchase/view | one-view purchase/view | over-budget purchase/view |
| --- | --- | --- | --- | --- |
| low | 3.27% | 1.20% | 2.28% | 0.56% |
| high | 5.34% | 2.21% | 4.80% | 1.92% |

## Delayed conversion

| event | same session | later session | later share |
| --- | --- | --- | --- |
| favorite | 3,679 | 706 | 16.10% |
| add_to_cart | 1,464 | 417 | 22.17% |
| purchase | 808 | 451 | 35.82% |

Later-session Purchases with Favorite/Cart state: 305.

## Top funnel paths

- view → exit: 21,316 (77.35%)
- view → favorite: 3,665 (13.30%)
- view → add_to_cart: 1,034 (3.75%)
- view → purchase: 490 (1.78%)
- view → add_to_cart → purchase: 334 (1.21%)
- view → favorite → add_to_cart: 285 (1.03%)
- view → favorite → add_to_cart → purchase: 228 (0.83%)
- view → favorite → purchase: 207 (0.75%)

## Single-feature Purchase AUC

- preference_match: 0.5865
- price_compatibility: 0.5962
- impulsiveness: 0.5913

## Interaction sparsity

- Matrix cells: 200,000
- Unique View-or-stronger pairs: 27,559 (13.78% density; 86.22% sparse)
- Favorite pairs: 4,385
- Cart pairs: 1,881
- Purchase pairs: 1,259 (0.63% density)

## Interpretation and freeze recommendation

- View, Favorite, Cart, and Purchase represent distinct intent levels and use different utilities.
- Observed Event is not true preference: exposure, price, state, impulsiveness, status, and noise all affect the funnel.
- Favorite and Cart are helpful prior states, not mandatory gates; direct Cart and Purchase paths remain possible.
- Event weights for recommendation datasets are intentionally not assigned in this generator.
- View→Purchase is 4.12%, only 0.12 percentage points above the initial non-binding 1–4% guide; its count and downstream diagnostics remain sane.
- The reported delayed paths, price signal, concentration, and single-feature AUCs support freezing this exact seed-42 v1 population.
