# Synthetic Session / Impression / View v1 Quality Report

This audit evaluates a synthetic exposure and browsing mechanism. It is not a recommendation model.

## Fixed world and window

- Products: `synthetic_product_v1`, seed `42`, 200 records
- Users: `synthetic_user_v1`, seed `42`, 1,000 records
- Interactions: `synthetic_session_event_v1`, seed `42`
- Window: `2026-01-01` through `2026-01-30` UTC

## Session summary

- Total sessions: 5,608
- Sessions/user: mean 5.608, std 2.713, min 1, median 5.0, max 17
- Duration seconds: mean 585.6, median 535.0, min 185, max 2100
- Impressions/session: mean 16.710, std 4.341, min 5, median 17.0, max 30
- Views/session: mean 5.445, std 2.435, min 0, median 5.0, max 17

Activity-tier sessions/user:

| tier | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| casual | 3.417 | 1.891 | 1 | 3.0 | 9 |
| heavy | 7.769 | 2.762 | 3 | 8.0 | 14 |
| regular | 5.621 | 2.539 | 1 | 5.0 | 17 |

## Event summary

- Total Events: 124,247
- Impressions: 93,710
- Views: 30,537
- Overall Impression → View rate: **32.59%**
- User-level view rate: mean 32.40%, std 6.64%, min 0.00%, median 32.49%, max 64.71%

View rate by exposure source:

| source | impressions | views | view rate |
| --- | --- | --- | --- |
| exploration | 18,165 | 5,690 | 31.32% |
| popular | 19,291 | 6,401 | 33.18% |
| preference | 47,134 | 15,572 | 33.04% |
| random | 9,120 | 2,874 | 31.51% |

## Archetype view rates

| group | impressions | views | view rate |
| --- | --- | --- | --- |
| Curious Generalist | 14,297 | 4,255 | 29.76% |
| Eclectic Browser | 14,795 | 4,602 | 31.11% |
| Fantasy Lover | 7,399 | 2,454 | 33.17% |
| History Collector | 5,672 | 1,851 | 32.63% |
| Luxury Collector | 6,321 | 2,013 | 31.85% |
| Nature Explorer | 8,931 | 2,773 | 31.05% |
| Power Seeker | 7,911 | 2,803 | 35.43% |
| Space Enthusiast | 8,201 | 2,882 | 35.14% |
| Tech Futurist | 10,648 | 3,592 | 33.73% |
| Thrill Seeker | 9,535 | 3,312 | 34.74% |

## Category view rates

| group | impressions | views | view rate |
| --- | --- | --- | --- |
| Abstract & Phenomena | 8,262 | 2,681 | 32.45% |
| Art & Culture | 8,474 | 2,676 | 31.58% |
| Creatures | 11,211 | 3,967 | 35.38% |
| Fantasy | 14,009 | 5,382 | 38.42% |
| Geography | 10,767 | 2,958 | 27.47% |
| History | 12,352 | 3,587 | 29.04% |
| Space | 14,843 | 4,530 | 30.52% |
| Technology | 13,792 | 4,756 | 34.48% |

## Exposure concentration

- Product impression count: min 310, median 454.0, max 748
- Gini coefficient: 0.1177
- Top-10 product impression share: 7.49%
- Products with zero impressions: 0

Top 10 exposed products:

- A Second Timeline: 748
- Luck: 738
- Golden Cosmic Object: 735
- Silent Cosmic Object: 732
- Crimson Legendary Object: 704
- Invisibility Cloak: 699
- Obsidian Enchanted Artifact: 681
- One Extra Hour Per Day: 665
- Jupiter: 664
- Crimson Impossible Device: 652

## Preference signal and task difficulty

- Viewed impressions mean match: 0.5502
- Non-viewed impressions mean match: 0.5267
- Match-only AUC: 0.5487
- Best single match threshold accuracy: 67.41%
- Majority-class baseline accuracy: 67.41%

Preference match contributes signal but does not nearly determine View labels by itself.

## Price signal

- Within-budget: 17,412 / 49,988 viewed (34.83%)
- Over-budget: 13,125 / 43,722 viewed (30.02%)

Over-budget Views remain possible; price is soft friction rather than a hard gate.

## Popularity signal

- Low popularity-preference users' viewed mean prior: 0.5891
- High popularity-preference users' viewed mean prior: 0.5998

## Exploration signal

- Outside-primary-category viewed share, low/high exploration: 67.56% / 75.07%
- Lower-match viewed share, low/high exploration: 48.16% / 51.82%
- Mean viewed match, low/high exploration: 0.5544 / 0.5465

## Weak session continuity

After a View, 15.30% of eligible next Impressions share its category (76,585 transitions).

## Interpretation and freeze recommendation

- Exposure mixes preference, popularity, exploration, and random sources. It is not identical to user preference.
- A non-view does not prove dislike: the product may have been shown under noisy exposure, while an unexposed product produces no label at all.
- View rate, event volume, activity heterogeneity, exposure concentration, soft price friction, and match-only predictability are all within the intended v1 sanity ranges.
- Freeze this exact `synthetic_session_event_v1 / seed 42` population for the next funnel-design phase. Favorite/cart/purchase remain unimplemented.
