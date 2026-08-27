# Synthetic Product v1 Catalog Quality Report

This report audits the frozen candidate catalog without changing category prototypes, noise, rarity, or price generation rules.
Population standard deviation (`ddof=0`) and linearly interpolated quartiles are used.

## Catalog

- Version: `synthetic_product_v1`
- Seed: `42`
- Product count: `200`

## Attribute descriptive statistics

| attribute | count | mean | std | min | 25% | median | 75% | max | <=0.05 | >=0.95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| danger | 200 | 0.5773 | 0.2449 | 0.0138 | 0.3724 | 0.6322 | 0.7669 | 1.0000 | 1.0% | 3.5% |
| luxury | 200 | 0.8173 | 0.1120 | 0.5101 | 0.7540 | 0.8244 | 0.8941 | 1.0000 | 0.0% | 11.5% |
| novelty | 200 | 0.8301 | 0.1185 | 0.5816 | 0.7359 | 0.8282 | 0.9413 | 1.0000 | 0.0% | 23.0% |
| historical_value | 200 | 0.6405 | 0.2366 | 0.1274 | 0.4361 | 0.6418 | 0.8518 | 1.0000 | 0.0% | 15.5% |
| technology_level | 200 | 0.3225 | 0.3182 | 0.0000 | 0.1016 | 0.1839 | 0.3746 | 1.0000 | 10.0% | 7.0% |
| natural_significance | 200 | 0.4749 | 0.3596 | 0.0000 | 0.1546 | 0.3359 | 0.9134 | 1.0000 | 4.0% | 19.0% |
| fantasy_level | 200 | 0.3954 | 0.3202 | 0.0000 | 0.1738 | 0.2529 | 0.6069 | 1.0000 | 2.5% | 14.0% |
| space_affinity | 200 | 0.2206 | 0.3292 | 0.0000 | 0.0061 | 0.0889 | 0.2222 | 1.0000 | 38.0% | 11.5% |
| power | 200 | 0.8085 | 0.1467 | 0.3831 | 0.7006 | 0.8158 | 0.9466 | 1.0000 | 0.0% | 24.5% |

## Parent-category attribute means

| category | danger | luxury | novelty | historical_value | technology_level | natural_significance | fantasy_level | space_affinity | power |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Abstract & Phenomena | 0.716 | 0.761 | 0.941 | 0.560 | 0.312 | 0.476 | 0.735 | 0.196 | 0.952 |
| Art & Culture | 0.153 | 0.911 | 0.713 | 0.929 | 0.144 | 0.247 | 0.372 | 0.032 | 0.635 |
| Creatures | 0.794 | 0.657 | 0.905 | 0.641 | 0.072 | 0.796 | 0.450 | 0.047 | 0.740 |
| Fantasy | 0.642 | 0.846 | 0.958 | 0.491 | 0.248 | 0.155 | 0.963 | 0.119 | 0.898 |
| Geography | 0.479 | 0.839 | 0.776 | 0.660 | 0.095 | 0.952 | 0.171 | 0.027 | 0.816 |
| History | 0.454 | 0.834 | 0.732 | 0.961 | 0.272 | 0.351 | 0.212 | 0.038 | 0.777 |
| Space | 0.739 | 0.876 | 0.855 | 0.536 | 0.359 | 0.705 | 0.137 | 0.972 | 0.824 |
| Technology | 0.599 | 0.800 | 0.803 | 0.381 | 0.921 | 0.121 | 0.352 | 0.179 | 0.845 |

## Child-category attribute means

| category | danger | luxury | novelty | historical_value | technology_level | natural_significance | fantasy_level | space_affinity | power |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Artwork | 0.101 | 0.951 | 0.730 | 0.879 | 0.134 | 0.211 | 0.398 | 0.003 | 0.531 |
| Concept | 0.413 | 0.755 | 0.964 | 0.737 | 0.146 | 0.306 | 0.850 | 0.093 | 0.994 |
| Cultural Treasure | 0.206 | 0.870 | 0.697 | 0.979 | 0.154 | 0.283 | 0.345 | 0.061 | 0.739 |
| Empire & Nation | 0.723 | 0.887 | 0.733 | 0.964 | 0.307 | 0.379 | 0.214 | 0.042 | 0.928 |
| Extinct Creature | 0.898 | 0.567 | 0.912 | 0.843 | 0.053 | 0.914 | 0.150 | 0.022 | 0.669 |
| Extraordinary Animal | 0.636 | 0.666 | 0.826 | 0.369 | 0.099 | 0.961 | 0.265 | 0.012 | 0.661 |
| Fictional Object | 0.563 | 0.888 | 0.953 | 0.526 | 0.436 | 0.135 | 0.947 | 0.176 | 0.857 |
| Historical Artifact | 0.358 | 0.796 | 0.800 | 0.971 | 0.272 | 0.246 | 0.250 | 0.041 | 0.628 |
| Impossible Technology | 0.687 | 0.880 | 0.975 | 0.307 | 0.962 | 0.111 | 0.633 | 0.234 | 0.982 |
| Island | 0.266 | 0.791 | 0.768 | 0.598 | 0.098 | 0.937 | 0.236 | 0.039 | 0.617 |
| Land & Territory | 0.413 | 0.874 | 0.715 | 0.726 | 0.111 | 0.949 | 0.095 | 0.021 | 0.906 |
| Machine & Device | 0.424 | 0.703 | 0.722 | 0.407 | 0.938 | 0.108 | 0.243 | 0.127 | 0.783 |
| Magical Item | 0.598 | 0.882 | 0.950 | 0.673 | 0.131 | 0.211 | 0.965 | 0.082 | 0.849 |
| Monument & Architecture | 0.282 | 0.819 | 0.662 | 0.947 | 0.237 | 0.429 | 0.171 | 0.033 | 0.775 |
| Mythical Creature | 0.836 | 0.748 | 0.976 | 0.686 | 0.065 | 0.497 | 0.972 | 0.110 | 0.898 |
| Natural Phenomenon | 0.907 | 0.617 | 0.922 | 0.296 | 0.089 | 0.973 | 0.379 | 0.089 | 0.885 |
| Ocean & Sea | 0.768 | 0.849 | 0.854 | 0.648 | 0.073 | 0.971 | 0.193 | 0.022 | 0.915 |
| Planet | 0.763 | 0.912 | 0.837 | 0.549 | 0.170 | 0.941 | 0.146 | 0.957 | 0.917 |
| Satellite | 0.544 | 0.869 | 0.867 | 0.596 | 0.244 | 0.889 | 0.044 | 0.982 | 0.673 |
| Spacecraft | 0.714 | 0.821 | 0.788 | 0.629 | 0.914 | 0.075 | 0.125 | 0.990 | 0.711 |
| Star & Cosmic Object | 0.936 | 0.906 | 0.939 | 0.356 | 0.055 | 0.970 | 0.236 | 0.961 | 0.997 |
| Superpower | 0.770 | 0.762 | 0.971 | 0.251 | 0.192 | 0.113 | 0.976 | 0.105 | 0.996 |
| Time & Reality | 0.828 | 0.912 | 0.938 | 0.648 | 0.702 | 0.149 | 0.977 | 0.407 | 0.978 |
| Vehicle | 0.686 | 0.816 | 0.711 | 0.428 | 0.864 | 0.145 | 0.179 | 0.177 | 0.768 |

## Pearson attribute correlation matrix

| attribute | danger | luxury | novelty | historical_value | technology_level | natural_significance | fantasy_level | space_affinity | power |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| danger | 1.000 | -0.227 | 0.528 | -0.464 | 0.070 | 0.234 | 0.153 | 0.343 | 0.517 |
| luxury | -0.227 | 1.000 | -0.076 | 0.152 | 0.051 | -0.199 | 0.075 | 0.237 | 0.151 |
| novelty | 0.528 | -0.076 | 1.000 | -0.410 | -0.110 | 0.042 | 0.598 | 0.167 | 0.403 |
| historical_value | -0.464 | 0.152 | -0.410 | 1.000 | -0.345 | -0.043 | -0.214 | -0.284 | -0.352 |
| technology_level | 0.070 | 0.051 | -0.110 | -0.345 | 1.000 | -0.621 | -0.009 | 0.204 | 0.085 |
| natural_significance | 0.234 | -0.199 | 0.042 | -0.043 | -0.621 | 1.000 | -0.455 | 0.145 | -0.025 |
| fantasy_level | 0.153 | 0.075 | 0.598 | -0.214 | -0.009 | -0.455 | 1.000 | -0.209 | 0.400 |
| space_affinity | 0.343 | 0.237 | 0.167 | -0.284 | 0.204 | 0.145 | -0.209 | 1.000 | 0.132 |
| power | 0.517 | 0.151 | 0.403 | -0.352 | 0.085 | -0.025 | 0.400 | 0.132 | 1.000 |

Absolute correlations >= 0.8:

- None

## Rarity and novelty

Rarity: mean 0.7531, std 0.1005, min 0.5322, 25% 0.6786, median 0.7579, 75% 0.8246, max 0.9895.
Pearson correlation between rarity and novelty: **0.6279**.

## Price

| metric | min | 25% | median | 75% | max |
| --- | --- | --- | --- | --- | --- |
| price | 21,060,000 | 36,515,000,000 | 20,160,000,000,000 | 110,830,000,000,000,000,000 | 198,700,000,000,000,000,000,000,000 |
| log10(price) | 7.3235 | 10.5624 | 13.3027 | 19.8587 | 26.2982 |

Parent-category median price:

- Abstract & Phenomena: 342,500,000,000,000,000,000
- Art & Culture: 343,570,000,000
- Creatures: 11,570,000,000
- Fantasy: 44,820,000,000
- Geography: 11,620,000,000,000,000,000,000
- History: 7,326,500,000,000,000
- Space: 159,590,500,000,000,000,000
- Technology: 7,464,500,000

Price digit-count distribution:

- 8 digits: 1
- 9 digits: 11
- 10 digits: 21
- 11 digits: 32
- 12 digits: 23
- 13 digits: 11
- 14 digits: 13
- 15 digits: 15
- 16 digits: 4
- 17 digits: 5
- 18 digits: 1
- 19 digits: 10
- 20 digits: 3
- 21 digits: 10
- 22 digits: 8
- 23 digits: 18
- 24 digits: 3
- 25 digits: 4
- 26 digits: 4
- 27 digits: 3

Correlation with `log10(price)`:

- `rarity`: 0.1933
- `luxury`: 0.3648
- `power`: 0.4962
- `historical_value`: 0.1034

## Reality type and status

Reality type:

- `abstract`: 7
- `fictional`: 36
- `historical`: 45
- `real`: 74
- `speculative`: 38

Status:

- `available`: 170
- `coming_soon`: 10
- `sold_out`: 14
- `unavailable`: 6

## Tag usage

- `valuable`: 97
- `powerful`: 91
- `rare`: 66
- `exclusive`: 65
- `luxury`: 63
- `natural`: 63
- `mysterious`: 61
- `massive`: 52
- `cultural`: 51
- `historic`: 46
- `beautiful`: 41
- `technology`: 38
- `collectible`: 37
- `fantasy`: 35
- `scientific`: 32
- `space`: 30
- `artificial`: 28
- `dangerous`: 28
- `impossible`: 28
- `legendary`: 28
- `unexplored`: 25
- `destructive`: 21
- `prehistoric`: 9

Unused vocabulary tags: habitable, portable.

## Representative cosine nearest neighbors

### Moon

1. Silent Moon — 0.9965
2. Obsidian Moon — 0.9957
3. Titan — 0.9950
4. Jupiter — 0.9943
5. Crimson Exoplanet — 0.9942

### Time Machine

1. Crimson Impossible Device — 0.9990
2. Celestial Impossible Device — 0.9990
3. Silent Impossible Device — 0.9982
4. Teleportation Device — 0.9976
5. Obsidian Impossible Device — 0.9955

### Tyrannosaurus Rex

1. Forgotten Extinct Beast — 0.9981
2. Crimson Extinct Beast — 0.9968
3. Silent Extinct Beast — 0.9960
4. Luminous Extinct Beast — 0.9929
5. Obsidian Extinct Beast — 0.9919

### Roman Empire

1. Silent Lost Realm — 0.9979
2. Byzantine Empire — 0.9973
3. Celestial Lost Realm — 0.9966
4. Forgotten Lost Realm — 0.9942
5. Crimson Lost Realm — 0.9938

### Luck

1. Silence — 0.9959
2. Infinite Patience — 0.9953
3. Silent Abstract Concept — 0.9946
4. Forgotten Enchanted Artifact — 0.9942
5. Crimson Abstract Concept — 0.9937

## Interpretation and detected concerns

- Category prototypes remain clearly visible: Space leads `space_affinity` (0.972), Technology leads `technology_level` (0.921), History leads `historical_value` (0.961), Fantasy leads `fantasy_level` (0.963), and Geography/Creatures are high on `natural_significance` (0.952/0.796). Child-category means retain additional local structure.
- The feature space is not collapsed, but luxury, novelty, and power have high centers and compressed upper tails. This reduces headroom for synthetic-user preference discrimination among premium items.
- No attribute pair reaches the |r| >= 0.8 redundancy threshold. Rarity and novelty are related but not interchangeable.
- Raw positive-valued vectors produce very high cosine scores, so neighbor ranking is mostly locally sensible but weakly separated. Future model experiments should compare centered/scaled features; this report does not implement a recommender.
- Price spans many orders of magnitude by design. Log-price correlations and category medians show that category base price remains a major driver alongside generated factors; extreme values fit the marketplace premise and should not be removed solely as outliers.
- 23 of 25 configured tags are used; `habitable` and `portable` are unused. The most common tag (`valuable`) appears on 48.5% of products, so no tag is close to universal. The used vocabulary has meaningful content-based variation.

## Freeze recommendation

Freeze this exact seed-42 snapshot as the tracked **v1 catalog** and use it for the first Synthetic User/Event experiments. Category separation, non-redundant attributes, reproducibility, and tag variation are sufficient for that milestone. Treat the high luxury/novelty/power centers, compressed raw cosine scores, and two unused tags as documented limitations. A v1.1 can broaden the catalog later if initial simulator diagnostics show weak preference separation. The current generator has not been modified by this audit.

Suggested v1.1 proposal (not implemented):

| Item | Proposal |
| --- | --- |
| Current issue | Luxury, novelty, and power are high across much of the catalog, raw cosine distances are compressed, and two vocabulary tags are unused. |
| Why it matters | Synthetic users with different premium/power preferences may receive less differentiated relevance signals. |
| Suggested v1.1 change | Add a controlled ordinary/lower-intensity slice or minimally lower only the affected category prototype axes; keep nine-axis schema and category-specific peaks. |
| Expected effect | Wider usable range and neighbor separation without removing the intentionally extravagant long-tail catalog. |
