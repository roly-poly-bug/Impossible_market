# Synthetic User v1 Quality Report

This audit evaluates hidden simulator ground truth. It does not generate Events or implement a recommendation model.
Population standard deviation (`ddof=0`) and linearly interpolated quartiles are used.

## Population

- User version: `synthetic_user_v1`
- User seed/count: `42` / `1000`
- Frozen catalog: `synthetic_product_v1`, seed `42`, count `200`

## Archetype distribution

- Power Seeker: 80 (8.0%)
- Tech Futurist: 100 (10.0%)
- Luxury Collector: 90 (9.0%)
- Thrill Seeker: 80 (8.0%)
- Curious Generalist: 150 (15.0%)
- Space Enthusiast: 90 (9.0%)
- Eclectic Browser: 130 (13.0%)
- History Collector: 90 (9.0%)
- Nature Explorer: 100 (10.0%)
- Fantasy Lover: 90 (9.0%)

Mixed-preference users: 323 (32.3%).

## Preference descriptive statistics

| preference | count | mean | std | min | 25% | median | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| danger_preference | 1000 | 0.5132 | 0.1989 | 0.0000 | 0.3743 | 0.5035 | 0.6457 | 1.0000 |
| luxury_preference | 1000 | 0.5657 | 0.1921 | 0.0000 | 0.4393 | 0.5544 | 0.6980 | 1.0000 |
| novelty_preference | 1000 | 0.6550 | 0.1697 | 0.1456 | 0.5464 | 0.6615 | 0.7710 | 1.0000 |
| historical_preference | 1000 | 0.4963 | 0.2010 | 0.0000 | 0.3518 | 0.4798 | 0.6304 | 1.0000 |
| technology_preference | 1000 | 0.5154 | 0.2083 | 0.0000 | 0.3702 | 0.5127 | 0.6553 | 1.0000 |
| nature_preference | 1000 | 0.5108 | 0.2107 | 0.0000 | 0.3709 | 0.5003 | 0.6448 | 1.0000 |
| fantasy_preference | 1000 | 0.4986 | 0.2092 | 0.0000 | 0.3515 | 0.4925 | 0.6410 | 1.0000 |
| space_preference | 1000 | 0.4465 | 0.2047 | 0.0000 | 0.3222 | 0.4369 | 0.5710 | 1.0000 |
| power_preference | 1000 | 0.6229 | 0.1784 | 0.0101 | 0.4899 | 0.6295 | 0.7443 | 1.0000 |

## Archetype preference means

| archetype | danger_preference | luxury_preference | novelty_preference | historical_preference | technology_preference | nature_preference | fantasy_preference | space_preference | power_preference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Curious Generalist | 0.485 | 0.530 | 0.631 | 0.501 | 0.516 | 0.512 | 0.509 | 0.486 | 0.560 |
| Eclectic Browser | 0.574 | 0.520 | 0.680 | 0.475 | 0.568 | 0.565 | 0.584 | 0.531 | 0.587 |
| Fantasy Lover | 0.527 | 0.578 | 0.702 | 0.458 | 0.361 | 0.396 | 0.831 | 0.303 | 0.657 |
| History Collector | 0.363 | 0.646 | 0.518 | 0.812 | 0.373 | 0.491 | 0.388 | 0.288 | 0.575 |
| Luxury Collector | 0.343 | 0.828 | 0.590 | 0.613 | 0.500 | 0.422 | 0.493 | 0.392 | 0.617 |
| Nature Explorer | 0.462 | 0.425 | 0.630 | 0.512 | 0.345 | 0.834 | 0.328 | 0.340 | 0.500 |
| Power Seeker | 0.631 | 0.588 | 0.619 | 0.464 | 0.562 | 0.392 | 0.572 | 0.424 | 0.835 |
| Space Enthusiast | 0.505 | 0.566 | 0.743 | 0.413 | 0.613 | 0.520 | 0.383 | 0.796 | 0.607 |
| Tech Futurist | 0.473 | 0.585 | 0.716 | 0.386 | 0.793 | 0.328 | 0.416 | 0.459 | 0.643 |
| Thrill Seeker | 0.817 | 0.437 | 0.725 | 0.324 | 0.490 | 0.605 | 0.469 | 0.384 | 0.756 |

## Pearson preference correlation matrix

| preference | danger_preference | luxury_preference | novelty_preference | historical_preference | technology_preference | nature_preference | fantasy_preference | space_preference | power_preference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| danger_preference | 1.000 | -0.243 | 0.147 | -0.276 | 0.054 | 0.030 | 0.113 | 0.052 | 0.173 |
| luxury_preference | -0.243 | 1.000 | -0.057 | 0.209 | 0.035 | -0.221 | 0.048 | -0.039 | 0.014 |
| novelty_preference | 0.147 | -0.057 | 1.000 | -0.209 | 0.107 | -0.002 | 0.044 | 0.124 | -0.024 |
| historical_preference | -0.276 | 0.209 | -0.209 | 1.000 | -0.190 | -0.009 | -0.065 | -0.203 | -0.122 |
| technology_preference | 0.054 | 0.035 | 0.107 | -0.190 | 1.000 | -0.226 | -0.066 | 0.259 | 0.086 |
| nature_preference | 0.030 | -0.221 | -0.002 | -0.009 | -0.226 | 1.000 | -0.173 | -0.020 | -0.167 |
| fantasy_preference | 0.113 | 0.048 | 0.044 | -0.065 | -0.066 | -0.173 | 1.000 | -0.107 | 0.126 |
| space_preference | 0.052 | -0.039 | 0.124 | -0.203 | 0.259 | -0.020 | -0.107 | 1.000 | -0.046 |
| power_preference | 0.173 | 0.014 | -0.024 | -0.122 | 0.086 | -0.167 | 0.126 | -0.046 | 1.000 |

Absolute correlations >= 0.8:

- None

## Archetype separation sanity check

- Nearest-prototype accuracy: **64.4%**
- Pure-profile accuracy (677 users): 74.3%
- Mixed-profile accuracy (323 users): 43.7%
- Mean distance to assigned prototype: 0.4727
- Mean distance to closest competing prototype: 0.5272

The primary archetype signal is detectable but far from perfectly recoverable. Mixed profiles materially increase overlap, avoiding a trivially separable ten-cluster dataset.

## Budget versus frozen product prices

Budget log10: min 7.2000, 25% 12.1908, median 14.9837, 75% 17.7328, max 27.2000.
Budget / price-sensitivity correlation: -0.0495.
Affordable catalog share: min 0.0%, 25% 44.5%, median 63.5%, 75% 68.5%, max 100.0%.
Users with zero affordable products: 19; users able to afford all products: 2.

| budget tier | users | mean affordable | min | max |
| --- | --- | --- | --- | --- |
| absurd | 11 | 96.7% | 94.5% | 100.0% |
| high | 388 | 68.2% | 63.5% | 75.0% |
| low | 153 | 12.9% | 0.0% | 32.5% |
| medium | 350 | 49.3% | 32.5% | 63.5% |
| ultra_high | 98 | 81.9% | 75.0% | 94.5% |

Budget tier distribution:

- `ultra_high`: 98
- `high`: 388
- `low`: 153
- `medium`: 350
- `absurd`: 11

Behavioral parameter statistics:

| parameter | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| price_sensitivity | 0.4916 | 0.2136 | 0.0000 | 0.4871 | 1.0000 |
| popularity_preference | 0.5368 | 0.2083 | 0.0000 | 0.5433 | 1.0000 |
| exploration_tendency | 0.6159 | 0.2308 | 0.0000 | 0.6270 | 1.0000 |
| impulsiveness | 0.5229 | 0.2361 | 0.0000 | 0.5184 | 1.0000 |
| activity_level | 0.5741 | 0.1869 | 0.0263 | 0.5713 | 1.0000 |

Activity tier distribution:

- `regular`: 763
- `heavy`: 117
- `casual`: 120

## Representative ground-truth preference matches

### Space Enthusiast

1. Voyager 1 (Space) — alignment 0.5981
2. Crimson Moon (Space) — alignment 0.5773
3. Titan (Space) — alignment 0.5738
4. Crimson Starship (Space) — alignment 0.5724
5. Luminous Starship (Space) — alignment 0.5720

### History Collector

1. Hanging Gardens (History) — alignment 0.7111
2. Forgotten Ancient Relic (History) — alignment 0.6770
3. Imperial Regalia (Art & Culture) — alignment 0.6756
4. Silent Monument (History) — alignment 0.6753
5. Forgotten Cultural Treasure (Art & Culture) — alignment 0.6711

### Tech Futurist

1. Luminous Experimental Machine (Technology) — alignment 0.7235
2. Silent Experimental Machine (Technology) — alignment 0.7003
3. Celestial Concept Vehicle (Technology) — alignment 0.6949
4. Crimson Concept Vehicle (Technology) — alignment 0.6793
5. Personal Submarine (Technology) — alignment 0.6742

### Nature Explorer

1. Giant Blue Whale (Creatures) — alignment 0.6536
2. Crimson Hidden Island (Geography) — alignment 0.6535
3. Silent Extraordinary Animal (Creatures) — alignment 0.6315
4. Golden Hidden Island (Geography) — alignment 0.6234
5. Luminous Hidden Island (Geography) — alignment 0.6203

### Fantasy Lover

1. Silent Legendary Object (Fantasy) — alignment 0.6830
2. Luminous Legendary Object (Fantasy) — alignment 0.6640
3. One Ring (Fantasy) — alignment 0.6439
4. Golden Legendary Object (Fantasy) — alignment 0.6375
5. Golden Enchanted Artifact (Fantasy) — alignment 0.6360

## Assessment

- All nine preference axes have useful dispersion, with standard deviations around 0.17–0.21 and no collapsed dimension.
- Intended archetype signatures remain visible, while 32.3% mixed users and individual noise keep nearest-prototype accuracy at 64.4% rather than near 100%.
- No preference pair reaches the |r| >= 0.8 redundancy threshold.
- Budget tiers create monotonic affordability differences; only a small edge population can afford none or all of the frozen catalog.
- Representative alignment checks return semantically appropriate product categories. These checks validate simulator structure and are not recommendation output.

## Freeze recommendation

Freeze `synthetic_user_v1 / seed 42 / 1000 users` for the first Session/Event-generation design. The population is structured, heterogeneous, reproducible, and not trivially separable. Revisit only if later event-funnel diagnostics reveal insufficient behavior variance or excessive budget gating.
