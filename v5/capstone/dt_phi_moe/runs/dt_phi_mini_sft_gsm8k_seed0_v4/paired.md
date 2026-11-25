## Paired comparison with the base model (same questions)

| variant, format | n | base % | variant % | change (points) | 95% interval | base-only / variant-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deletion Tuning, training format | 200 | 33.5 | 78.5 | +45.0 | [+37.0, +53.0] | 8 / 98 | 8.1e-21 |
| Deletion Tuning, chat template | 200 | 58.5 | 71.5 | +13.0 | [+7.0, +19.0] | 7 / 33 | 4.2e-05 |

Interval: bootstrap over questions, 10000 resamples, seed 0. 'base-only / variant-only' are the questions only the base model / only the variant answered correctly; McNemar p is the exact two-sided test on those two counts.
