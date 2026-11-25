## Paired comparison with the base model (same questions)

| variant, format | n | base % | variant % | change (points) | 95% interval | base-only / variant-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deletion Tuning, chat template | 1319 | 56.0 | 67.1 | +11.1 | [+8.8, +13.5] | 62 / 209 | 8.5e-20 |
| LoRA, chat template | 1319 | 56.0 | 75.0 | +19.0 | [+16.5, +21.6] | 49 / 300 | 3.9e-45 |
| Full FT, chat template | 1319 | 56.0 | 54.4 | -1.6 | [-4.6, +1.5] | 226 / 205 | 0.34 |

Interval: bootstrap over questions, 10000 resamples, seed 0. 'base-only / variant-only' are the questions only the base model / only the variant answered correctly; McNemar p is the exact two-sided test on those two counts.
