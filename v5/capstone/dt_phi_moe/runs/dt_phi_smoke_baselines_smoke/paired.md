## Paired comparison with the base model (same questions)

| variant, format | n | base % | variant % | change (points) | 95% interval | base-only / variant-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRA, training format | 16 | 18.8 | 18.8 | +0.0 | [-18.8, +18.8] | 1 / 1 | 1 |
| LoRA, chat template | 16 | 12.5 | 12.5 | +0.0 | [+0.0, +0.0] | 0 / 0 | 1 |
| Full FT, training format | 16 | 18.8 | 25.0 | +6.2 | [-12.5, +25.0] | 1 / 2 | 1 |
| Full FT, chat template | 16 | 12.5 | 12.5 | +0.0 | [+0.0, +0.0] | 0 / 0 | 1 |

Interval: bootstrap over questions, 10000 resamples, seed 0. 'base-only / variant-only' are the questions only the base model / only the variant answered correctly; McNemar p is the exact two-sided test on those two counts.
