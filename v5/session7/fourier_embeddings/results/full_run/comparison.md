# Experiment: full_run

Total wall clock: 7605.5s

| Arm | Params (total) | Params (codec) | Final val loss | Best val loss | Final val ppl | Tokens trained |
|---|---|---|---|---|---|---|
| dense | 110,906,112 | 12,582,912 | 0.2083 | 0.2089 | 1.23 | 327,680,000 |
| kronecker | 104,614,656 | 6,291,456 | 0.2197 | 0.2136 | 1.25 | 327,680,000 |
| fourier | 104,614,656 | 6,291,456 | 0.2181 | 0.2118 | 1.24 | 327,680,000 |
| fourier_narrow | 104,614,656 | 6,291,456 | 0.2239 | 0.2171 | 1.25 | 327,680,000 |
