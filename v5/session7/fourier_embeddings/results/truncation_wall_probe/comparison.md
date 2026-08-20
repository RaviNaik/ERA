# Experiment: truncation_wall_probe

Total wall clock: 7658.3s

| Arm | Params (total) | Params (codec) | Final val loss | Best val loss | Final val ppl | Tokens trained |
|---|---|---|---|---|---|---|
| dense | 110,906,112 | 12,582,912 | 0.2077 | 0.2083 | 1.23 | 327,680,000 |
| kronecker | 101,468,928 | 3,145,728 | 0.2100 | 0.2122 | 1.23 | 327,680,000 |
| fourier | 104,614,656 | 6,291,456 | 0.2185 | 0.2123 | 1.24 | 327,680,000 |
| fourier_narrow | 104,614,656 | 6,291,456 | 0.2238 | 0.2171 | 1.25 | 327,680,000 |
