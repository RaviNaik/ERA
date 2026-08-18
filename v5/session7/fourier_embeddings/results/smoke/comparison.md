# Experiment: smoke

Total wall clock: 33.8s

| Arm | Params (total) | Params (codec) | Final val loss | Best val loss | Final val ppl | Tokens trained |
|---|---|---|---|---|---|---|
| dense | 627,008 | 262,144 | 8.2336 | 8.2369 | 3765.53 | 20,480 |
| kronecker | 627,008 | 262,144 | 8.2264 | 8.2290 | 3738.26 | 20,480 |
| fourier | 627,008 | 262,144 | 8.1844 | 8.1925 | 3584.44 | 20,480 |
| fourier_narrow | 627,008 | 262,144 | 8.1836 | 8.1889 | 3581.78 | 20,480 |
| hrr | 381,248 | 16,384 | 8.1958 | 8.1968 | 3625.60 | 20,480 |
