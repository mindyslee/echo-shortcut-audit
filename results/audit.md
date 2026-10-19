# Shortcut audit: `echonet_embeddings.txt`

n = 10030 videos, d = 400 dims. 1296.0 s.

## Decodability (cross-validated linear probe)

| target | role | kind | score | chance | random-matrix control | permutation null | z | p | decodable |
|---|---|---|---|---|---|---|---|---|---|
| EF | clinical | regression | **0.483** ± 0.02 | 0.00 | -0.006 | -0.005 ± 0.002 | 264.2 | 0.048 | yes |
| ESV | clinical | regression | **0.546** ± 0.03 | 0.00 | -0.008 | -0.005 ± 0.002 | 336.6 | 0.048 | yes |
| EDV | clinical | regression | **0.468** ± 0.03 | 0.00 | -0.009 | -0.005 ± 0.001 | 363.1 | 0.048 | yes |
| FPS | nuisance | regression | **0.299** ± 0.02 | 0.00 | -0.011 | -0.005 ± 0.001 | 265.2 | 0.048 | yes |
| FPS_modal | nuisance | classification | **0.826** ± 0.01 | 0.50 | 0.500 | 0.501 ± 0.008 | 43.1 | 0.048 | yes |
| NumberOfFrames | nuisance | regression | **0.436** ± 0.02 | 0.00 | -0.011 | -0.005 ± 0.001 | 294.5 | 0.048 | yes |

Regression score is R²; classification score is balanced accuracy (chance = 1/k). The permutation p-value is floored at 1/(20+1); z is (score − null mean)/null sd.

## Metadata-only baseline for EF

- linear R² from acquisition metadata alone: **0.017** (p = 0.048)
- boosted-trees R² from metadata alone: **0.033**
- embedding R² = 0.483; embedding + metadata R² = 0.488
- embedding beyond metadata: **0.471**; metadata beyond embedding: **0.005**

## EF error by acquisition stratum

| stratum | bins (n, MAE) | MAE range | Kruskal-Wallis p |
|---|---|---|---|
| FPS | (17.999, 50.0]: 8597, 6.71; (50.0, 138.0]: 1433, 6.67 | 0.04 | 0.961 |
| FPS_modal | fps=50: 7959, 6.51; other: 2071, 7.47 | 0.96 | 0.000 |
| NumberOfFrames | (144.0, 171.0]: 2472, 6.48; (171.0, 201.0]: 2637, 6.37; (201.0, 1002.0]: 2372, 6.30; (27.999, 144.0]: 2549, 7.65 | 1.35 | 0.000 |

## Price of erasing each nuisance (INLP)

| nuisance | decodability before → after | rank removed | EF: before → after (random ctrl) → excess price | ESV: before → after (random ctrl) → excess price | EDV: before → after (random ctrl) → excess price |
|---|---|---|---|---|---|
| FPS | 0.302 → -0.052 (chance 0.00) | 2 | 0.480 → 0.476 (0.479 ± 0.001) → **+0.002** | 0.543 → 0.535 (0.542 ± 0.001) → **+0.006** | 0.463 → 0.455 (0.462 ± 0.000) → **+0.008** |
| FPS_modal | 0.824 → 0.345 (chance 0.50) | 2 | 0.480 → 0.468 (0.479 ± 0.001) → **+0.010** | 0.543 → 0.529 (0.542 ± 0.001) → **+0.013** | 0.463 → 0.452 (0.462 ± 0.000) → **+0.011** |
| NumberOfFrames | 0.437 → -0.043 (chance 0.00) | 2 | 0.480 → 0.476 (0.479 ± 0.001) → **+0.003** | 0.543 → 0.542 (0.542 ± 0.001) → **-0.000** | 0.463 → 0.463 (0.462 ± 0.000) → **-0.001** |

Erasure runs in a PCA-whitened copy of the embedding until the nuisance is no longer linearly decodable. Every projection costs a little, so the same number of random directions is removed as a control; the **excess price** (random-control score minus post-erasure score) is the amount of clinical signal that shared directions with the nuisance. Near zero: encoded but not load-bearing. Clearly positive: the clinical signal was riding on acquisition geometry. A nuisance score that overshoots below chance is an INLP artifact (directions are fit on all rows), not evidence of anything.
