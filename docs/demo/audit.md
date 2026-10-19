# Shortcut audit: `echonet_embeddings.txt + synthetic labels`

n = 10030 videos, d = 400 dims. 1098.8 s.
- note: labels are SYNTHETIC; numbers here validate the pipeline, not the model

## Decodability (cross-validated linear probe)

| target | role | kind | score | chance | random-matrix control | permutation null | z | p | decodable |
|---|---|---|---|---|---|---|---|---|---|
| EF | clinical | regression | **0.683** ± 0.01 | 0.00 | -0.012 | -0.005 ± 0.001 | 688.9 | 0.091 | yes |
| ESV | clinical | regression | **0.428** ± 0.01 | 0.00 | -0.011 | -0.005 ± 0.001 | 464.1 | 0.091 | yes |
| EDV | clinical | regression | **0.068** ± 0.01 | 0.00 | -0.009 | -0.006 ± 0.001 | 71.7 | 0.091 | yes |
| FPS | nuisance | classification | **0.631** ± 0.01 | 0.33 | 0.335 | 0.334 ± 0.005 | 56.6 | 0.091 | yes |
| NumberOfFrames | nuisance | regression | **0.575** ± 0.01 | 0.00 | -0.013 | -0.005 ± 0.001 | 421.0 | 0.091 | yes |

Regression score is R²; classification score is balanced accuracy (chance = 1/k). The permutation p-value is floored at 1/(10+1); z is (score − null mean)/null sd.

## Metadata-only baseline for EF

- linear R² from acquisition metadata alone: **0.008** (p = 0.048)
- boosted-trees R² from metadata alone: **0.003**
- embedding R² = 0.683; embedding + metadata R² = 0.683
- embedding beyond metadata: **0.675**; metadata beyond embedding: **-0.000**

## EF error by acquisition stratum

| stratum | bins (n, MAE) | MAE range | Kruskal-Wallis p |
|---|---|---|---|
| FPS | 30: 3533, 4.81; 50: 3552, 4.89; 60: 2945, 4.82 | 0.08 | 0.427 |
| NumberOfFrames | (120.0, 166.0]: 2525, 4.79; (166.0, 444.0]: 2474, 4.90; (23.999, 73.0]: 2513, 4.80; (73.0, 120.0]: 2518, 4.88 | 0.11 | 0.677 |

## Price of erasing each nuisance (INLP)

| nuisance | decodability before → after | rank removed | EF: before → after (random ctrl) → excess price | ESV: before → after (random ctrl) → excess price | EDV: before → after (random ctrl) → excess price |
|---|---|---|---|---|---|
| FPS | 0.623 → 0.177 (chance 0.33) | 6 | 0.676 → 0.634 (0.661 ± 0.006) → **+0.027** | 0.413 → 0.386 (0.403 ± 0.005) → **+0.018** | 0.045 → 0.038 (0.043 ± 0.001) → **+0.005** |
| NumberOfFrames | 0.565 → -0.042 (chance 0.00) | 2 | 0.676 → 0.660 (0.670 ± 0.003) → **+0.011** | 0.413 → 0.403 (0.410 ± 0.002) → **+0.007** | 0.045 → 0.043 (0.044 ± 0.000) → **+0.002** |

Erasure runs in a PCA-whitened copy of the embedding until the nuisance is no longer linearly decodable. Every projection costs a little, so the same number of random directions is removed as a control; the **excess price** (random-control score minus post-erasure score) is the amount of clinical signal that shared directions with the nuisance. Near zero: encoded but not load-bearing. Clearly positive: the clinical signal was riding on acquisition geometry. A nuisance score that overshoots below chance is an INLP artifact (directions are fit on all rows), not evidence of anything.
