# echo-shortcut-audit

**What does an ejection-fraction model's embedding know that it shouldn't, and what does it cost to make it forget?**

Deep echocardiography models are trained to read the heart, but they see the whole video: frame rate, frame geometry, clip length, and whatever else the scanner stamps on the signal. If those acquisition parameters correlate with the label, the model can lean on them. That shortcut will not transfer to the next scanner, the next hospital, or the next age group.

This repository audits a *frozen* echo embedding for exactly that, using only the embedding vectors and the dataset's metadata table. No videos, no GPU.

1. **Probe** — cross-validated linear probes for clinical targets (EF, ESV, EDV) and acquisition nuisances (FPS, frame height/width, number of frames), each against a permutation null and a random-matrix control.
2. **Baseline** — how much EF is predictable from acquisition metadata *alone*, and how much the embedding adds beyond it (partial R²).
3. **Stratify** — is the embedding's EF error uneven across frame-rate / clip-length bins? (Kruskal–Wallis)
4. **Erase and price** — iterative null-space projection ([INLP](https://aclanthology.org/2020.acl-main.647/)) removes each nuisance until it is no longer linearly decodable, then every clinical target is re-probed. The drop is the price of the shortcut.

The default subject is the 400-d R3D embedding of all 10,030 EchoNet-Dynamic videos released in [Team-Echo-MIT/r3d-v0-embeddings](https://github.com/Team-Echo-MIT/r3d-v0-embeddings) (Chung, Lee, Kaker, Zhao; *JACC: Advances* 2024 / IEEE I2MTC 2024). Any other embedding of the same videos — EchoCLIP, EchoPrime, PanEcho, EchoFM, EchoJEPA — can be dropped in for a side-by-side table.

## Why this is new

Demographic leakage in echo models has been studied (Duffy et al. 2022 for end-to-end models; Tohyama et al. 2025 probing EchoPrime for sex and race). Scanner / site fingerprints in frozen foundation-model embeddings have been shown in brain MRI, fMRI, hematology, and fetal ultrasound. Nobody has yet probed an EF-trained echo video embedding for **acquisition parameters** or measured how much EF accuracy depends on them. `docs/related_work.md` has the full search with citations.

## Install

```bash
git clone https://github.com/mindyslee/echo-shortcut-audit
cd echo-shortcut-audit
pip install -e ".[dev]"
```

Python ≥ 3.10; numpy, pandas, scikit-learn, scipy, matplotlib.

## Data

| file | where | needed for |
|---|---|---|
| `echonet_embeddings.txt` | public, [r3d-v0-embeddings](https://github.com/Team-Echo-MIT/r3d-v0-embeddings) (78 MB) | the embedding |
| `FileList.csv` | EchoNet-Dynamic, [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a) (free non-commercial DUA) | EF, ESV, EDV, FPS, FrameHeight, FrameWidth, NumberOfFrames |

`FileList.csv` is not redistributed here. The embeddings are derived data released by the original authors under MIT.

## Run

```bash
# the real audit
echoaudit run --embeddings echonet_embeddings.txt --filelist FileList.csv --out results/

# quick pass: 3,000 videos, 5 permutations
echoaudit run --embeddings echonet_embeddings.txt --filelist FileList.csv --out results_quick/ --max-rows 3000 --permutations 5

# no DUA yet? validate the pipeline on the real embeddings with SYNTHETIC labels
# that carry a planted, known-strength nuisance leak
echoaudit demo --embeddings echonet_embeddings.txt --out results_demo/ --leak 0.8

# or with nothing at all
echoaudit demo --out results_demo/
```

Outputs: `audit.json` (everything), `audit.md` (tables), `decodability.png`, `cost_curves.png`.

A full run (10,030 × 400, 20 permutations, 5-fold) takes roughly 10–20 minutes on a laptop; the permutation null dominates. `--permutations 0` skips it; the random-matrix control and the z-score gate still run.

## Reading the report

**Decodability table.** For each target, the cross-validated score (R² or balanced accuracy), chance level, the same probe on a Gaussian random matrix of identical shape (should sit at chance), and the permutation null. A nuisance that is decodable far above its null is *encoded*. Encoded is not the same as *used* — that is what the next two sections settle.

**Metadata-only baseline.** If FPS and clip length alone predict EF with meaningful R², the dataset itself offers the shortcut; a model does not even need to look at the ventricle to get some of the way there.

**Stratified error.** If EF error differs across FPS bins or clip-length quartiles, the model is not equally reliable across acquisition settings, and that is what will show up at the next site.

**Price of erasure.** INLP removes one linear direction per class per iteration until the nuisance sits at chance. Each clinical target is re-probed after every step:

- EF holds, nuisance falls → the nuisance was encoded but not load-bearing. Reassuring.
- EF falls with the nuisance → the EF-relevant subspace overlaps the acquisition subspace. The embedding is partly reading the scanner, not the heart.

The synthetic demo plants EF and the nuisance along *independent* random directions, so a correct audit shows a large nuisance drop with near-zero EF cost; it is the pipeline's own sanity check.

## Plugging in other embeddings

Any file keyed by EchoNet video id works: `.txt` (`<id> v1 … vd` per line), `.npz` with arrays `ids` and `X`, or `.csv`/`.parquet` with the id in the first column. Generate them from the released weights of EchoCLIP, EchoPrime, PanEcho, EchoFM or EchoJEPA, point `--embeddings` at each, and compare `audit.json` files. `--pca 64` probes a reduced embedding if you want to control for dimensionality.

Other datasets map onto the same columns: rename to `FileName`, `EF`, `FPS`, `NumberOfFrames`, … (anything absent or constant is skipped) and pass `--nuisance` / `--clinical` to override the target lists. EchoNet-Pediatric adds `Age`, `Sex`, `Weight`, `Height`, which the same machinery can treat as nuisances or targets.

## Caveats, stated up front

- EchoNet-Dynamic is single-site, single-vendor, 112 × 112, so frame geometry is constant and vendor/probe labels do not exist. Frame rate and clip length are the only acquisition fields it offers; a multi-vendor audit needs another dataset.
- Linear probes bound *linear* decodability. A nuisance that survives INLP could still be read non-linearly; an MLP probe is the obvious extension.
- "Price of erasure" is measured with a linear EF probe on the projected embedding, not by retraining the video model. It bounds what a linear head would lose, which is the relevant quantity for the frozen-embedding use case.
- INLP removes directions greedily and can over-remove when nuisance and clinical subspaces overlap; the per-iteration curve, not just the endpoint, is the thing to look at.

## Roadmap

- [ ] Run on the real `FileList.csv`; publish `results/` as the first finding.
- [ ] Add EchoCLIP / EchoPrime / PanEcho / EchoJEPA embeddings of the same 10,030 videos → cross-model shortcut table.
- [ ] Non-linear (MLP) probes and concept-erasure baselines (LEACE).
- [ ] EchoNet-Pediatric: do the acquisition directions found on adults predict where pediatric EF error concentrates?
- [ ] Write-up.

## License

MIT. The embeddings and the EchoNet-Dynamic data keep their own licenses.
