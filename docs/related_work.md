# Related work and the gap

Literature search run 19 October 2026 across arXiv, PubMed/PMC, medRxiv, MICCAI/MIDL proceedings, Semantic Scholar and GitHub, roughly seventy distinct queries over three directions. Short summaries in my own words; follow the links for the papers.

## The question

Does a frozen echocardiography embedding trained for ejection fraction encode acquisition parameters (frame rate, frame geometry, clip length, vendor, probe), and does its EF signal depend on them?

## Closest prior work

### Echocardiography

| work | what it did | what it leaves open |
|---|---|---|
| Duffy et al., *npj Digit Med* 2022 — [Confounders mediate AI prediction of demographics in medical imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC9780355/) | End-to-end R(2+1)D models predict age and sex from echo well; race poorly and in a way that tracks dataset confounding (Cedars-Sinai ↔ Stanford). | Demographics only; no acquisition metadata; no frozen-embedding probing; no test of whether EF relies on the encoded attribute. |
| Tohyama et al., medRxiv 2025 — [Multi-View Echocardiographic Embedding for Accessible AI Development](https://pmc.ncbi.nlm.nih.gov/articles/PMC12393585) | Probes frozen EchoPrime embeddings for sex and race on MIMIC-IV-ECHO; gradient-reversal removal of sex costs clinical performance, race is not removable. | The nearest "probe an echo embedding for a nuisance" paper, but sex/race only; not FPS/size/frame count; not EF on EchoNet-Dynamic. |
| Elyasi et al., arXiv 2026 — [Domain Shift in Echocardiography](https://arxiv.org/abs/2607.19643) | Dataset identity (~0.95) and vendor (~0.82–0.90) predictable from handcrafted features across six echo datasets; shift scores predict segmentation Dice drop. | Segmentation, not EF; handcrafted and VAE/U-Net features, not an EF-trained video embedding. |
| Taratynova et al., arXiv 2025 — [CardioBench](https://arxiv.org/abs/2510.00520) | Linear-probe benchmark of echo foundation models; EF error stratified by sex, age, BMI, image quality; notes dataset-specific artifacts on CAMUS. | No metadata probing, no erasure. |
| Pandey, medRxiv 2026 — [Ejection fraction on a budget](https://www.medrxiv.org/content/10.64898/2026.09.02.26362055v1) | Clip length × sampling period grid on EchoNet-Dynamic; sparse sampling beats dense. | Treats clip length as a design knob, not a confound; no metadata-only baseline. |
| Lin et al., arXiv 2024 — [Shortcut Learning in Medical Image Segmentation](https://arxiv.org/abs/2403.06748) | Calipers and zero-padding act as segmentation shortcuts in ultrasound. | Pixel-level shortcuts in segmentation, not embedding-level in EF regression. |
| Munim et al., arXiv 2026 — [EchoJEPA](https://arxiv.org/abs/2602.02603) | Robustness to simulated attenuation/shadow; pretraining FPS changed 4→24 without analysis. | No nuisance probing or vendor stratification. |

### The probing template in other modalities

| work | contribution |
|---|---|
| Gichoya et al., *Lancet Digit Health* 2022 — [AI recognition of patient race in medical imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC9650160) | Landmark: race decodable from CXR/CT/mammography even after heavy degradation. |
| Glocker et al., *eBioMedicine* 2023 — [Algorithmic encoding of protected characteristics](https://pmc.ncbi.nlm.nih.gov/articles/PMC10025760/); *Radiology: AI* 2023 — [Risk of bias in CXR foundation models](https://arxiv.org/abs/2209.02965) | Subgroup separability in feature space tied to performance gaps. |
| Wong et al., arXiv 2025/MIDL 2026 — [Weight Space Correlation Analysis](https://arxiv.org/abs/2512.13144) | Fetal/cervical ultrasound: probes frozen embeddings for scanner model, pixel spacing, hospital, year; then tests whether the task head *uses* the same directions. Closest methodological neighbour, different ultrasound domain, still images. |
| Rahbar et al., arXiv 2026 — Frozen brain-MRI foundation models are site fingerprints | Site decodable (~0.9) from frozen embeddings at every depth; null-space / ComBat removal and its cost to downstream Dice. |
| Tao et al., ICML 2026 workshop — [Batch effects in brain foundation model embeddings](https://arxiv.org/abs/2604.14441) | fMRI: site probe ≫ diagnosis probe; site shortcut inflates pooled accuracy. |
| Sharma et al., MICCAI 2026 workshop — [Frozen hematology foundation models under acquisition shift](https://arxiv.org/pdf/2608.25148) | Scanner identity decodable at 0.997 from fifteen frozen encoders; projects out between-domain directions and retests. |
| Rashtchian et al., arXiv 2023 — [Substance or Style](https://arxiv.org/abs/2307.05610) | General-vision version of "what non-semantic attributes does the embedding know". |
| Ravfogel et al., ACL 2020 — [Null It Out (INLP)](https://aclanthology.org/2020.acl-main.647/) | The erasure method used here. |

### Adjacent directions checked and set aside

- **Conformal / selective EF prediction.** Partially occupied: Kazemi Esfeh et al. (MICCAI 2020, 2022) plot EF error vs. fraction rejected on EchoNet-Dynamic; an under-review 2026 repository ([echo-ef-reliable-dl](https://github.com/Barbodyad/echo-ef-reliable-dl)) does conformalized quantile regression plus risk–coverage tables on EchoNet-Dynamic. What remains (guideline-bin prediction sets on frozen embeddings) is a thinner wedge.
- **Cross-dataset embedding drift.** Bordered by Elyasi 2026 (segmentation) and by the CardioBench / EchoJEPA / EchoNet-Peds external-validation results. A natural second phase for this repository (EchoNet-Pediatric is public under the same AIMI terms), not a first contribution.

## What is open

1. No one has probed an EF-trained echo video embedding for FPS, frame dimensions, or clip length.
2. No one has built a metadata-only EF baseline on EchoNet-Dynamic, stratified EF error by acquisition fields, or erased those fields from an embedding and re-measured EF.
3. No one has compared echo foundation models on *how much acquisition information their embeddings carry* and *how much their EF signal depends on it* — the released embeddings of the same 10,030 videos make that a cheap table to build.

## Honest limits of the gap

- Demographic probing of echo embeddings exists (Tohyama 2025); the novel axis here is acquisition, and the write-up should say so.
- EchoNet-Dynamic is single-site and has no vendor/probe labels; FPS and clip length are the only acquisition fields. A multi-vendor claim needs another dataset.
- The probing-then-erasure recipe is established in other modalities; the contribution is the application and the finding, not the method.
