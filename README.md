# Attribute-Level Analysis of Cross-Domain Aesthetic Transfer from Photography to Art

Which components of **photographic** aesthetics actually transfer to **art** — and which do not?
Instead of comparing whole models with a single opaque score, we probe *individual aesthetic
attributes* out of frozen visual representations and transport each one, unchanged, across the
photography → art boundary. This turns the vague claim "art is different" into a measurable,
per-attribute diagnosis.

Per-section Markdown sources are in [`paper/`](paper/) (full assembled paper:
[`paper/paper_full.md`](paper/paper_full.md)).

---

## TL;DR

- We train **linear ridge probes** on **frozen** features from three architecturally distinct
  encoders (**DINOv2**, **CLIP**, **SigLIP2**) to read eleven photographic aesthetic attributes.
- **7 of 11** attributes are linearly readable (pass a pre-registered unanimity gate). The **4**
  that fail are exactly the **spatial/compositional** ones a global-pooled vector cannot expose.
- Transporting each readable probe from photographs (AVA) to art (BAID):
  - **5 attributes** — dominated by the **color & tone** family — lose their aesthetic relevance in
    art, several reversing sign.
  - **2 attributes** — **depth of field** and **object emphasis** — *reverse* and become **more**
    relevant in art.
- The three encoders agree on this ordering almost perfectly (**Kendall's W = 0.849**, unanimous
  sign agreement) → the pattern is a property of the *domain pair*, not of any single backbone.
- As a system, the readable attributes cover ~**45%** of predictable aesthetic signal in
  photography but essentially **none (3%)** in art.

---

## Key results

### Attribute readability gate (AADB test, τ = 0.30, unanimous)

Readable: **Object, Content, VividColor, DoF, ColorHarmony, Light, Repetition**
Unreadable (spatial/compositional): RuleOfThirds, Symmetry, BalancingElements, MotionBlur

### Transfer shift photography → art

`T_a = r_a(photo) − r_a(art)`; positive = photo-specific, negative = stronger in art. All shifts
significant (p < 0.001, 2000-resample bootstrap).

![Per-attribute transfer shift](results/fig1_shift.png)

The pattern is near-identical across all three backbones — the visual counterpart of W = 0.849:

![Transfer shift per backbone](results/fig2_heatmap.png)

### The specialization ladder (BAID test, cross-domain)

| Model | Type | Photo (AVA) | **Art (BAID)** |
|---|---|---|---|
| Aesthetic-Predictor-V2.5 | photo-specialist | 0.473 | 0.009 |
| Q-Align | photo-specialist | 0.817 | 0.126 |
| **ArtiMuse** | **multi-domain** | 0.532 | **0.199** |
| SAAN | art-specialist | — | **0.472** |

photo-specialist **<** multi-domain **<** art-specialist. Multi-domain training partially recovers
artistic transfer, but does not substitute for domain-specific adaptation.

![Aesthetic competence on art by specialization](results/fig3_grid.png)

---

## Repository structure

```
.
├── README.md
├── paper/                       # the paper, one Markdown file per section
│   ├── paper_full.md            # full assembled paper (Markdown)
│   ├── abstract_conclusion.md
│   ├── introduction.md
│   ├── related_work.md
│   ├── method.md                # probing, gating, transfer-shift, concordance (Eq. 1–5)
│   ├── results.md
│   ├── discussion.md
│   ├── references.md
│   ├── appendix.md              # data-exclusion & reproducibility log
│   └── freeze_2026-07-18.md     # PRE-REGISTRATION (binding, signed before any result)
├── results/                     # raw outputs, tables, and figures behind every number
│   ├── fig1_shift.{png,pdf}     # per-attribute transfer shift
│   ├── fig2_heatmap.{png,pdf}   # shift per backbone (rank agreement)
│   ├── fig3_grid.{png,pdf}      # specialization ladder
│   ├── gate_table.txt           # gate SRCC per attribute × backbone (Table 1)
│   ├── transfer.txt, table2.txt # transfer shifts (Table 2)
│   ├── kendall_w.txt            # concordance W + permutation test
│   ├── coverage.txt             # R²_attr / R²_full coverage (Table 4)
│   ├── ablation.txt             # readable-set robustness to λ and τ (Table 5)
│   ├── model_grid.txt           # cross-domain scorer grid (Table 3)
│   ├── stats.txt                # bootstrap CIs / sign tests
│   ├── aes_scores.txt, qalign_scores.txt, artimuse_scores*.txt
│   └── log_{tr,te,}_{dino,clip,siglip}.txt   # extraction / probe logs
└── src/                         # analysis code
    ├── data.py                  # dataset loading, splits, label matrix
    ├── extract.py               # Stage 1 — cache 768-d pooled features per backbone × domain
    ├── probe.py                 # ridge probe fit + readability gate (Stage 2)
    ├── transfer.py              # Stage 3 — apply photo-probe to both media, transfer shift
    ├── metrics.py               # SRCC / R² helpers
    ├── coverage.py              # coverage ratio R²_attr / R²_full
    ├── ablation.py              # λ / τ robustness sweep
    ├── stats.py                 # bootstrap CI, sign test, Kendall-W permutation
    ├── score_aes.py             # Aesthetic-Predictor-V2.5 scoring
    ├── score_qalign.py          # Q-Align scoring
    ├── score_artimuse.py        # ArtiMuse scoring
    └── fig1_shift.py, fig2_heatmap.py, fig3_grid.py   # figures
```

## Code / pipeline

The frozen backbones are extracted once and cached; everything after that runs on CPU in seconds.
Each script maps to a stage of the method (paths and configs are set inside the scripts):

| Stage | Script(s) | Output |
|---|---|---|
| 1. Feature extraction | `extract.py` (+ `data.py`) | cached 768-d features, `log_{tr,te}_*.txt` |
| 2. Probe + readability gate | `probe.py`, `metrics.py` | `gate_table.txt` (Table 1) |
| 3. Transfer shift | `transfer.py` | `transfer.txt`, `table2.txt` (Table 2) |
| Rank agreement / stats | `stats.py` | `kendall_w.txt`, `stats.txt` |
| Coverage | `coverage.py` | `coverage.txt` (Table 4) |
| Robustness ablation | `ablation.py` | `ablation.txt` (Table 5) |
| Model grid | `score_aes.py`, `score_qalign.py`, `score_artimuse.py` | `*_scores.txt`, `model_grid.txt` (Table 3) |
| Figures | `fig1_shift.py`, `fig2_heatmap.py`, `fig3_grid.py` | `results/fig{1,2,3}_*` |

## Data & models

| Role | Dataset / Model | Notes |
|---|---|---|
| Attribute source `D_A` | **AADB** | 8,458 / 500 / 1,000; eleven aesthetic attributes |
| Photographic domain `D_P` | **AVA** | 4,976 images (5,000 sampled, seed 42) |
| Artistic domain `D_R` | **BAID** | 6,394 test artworks; CC BY-NC-ND 4.0 (non-commercial research) |
| Backbones (frozen) | DINOv2 ViT-B/14 · CLIP ViT-B/16 · SigLIP2 ViT-B/16 | matched capacity, different pretraining |
| Scorers (grid) | Q-Align · Aesthetic-Predictor-V2.5 · ArtiMuse · SAAN | photo / photo / multi-domain / art |

## Reproducibility & pre-registration

Every analysis decision — the readability threshold (τ = 0.30), the unanimity rule, the ridge
strength (λ = 10), the pooled-feature definition, and the AVA sampling seed — was **fixed and
version-controlled before any gate, correlation, or gap value was computed**. The binding record is
[`paper/freeze_2026-07-18.md`](paper/freeze_2026-07-18.md); every excluded sample and processing
decision is logged in [`paper/appendix.md`](paper/appendix.md). The four unreadable attributes are
reported as a **result**, not silently dropped, and the readable set is shown to be robust to λ and
τ (see `results/ablation.txt`).

## License

The BAID artworks are distributed under **CC BY-NC-ND 4.0** and are used here strictly for
non-commercial academic research; please observe the original dataset licenses for AADB, AVA, and
BAID. <!-- TODO: add a license for the code in this repository (e.g. MIT). -->