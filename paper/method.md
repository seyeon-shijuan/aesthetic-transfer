# 3. Methodology

## 3.1 Problem Definition and Notation

We study whether the aesthetic attribute axes of one visual medium retain their explanatory power in another. We denote the three domains by their **role**, not their dataset name, so that the framework extends to other datasets of the same medium: $\mathcal{D}_A$ (attribute source, AADB), $\mathcal{D}_P$ (photographic, AVA), and $\mathcal{D}_R$ (a**R**tistic, BAID).

Each domain provides images $x$ and a human aesthetic score $s(x) \in \mathbb{R}$. The attribute source additionally provides eleven aesthetic attribute labels $a(x)$ per image (e.g. color harmony, depth of field). A frozen vision encoder $\varphi \in \Phi = \{\text{DINOv2 [9]}, \text{CLIP [1]}, \text{SigLIP2 [2]}\}$ maps an image to a fixed feature vector $\varphi(x) \in \mathbb{R}^{768}$. No encoder is fine-tuned at any stage; only a linear probe is trained on top of the frozen features.

**Probe.** For a given attribute $a$ and backbone $\varphi$, we fit a linear ridge probe $g_a$ on the attribute-source training features to predict the attribute label:

```math
g_a \;=\; \arg\min_{w \in \mathbb{R}^{768}} \; \sum_{x \in \mathcal{D}_A^{\text{train}}} \big(w^\top \varphi(x) - a(x)\big)^2 \;+\; \lambda \, \|w\|_2^2 . \qquad (1)
```

The ridge penalty keeps the probe low-capacity by design. A linear probe [8] that reads an attribute well demonstrates that the attribute axis is *already present* in the frozen representation $\varphi$, rather than being synthesized by a high-capacity head. This interpretive contract — that the read reflects the representation, not the reader — is the reason every downstream design choice in this section is biased toward the *least* expressive option available: a linear (not kernel or MLP) probe, a single pooled feature (not a trained aggregation), and a fixed regularization. Any expressiveness we add to the probe is expressiveness we could no longer attribute to the encoder.

**Attribute explanatory power.** Given a fitted probe, we measure how well the *predicted* attribute explains the *human aesthetic score* in an arbitrary domain $\mathcal{D}$, using Spearman's rank correlation coefficient (SRCC):

```math
r_a(\mathcal{D}, \varphi) \;=\; \mathrm{SRCC}\Big( \big\{\, g_a(\varphi(x)) \,\big\}_{x \in \mathcal{D}},\; \big\{\, s(x) \,\big\}_{x \in \mathcal{D}} \Big) . \qquad (2)
```

We use rank correlation rather than $R^2$ so that the quantity is invariant to any monotonic rescaling between the probe output and the aesthetic score. This invariance is not a cosmetic choice: the probe is trained to predict an AADB attribute on a $[0,1]$-style scale, but is then correlated against aesthetic scores on entirely different scales in AVA and BAID. A rank statistic asks only whether the probe *orders* images the way human aesthetic preference does — which is the question of transfer, and sidesteps the incomparable score distributions of the two target datasets.

**Readability gate.** An attribute is admitted to the analysis only if its probe reads reliably on held-out attribute-source data, *unanimously* across all three backbones:

```math
\mathrm{gate}_a \;=\; \min_{\varphi \in \Phi} \, r_a\big(\mathcal{D}_A^{\text{test}}, \varphi\big), \qquad a \ \text{ is readable} \iff \mathrm{gate}_a \geq \tau . \qquad (3)
```

The threshold $\tau = 0.30$ is fixed in advance (Section 3.4). Two aspects of this gate are deliberate. First, taking the **minimum** over $\Phi$ — rather than the mean or the maximum — imposes unanimity: an attribute survives only if *every* backbone can read it, so that the transfer we later measure is never carried by a single lucky encoder. A mean gate would let one strong backbone rescue an attribute the others cannot see; the minimum refuses that. Second, unanimity fixes a *single common attribute support* across all three backbones, which is a precondition for the concordance analysis in Equation (5): the three encoders can only be compared on their ranking of a shared set of attributes. The gate is thus not merely a quality filter but the step that makes the cross-architecture comparison well defined.

**Transfer shift.** For each readable attribute we apply the *photo-trained* probe, without any refitting, to both media, and take the difference in explanatory power between the photographic and artistic domains:

```math
T_a \;=\; \bar r_a(\mathcal{D}_P) - \bar r_a(\mathcal{D}_R), \qquad \bar r_a(\mathcal{D}) \;=\; \frac{1}{|\Phi|} \sum_{\varphi \in \Phi} r_a(\mathcal{D}, \varphi) . \qquad (4)
```

A positive $T_a$ marks a **photo-specific** axis, whose aesthetic relevance does not survive transfer to art; a negative $T_a$ marks an axis that is *more* aesthetically relevant in art than in photography. The probe is *not* refitted on the target domain — the same photographic probe is carried unchanged into art. This is what isolates transfer from adaptation — we measure how a photographically-defined aesthetic axis behaves in art, not how well a fresh art-trained probe would perform. Refitting would answer a different, easier question and would destroy the diagnostic we are after.

**Representation invariance.** To test whether the ordering of shifts is a property of the domain pair rather than of any single encoder, we compute Kendall's coefficient of concordance $W$ over the rankings that the $m = |\Phi|$ backbones assign to the $n$ readable attributes by their shift $T_a$:

```math
W \;=\; \frac{12\,S}{m^2\,(n^3 - n)} , \qquad (5)
```

where $S$ is the sum of squared deviations of the per-attribute summed ranks from their mean. $W = 1$ denotes perfect agreement across backbones and $W = 0$ denotes none. We assess its significance by a permutation test (Section 4.4) rather than a parametric table, because $n$ is small and the parametric null for $W$ is only asymptotically valid.

## 3.2 Course-Theoretical Preliminaries and Design Rationale

This work instantiates four core modules of the course. From **Structural Intelligence**, our diagnostic instrument is the most elementary structural unit of all — a single linear (fully-connected) mapping, the ridge probe of Equation (1); the entire analysis rests on what this minimal linear layer can and cannot read from a fixed feature vector. From **Visual Intelligence**, we use the notion of a frozen visual representation evaluated by *linear probing*: a standard diagnostic in which that linear model is trained on top of fixed features to test what information the representation linearly exposes. The three encoders themselves span the module's development, from purely visual self-supervision (DINOv2) to **Multimodal Intelligence**, where image–text contrastive pretraining (CLIP, SigLIP2) aligns visual features with language. **Language Intelligence** enters not in the probe but at the evaluation layer: the strongest published scorers we benchmark against (Q-Align, ArtiMuse) read aesthetic quality through language-token prediction, so our comparison grid sets the frozen-probe diagnostic against the course's language module as well as its visual and multimodal ones.

The choice of these three specific backbones is itself part of the design. They are matched in capacity (all ViT-B, 768-d) but maximally separated in *pretraining signal*: DINOv2 sees only images, through self-distillation; CLIP and SigLIP2 see image–text pairs, but under different contrastive objectives (softmax versus sigmoid). If a transfer pattern held for only one of them it could be dismissed as an idiosyncrasy of that objective. By spanning a self-supervised encoder and two differently-trained vision–language encoders, we turn the choice of representation into a variable to be controlled rather than a confound to be hoped away — the concordance statistic $W$ then directly measures whether our conclusions survive that variation. Our central instrument, a ridge-regularized linear probe, is the simplest diagnostic consistent with this aim, chosen because its low capacity licenses the interpretation that a successful read reflects the representation and not the probe.

## 3.3 Algorithm

The full procedure is three stages, each corresponding to one equation group above.

```
Algorithm 1: Attribute transfer analysis
Input : frozen backbones Φ; datasets D_A, D_P, D_R; threshold τ
Output: readable set R; transfer shifts {T_a}; concordance W

# Stage 1 — feature extraction (once; cached)
for φ in Φ:
    for D in {D_A^train, D_A^test, D_P, D_R}:
        Z[φ, D] = φ(x) for x in D            # 768-d pooled features

# Stage 2 — gate on attribute source
R = {}
for a in attributes:
    for φ in Φ:
        g = ridge_fit(Z[φ, D_A^train], label_a)          # Eq. (1)
        gate[a, φ] = SRCC(g(Z[φ, D_A^test]), score)       # Eq. (2)
    if min_φ gate[a, φ] >= τ:                             # Eq. (3)
        R.add(a)

# Stage 3 — transfer readable probes to both media
for a in R:
    for φ in Φ:
        g = ridge_fit(Z[φ, D_A^train], label_a)
        r_P[a, φ] = SRCC(g(Z[φ, D_P]), score_P)           # Eq. (2)
        r_R[a, φ] = SRCC(g(Z[φ, D_R]), score_R)
    T[a] = mean_φ r_P[a, φ] - mean_φ r_R[a, φ]            # Eq. (4)
W = kendall_concordance({rank of T over Φ})               # Eq. (5)
```

Because every backbone is frozen, Stage 1 is run once and cached; Stages 2 and 3 operate entirely on the cached feature matrices and require no GPU. Fitting a probe is a closed-form ridge solution, so the entire analysis after extraction completes in seconds. This separation is also what makes the statistical layer of Section 4 affordable: because a probe costs nothing once features are cached, we can resample it thousands of times to obtain bootstrap confidence intervals for every reported quantity without re-touching the encoders.

## 3.4 Implementation Details

**Platform.** Feature extraction runs on a single NVIDIA A100 80GB GPU; all subsequent probing and correlation analysis run on CPU. The pipeline is implemented in Python with PyTorch and Hugging Face Transformers for the encoders, and scikit-learn / SciPy for ridge regression and rank statistics.

**Backbones.** DINOv2 ViT-B/14, CLIP ViT-B/16, and SigLIP2 ViT-B/16, all at base scale to control for model capacity. For each image we take the encoder's native pooled output (768-d for all three); each encoder uses its own native preprocessing, applied identically across the three domains. We use the pooled vector rather than patch tokens deliberately: pooling summarizes *what* is present while discarding *where* it sits, which both matches the appearance-based attributes we study and, as Section 4.2 shows, cleanly separates them from the spatial-compositional attributes a pooled read cannot recover.

**Probe.** Ridge regression with $\lambda = 10$, identical for the gate and the transfer stages so that a single fitted probe underlies both. We verified that the readable set is unchanged for $\lambda \in \{1, 10, 100\}$ (Section 4.7), indicating the result is not an artifact of regularization strength.

**Pre-registered choices.** The readability threshold $\tau = 0.30$, the unanimity rule (Eq. 3), the pooled-feature definition, the identical preprocessing across domains, and the AVA test subsample (5,000 images from the `generic_test` split, seed 42) were all fixed and version-controlled *before* any gate, transfer, or concordance value was computed. Pre-registration matters here more than in a conventional benchmark study: because the analysis has many defensible variants (threshold, aggregation, feature choice), fixing them in advance is what separates a genuine test from a search over analyses for a striking result. The AADB attribute matrix column order was verified by exhaustive numerical matching against the per-attribute list files rather than assumed — a check that caught a column-offset error early and is logged in Appendix A.

**Data.** $\mathcal{D}_A$: AADB, 8,458 / 1,000 train/test with eleven attributes; test uses the `TestNew` split. $\mathcal{D}_P$: AVA, 4,976 images (5,000 sampled; 24 no longer hosted). $\mathcal{D}_R$: BAID, 6,394 test artworks (49 unreachable from the original host, logged for reproducibility). AADB test filenames carry a score prefix that is stripped before joining features to labels.