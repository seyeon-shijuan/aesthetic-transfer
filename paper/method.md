# 5. Methodology

## 5.1 Problem Definition and Notation

We study whether the aesthetic attribute axes of one visual medium retain their explanatory power in another. We denote the three domains by their **role**, not their dataset name, so that the framework extends to other datasets of the same medium: $\mathcal{D}_A$ (attribute source, AADB), $\mathcal{D}_P$ (photographic, AVA), and $\mathcal{D}_R$ (a**R**tistic, BAID).

Each domain provides images $x$ and a human aesthetic score $s(x) \in \mathbb{R}$. The attribute source additionally provides eleven aesthetic attribute labels $a(x)$ per image (e.g. colour harmony, depth of field). A frozen vision encoder $\varphi \in \Phi = \{\text{DINOv2}, \text{CLIP}, \text{SigLIP2}\}$ maps an image to a fixed feature vector $\varphi(x) \in \mathbb{R}^{768}$. No encoder is fine-tuned at any stage; only a linear probe is trained on top of the frozen features.

**Probe.** For a given attribute $a$ and backbone $\varphi$, we fit a linear ridge probe $g_a$ on the attribute-source training features to predict the attribute label:

$$
g_a \;=\; \arg\min_{w \in \mathbb{R}^{768}} \; \sum_{x \in \mathcal{D}_A^{\text{train}}} \big(w^\top \varphi(x) - a(x)\big)^2 \;+\; \lambda \, \|w\|_2^2 . \tag{1}
$$

The ridge penalty keeps the probe low-capacity by design. A linear probe that reads an attribute well demonstrates that the attribute axis is *already present* in the frozen representation $\varphi$, rather than being synthesized by a high-capacity head.

**Attribute explanatory power.** Given a fitted probe, we measure how well the *predicted* attribute explains the *human aesthetic score* in an arbitrary domain $\mathcal{D}$, using Spearman's rank correlation coefficient (SRCC):

$$
r_a(\mathcal{D}, \varphi) \;=\; \operatorname{SRCC}\Big( \big\{\, g_a(\varphi(x)) \,\big\}_{x \in \mathcal{D}},\; \big\{\, s(x) \,\big\}_{x \in \mathcal{D}} \Big) . \tag{2}
$$

We use rank correlation rather than $R^2$ so that the quantity is invariant to any monotonic rescaling between the probe output and the aesthetic score, which is essential when comparing across two datasets whose score distributions differ.

**Readability gate.** An attribute is admitted to the analysis only if its probe reads reliably on held-out attribute-source data, *unanimously* across all three backbones:

$$
\operatorname{gate}_a \;=\; \min_{\varphi \in \Phi} \, r_a\big(\mathcal{D}_A^{\text{test}}, \varphi\big),
\qquad a \ \text{is \emph{readable}} \iff \operatorname{gate}_a \geq \tau . \tag{3}
$$

The threshold $\tau = 0.30$ is fixed in advance (Section 5.4). The minimum over $\Phi$ enforces unanimity: an attribute survives only if every backbone can read it, which fixes a single common attribute support required by the concordance analysis in Equation (5).

**Transfer shift.** For each readable attribute we apply the *photo-trained* probe, without any refitting, to both media, and take the difference in explanatory power between the photographic and artistic domains:

$$
T_a \;=\; \bar r_a(\mathcal{D}_P) - \bar r_a(\mathcal{D}_R),
\qquad
\bar r_a(\mathcal{D}) \;=\; \frac{1}{|\Phi|} \sum_{\varphi \in \Phi} r_a(\mathcal{D}, \varphi) . \tag{4}
$$

A positive $T_a$ marks a **photo-specific** axis, whose aesthetic relevance does not survive transfer to art; a negative $T_a$ marks an axis that is *more* aesthetically relevant in art than in photography.

**Representation invariance.** To test whether the ordering of shifts is a property of the domain pair rather than of any single encoder, we compute Kendall's coefficient of concordance $W$ over the rankings that the $m = |\Phi|$ backbones assign to the $n$ readable attributes by their shift $T_a$:

$$
W \;=\; \frac{12\,S}{m^2\,(n^3 - n)} , \tag{5}
$$

where $S$ is the sum of squared deviations of the per-attribute summed ranks from their mean. $W = 1$ denotes perfect agreement across backbones and $W = 0$ denotes none.

## 5.2 Course-Theoretical Preliminaries

This work instantiates two core modules of the course. From **Visual Intelligence**, we use the notion of a frozen visual representation evaluated by *linear probing*: a standard diagnostic in which a linear model is trained on top of fixed features to test what information the representation linearly exposes. The three encoders themselves span the module's development, from purely visual self-supervision (DINOv2) to **Multimodal Intelligence**, where image–text contrastive pretraining (CLIP, SigLIP2) aligns visual features with language. Our central instrument — a ridge-regularized linear probe — is the simplest such diagnostic, chosen precisely because its low capacity licenses the interpretation that a successful read reflects the representation and not the probe.

## 5.3 Algorithm

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

Because every backbone is frozen, Stage 1 is run once and cached; Stages 2 and 3 operate entirely on the cached feature matrices and require no GPU. Fitting a probe is a closed-form ridge solution, so the entire analysis after extraction completes in seconds.

## 5.4 Implementation Details

**Platform.** Feature extraction runs on a single NVIDIA A100 80GB GPU; all subsequent probing and correlation analysis run on CPU. The pipeline is implemented in Python with PyTorch and Hugging Face Transformers for the encoders, and scikit-learn / SciPy for ridge regression and rank statistics.

**Backbones.** DINOv2 ViT-B/14, CLIP ViT-B/16, and SigLIP2 ViT-B/16, all at base scale to control for model capacity. For each image we take the encoder's native pooled output (768-d for all three); each encoder uses its own native preprocessing, applied identically across the three domains.

**Probe.** Ridge regression with $\lambda = 10$, identical for the gate and the transfer stages so that a single fitted probe underlies both. We verified that the readable set is unchanged for $\lambda \in \{1, 10\}$, indicating the result is not an artifact of regularization strength.

**Pre-registered choices.** The readability threshold $\tau = 0.30$, the unanimity rule (Eq. 3), the pooled-feature definition, the identical preprocessing across domains, and the AVA test subsample (5,000 images from the `generic_test` split, seed 42) were all fixed and version-controlled *before* any gate, transfer, or concordance value was computed. The AADB attribute matrix column order was verified by exhaustive numerical matching against the per-attribute list files rather than assumed.

**Data.** $\mathcal{D}_A$: AADB, 8,458 / 1,000 train/test with eleven attributes; test uses the `TestNew` split. $\mathcal{D}_P$: AVA, 4,976 images (5,000 sampled; 24 no longer hosted). $\mathcal{D}_R$: BAID, 6,394 test artworks (49 of 6,400 unreachable from the original host, logged for reproducibility). AADB test filenames carry a score prefix that is stripped before joining features to labels.
```
