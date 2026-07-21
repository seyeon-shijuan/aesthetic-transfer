# Abstract

Image aesthetic assessment has developed almost entirely on photographs, yet artworks are judged by criteria that need not coincide with those of photography. We ask, attribute by attribute, which components of photographic aesthetics transfer to art and which do not. Using linear probes trained on frozen visual representations from three architecturally distinct encoders (DINOv2, CLIP, SigLIP2), we first identify which of eleven photographic aesthetic attributes are linearly readable from each representation: seven pass a pre-registered unanimity gate, and the four that fail are the spatial-compositional attributes that global pooled features discard. We then transport each readable probe, unchanged, from photographs (AVA) to art (BAID). Five attributes, dominated by the color and tone family, lose their aesthetic relevance in art; two — depth of field and object emphasis — reverse and become *more* relevant in art. The three encoders agree closely on this ordering (Kendall's $W = 0.849$, with unanimous sign agreement), indicating the pattern is a property of the domain pair rather than of any single representation. At the system level, the readable attributes account for roughly 45% of predictable aesthetic signal in photography but essentially none (3%) in art. An evaluation grid of specialized scorers shows the same gap: photographic specialists < a multi-domain model < an art specialist, so multi-domain training partially recovers artistic transfer but does not substitute for domain-specific adaptation. The result is a per-attribute diagnosis of photographic-to-artistic transfer, and a concrete specification of what any domain-adaptation method for art must correct.

**Keywords:** Visual Intelligence · Multimodal Intelligence · Language Intelligence · Structural Intelligence · Image Aesthetic Assessment

---

# 1. Introduction

## 1.1 Research Background and Problem Statement

Image aesthetic assessment sits at the junction of visual and multimodal representation. It maps directly onto the course's progression of intelligent modules — from **Structural Intelligence** (convolutional and recurrent networks), through **Visual Intelligence** (recognition and dense prediction over images), to **Multimodal Intelligence** (vision–language representation learning) — inheriting its backbones from the structural and visual modules (CNNs and vision transformers) and, in its most recent form, from the multimodal module (CLIP-style image–text encoders). Situating our study in this progression clarifies what it does and does not touch: we work entirely with *frozen* visual and vision–language representations and ask what aesthetic information they already expose, rather than modifying the encoders themselves.

Image aesthetic assessment — predicting how visually pleasing an image is — has accordingly become a central application of the **Visual Intelligence** module, and increasingly draws on the representations learned by **Multimodal Intelligence** models such as CLIP [1] and SigLIP [2]. The dominant benchmark, AVA [3], consists almost entirely of *photographs*, and the field's progress has been measured largely on photographic data. As a result, both the models and our understanding of "what makes an image good" are implicitly photographic.

Yet aesthetic judgment is not confined to photography. Paintings, illustrations, and digital artworks are judged by criteria that need not coincide with those of photography [4, 5]: a deliberately muted palette or an unbalanced composition may be a strength in a painting and a flaw in a photograph. This raises a question that the photograph-centric literature has largely left implicit: **which components of photographic aesthetics actually transfer to art, and which do not?** Answering this matters both scientifically — it tells us whether "aesthetic quality" is one construct or several — and practically, because any model or domain-adaptation method that hopes to work on art must know what it is adapting.

Prior work has approached cross-domain aesthetics mostly at the level of *whole-model performance*: train a scorer on one dataset, test its correlation on another, and report that it drops [6, 7]. Such end-to-end comparisons establish *that* transfer degrades, but not *why*. They cannot say which aesthetic factor failed to carry over, because the score is a single opaque number. What is missing is a **decomposition**: a way to ask, factor by factor, whether a specific aesthetic attribute — color harmony, depth of field, rule of thirds — retains its relationship to human preference when the medium changes.

We address exactly this gap. Rather than comparing whole models, we probe *individual aesthetic attributes* out of frozen visual representations and track each one across media. Because the probes are linear [8] and the backbones are frozen, a probe that succeeds tells us the attribute axis genuinely exists in the representation; carrying that same probe from photography to art then isolates the transfer behavior of *that attribute alone*. This turns a vague claim ("art is different") into a measurable, per-attribute quantity.

## 1.2 Research Objectives and Core Contributions

Building on the course's Visual and Multimodal Intelligence foundations, this paper makes three contributions:

1. **A per-attribute transfer diagnostic.** We introduce a simple, pre-registered protocol that (i) gates aesthetic attributes by whether they are *linearly readable* from a frozen representation, and (ii) measures each readable attribute's *transfer shift* between photographic and artistic domains. The protocol is deliberately low-capacity: the probe is a single linear (fully-connected) mapping — the most basic structural unit from the course's **Structural Intelligence** module — applied to frozen features, so that its readings reflect the representation rather than a trained head.

2. **Empirical evidence on what transfers, compared systematically across architectures.** We compare three architecturally distinct encoders -- a self-supervised one (DINOv2 [9]) and two vision-language ones (CLIP [1], SigLIP2 [2]) -- on the same transfer diagnostic. We find that most readable photographic attributes — dominated by the *color and tone* family — lose their aesthetic relevance in art, several even reversing sign, while a minority (depth of field, object emphasis) become *more* relevant in art. The three encoders, though trained by different objectives, still agree on this ordering (Kendall's $W = 0.849$), indicating the pattern is a property of the domain pair rather than of any one architecture — a conclusion only a cross-architecture comparison can support.

3. **A foundation for domain-adaptive aesthetic modeling.** The diagnostic pinpoints *what* a future art-aesthetics adaptation method must correct (the non-transferring color axes), *what* it must preserve (the axes that are already stronger in art), and *how* its success should be measured (the same transfer statistic, before vs. after adaptation). This paper is thus the diagnostic half of a diagnosis-then-treatment program: it establishes, with evidence rather than assumption, the problem that a subsequent domain-adaptation model is designed to solve.

To ground the analysis against a specialized baseline, we additionally reproduce SAAN [4], an art-specific aesthetic model, on its own benchmark; its modest in-domain ceiling (SRCC 0.472) frames how much aesthetic signal is available in the artistic domain at all, and cautions against over-reading raw correlation magnitudes.

## 1.3 Paper Organization

Section 2 reviews the relevant course modules and situates our work against prior cross-domain aesthetics research. Section 3 (Methodology) defines the probing, gating, and transfer statistics. Section 4 presents the experiments: which attributes are readable, how they transfer from photography to art, and whether the pattern is representation-invariant. Section 5 discusses the mechanism behind the observed shifts — including the color-axis collapse and the depth-of-field reversal — together with the study's limitations and its connection to future domain-adaptive modeling. Section 6 concludes.

---

# 2. Related Work

## 2.1 Image Aesthetic Assessment

Image aesthetic assessment (IAA) seeks to predict human aesthetic preference from image content, and its trajectory mirrors the broader course of computer vision, echoing the progression from **Structural** to **Visual** and **Multimodal Intelligence** covered in this course. Early work framed the problem as machine learning over hand-crafted features: Datta et al. [10] and Ke et al. [11] designed high-level descriptors -- color distributions, sharpness, and composition heuristics -- to separate high- from low-quality photographs, and Dhar et al. [12] recast aesthetics prediction in terms of explicitly *describable attributes*, a direction our own attribute-level analysis inherits. The field's central benchmark, AVA [3], provided roughly 250,000 rated photographs and shifted the community toward learned representations; AADB [6] complemented it with eleven interpretable photographic attributes, which we adopt as the vocabulary probed in this work.

The deep-learning era replaced hand-crafted descriptors with end-to-end models. NIMA [13] predicts the full human rating distribution rather than a single scalar and became a standard deep IAA baseline. Most recently, multimodal large language models have been repurposed as aesthetic scorers: Q-Align [14] casts scoring as discrete rating-token prediction and reads a continuous score from token probabilities, and ArtiMuse [15] extends this "token-as-score" idea on an InternVL backbone trained across photography, painting, design, and 3D content. These models -- spanning hand-crafted features, deep networks, and multimodal LLMs -- constitute the photographic-IAA lineage against which we measure transfer to art, and we include representatives of each family in our evaluation grid. The two most recent, Q-Align and ArtiMuse, score aesthetics through language-token prediction on large multimodal-LLM backbones — placing recent IAA squarely within the course's **Language Intelligence** module alongside its visual and multimodal ones.

## 2.2 Artistic Image Aesthetics

A parallel line of work holds that artistic images are judged by criteria that need not coincide with those of photography, and the recent history of art-specific datasets makes this assumption explicit. Early painting datasets such as JenAesthetics [16] paired small collections of artworks with subjective aesthetic scores, and VAPS [17] scored famous paintings along several expressive dimensions. BAID [4] scaled this effort to roughly 60,000 artworks with an accompanying style-aware model, SAAN, which we reproduce as the art-specialist in our grid. More recently, APDD [5] argued that photographic aesthetics has matured while painting aesthetics remained comparatively unexplored, and introduced the first painting dataset annotated with fine-grained aesthetic *attributes* -- a direct analogue, in the artistic domain, of the photographic attributes we probe. The very existence of these art-specific datasets and models embodies an implicit assumption that photographic aesthetic models do not transfer to art. Our work makes this assumption explicit and measurable: rather than proposing another art-specific dataset or model, we quantify *which components* of photographic aesthetics fail to transfer and by how much, providing the diagnostic basis that such specialized resources implicitly presuppose.

## 2.3 Probing Frozen Representations

Our methodology builds on the *linear-probing* framework introduced by Alain and Bengio [8], in which a linear classifier is trained on the features of a frozen network purely as a diagnostic: the probe reads the intermediate representation but, by construction, does not participate in or alter the model's training, and is fitted after the model itself is fixed. A property that such a low-capacity, non-invasive probe can recover is therefore attributable to the representation itself rather than to a high-capacity head. This makes probing a standard tool for characterizing what information a learned representation exposes [18], and it underlies the now-common practice of evaluating self-supervised and vision-language encoders -- including the DINOv2, CLIP, and SigLIP2 models we use [9, 1, 2] -- by the linear separability of their frozen features.

We adopt this framework but apply it in a comparative, cross-domain setting that differs from its usual use. Rather than probing a single representation for object categories or semantic attributes *within* one domain, we train aesthetic-attribute probes in a photographic domain and transport them, unchanged, across a medium boundary into art. The probe's role here is not to reveal a network's internal training dynamics but to serve as a fixed, interpretable instrument for measuring how an aesthetic axis behaves in one visual medium versus another.

## 2.4 Cross-Domain Transfer of Aesthetics

That aesthetic models generalize poorly beyond their training distribution is not itself new. Even within photography, Kong et al. [6] found that models trained on one aesthetic dataset transfer only weakly to another, attributing the gap to differing rater populations and tastes. More recently, cross-domain aesthetic assessment across genuinely different media has begun to receive attention: Hayashi et al. [7] introduce a dataset spanning art, fashion, and landscape and report that even a dedicated unsupervised domain-adaptation method recovers only about 60% of the supervised upper bound, underscoring how large the cross-domain gap remains.

These studies establish *that* aesthetic models degrade out of domain, but they operate at the level of whole-model performance: a scorer is trained on one distribution and its aggregate correlation drops on another. Because the output is a single opaque score, such comparisons cannot identify *which* aesthetic factor failed to carry over. Our contribution is to decompose this transfer gap attribute by attribute. By probing individual, interpretable attributes and measuring each one's shift across the photography-art boundary, we convert a global "art is different" observation into a structured account that separates attributes which lose relevance, those that reverse, and those the representation cannot read at all. To our knowledge, this attribute-level, cross-architecture treatment of photographic-to-artistic aesthetic transfer has not been reported previously.

---

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

---

# 4. Experiments and Results

## 4.1 Experimental Setup

We evaluate on three aesthetic assessment datasets spanning two visual media. **AADB** [6] provides 8,458 / 500 / 1,000 (train / val / test) photographs, each annotated with an overall aesthetic score and eleven aesthetic attributes; it serves as the attribute source $\mathcal{D}_A$. **AVA** [3] supplies the photographic domain $\mathcal{D}_P$; following the standard `generic_test` split we sample 5,000 images (seed 42, fixed before any correlation was computed), of which 4,976 remain after accounting for images no longer hosted. **BAID** [4] supplies the artistic domain $\mathcal{D}_R$, with 6,394 test artworks after excluding 49 images (0.08%) unreachable from the original host; the excluded identifiers are logged for reproducibility. BAID is released under the CC BY-NC-ND 4.0 license and is used here for non-commercial academic research.

As frozen backbones $\varphi$ we deliberately span three distinct pretraining paradigms, drawn from the course's **Visual Intelligence** and **Multimodal Intelligence** modules: **DINOv2** ViT-B/14 [9], trained by self-supervision without any language signal; **CLIP** ViT-B/16 [1], trained by contrastive image-text alignment; and **SigLIP2** ViT-B/16 [2], a more recent vision-language encoder with a sigmoid contrastive objective and improved dense features. This spread is intentional: if a self-supervised encoder and two differently trained vision-language encoders agree on the transfer structure (Section 4.4), that structure cannot be attributed to any single pretraining signal. All three are used at base scale to control for capacity, so that any observed difference reflects representational family rather than model size. We extract the native pooled representation (768-d for all three) and never fine-tune the backbone: only a linear probe is fit on top, so that a successful read indicates the target axis is *already present* in the representation rather than *constructed* by the probe.

For every attribute $a$ we fit a ridge probe $g_a$ on $\mathcal{D}_A$ train features and evaluate with Spearman's rank correlation coefficient (SRCC). All decisions governing the analysis -- the readability threshold $\tau$, the unanimity rule, the feature definition, and the AVA sampling seed -- were registered before any gate, correlation, or gap value was observed.

## 4.2 Attribute Readability Gate

We first ask which of the eleven photographic attributes are linearly decodable from each frozen representation. An attribute is deemed **readable** only if its gate SRCC on $\mathcal{D}_A$ test reaches $\tau = 0.30$ for **all three** backbones; the unanimity rule fixes a common attribute support across backbones, which the later rank-agreement analysis (Section 4.4) requires.

Seven of eleven attributes pass (Table 1). The four that fail -- RuleOfThirds, Symmetry, BalancingElements, MotionBlur -- are the *spatial and compositional* attributes, whereas all seven survivors concern *color, tone, and content*. This is not an incidental threshold effect: a global pooled vector summarizes *what* is present in an image but discards *where* it is placed, so composition-dependent attributes are structurally harder to recover with a linear read. That even the failures are semantically coherent indicates the gate is measuring a real property of the representation rather than noise. A further regularity supports this reading: across all eleven attributes the minimum backbone in Table 1 is consistently DINOv2, the only purely self-supervised encoder, with the two vision-language encoders reading each attribute slightly better. The gap is small and never changes which side of the threshold an attribute falls on, but it is systematic, and it hints that language-aligned pretraining exposes nameable aesthetic attributes marginally more linearly -- a plausible consequence of these attributes having natural linguistic labels. Because our gate takes the minimum over backbones, the readable set is set by the most conservative encoder, making the seven-attribute support a lower bound rather than an optimistic one.

**Table 1.** Gate SRCC on AADB test ($\tau = 0.30$, unanimous across backbones). Bold = readable.

| Attribute | DINOv2 | CLIP | SigLIP2 | Verdict |
|---|---|---|---|---|
| **Object** | **0.668** | **0.706** | **0.710** | **readable** |
| **Content** | **0.585** | **0.642** | **0.655** | **readable** |
| **VividColor** | **0.525** | **0.692** | **0.684** | **readable** |
| **DoF** | **0.503** | **0.518** | **0.527** | **readable** |
| **ColorHarmony** | **0.438** | **0.492** | **0.476** | **readable** |
| **Light** | **0.367** | **0.457** | **0.487** | **readable** |
| **Repetition** | **0.353** | **0.402** | **0.387** | **readable** |
| BalancingElements | 0.266 | 0.284 | 0.296 | unreadable |
| RuleOfThirds | 0.248 | 0.263 | 0.252 | unreadable |
| Symmetry | 0.228 | 0.276 | 0.257 | unreadable |
| MotionBlur | 0.106 | 0.196 | 0.172 | unreadable |

## 4.3 Transfer Shift from Photography to Art

For each of the seven readable attributes we apply the photo-trained probe $g_a$ -- without any refitting -- to both domains and measure how well it explains the human aesthetic score in each. The transfer shift

```math
T_a = r_a(\mathcal{D}_P) - r_a(\mathcal{D}_R)
```

quantifies how much of an attribute's explanatory power is lost (or gained) when moving from photography to art, where $r_a(\mathcal{D}) = \mathrm{SRCC}(g_a(\varphi(\mathcal{D})),\ \text{score})$. Table 2 reports values averaged over the three backbones, with 95% bootstrap confidence intervals (2000 resamples); the final column gives the bootstrap probability of the opposite sign, and all seven shifts are significant at $p < 0.001$.

**Table 2.** Attribute correlation with aesthetic score in each domain, transfer shift, and 95% bootstrap CI. Positive $T$ = photo-specific; negative $T$ = stronger in art. All shifts significant ($p < 0.001$).

| Attribute | $r$ (photo) | $r$ (art) | $T$ = shift | 95% CI | $p$ |
|---|---|---|---|---|---|
| Light | 0.380 | 0.060 | **+0.320** | [+0.293, +0.347] | <0.001 |
| ColorHarmony | 0.226 | -0.014 | +0.240 | [+0.209, +0.270] | <0.001 |
| Content | 0.368 | 0.152 | +0.216 | [+0.188, +0.244] | <0.001 |
| VividColor | 0.179 | 0.019 | +0.158 | [+0.126, +0.189] | <0.001 |
| Repetition | 0.087 | -0.056 | +0.142 | [+0.112, +0.172] | <0.001 |
| DoF *(reversal)* | 0.112 | 0.183 | **-0.072** | [-0.102, -0.042] | <0.001 |
| Object *(reversal)* | 0.143 | 0.244 | **-0.102** | [-0.133, -0.069] | <0.001 |

Two patterns emerge. First, **five of seven attributes lose explanatory power in the artistic domain**, and the losses concentrate in the *color and tone* family: Light collapses from 0.38 to 0.06, and ColorHarmony and VividColor both fall to near-zero or negative correlation. Axes that characterize a good photograph -- balanced exposure, harmonious color, saturation -- do not characterize a good artwork; in several cases the correlation reverses sign, indicating that a direction learned as positive under photographic supervision runs *against* aesthetic quality in art.

Second, **two attributes reverse**: depth-of-field and object emphasis correlate more strongly with aesthetic quality in art than in photography. Their negative shifts are individually significant (Table 2), so the effect is not a threshold artifact. We report this observation here and defer its interpretation to the Discussion; we note only that it rules out a trivial "everything transfers worse" account -- the shift is attribute-dependent in both magnitude and direction. Figure 1 visualizes these shifts, making the split between the five photo-specific attributes (red) and the two art-preferred reversals (blue) immediately apparent. The magnitudes are also orderly: the largest positive shifts belong to the pure color-tone attributes (Light, ColorHarmony, VividColor), Content -- which carries semantic rather than purely photographic information -- shifts less, and the two reversals are the smallest in absolute value. The gradient from strongly photo-specific, through domain-general, to art-preferred is itself orderly rather than arbitrary.

![Per-attribute transfer shift](../results/fig1_shift.png)

**Figure 1.** Per-attribute transfer shift $T_a = r_a(\text{photo}) - r_a(\text{art})$ for the seven readable attributes, averaged over three backbones. Positive (red) bars mark photo-specific attributes; negative (blue) bars mark depth of field and object emphasis, more relevant in art.

## 4.4 Representation Invariance via Rank Agreement

A natural objection is that the shifts in Table 2 might be an artifact of one particular encoder. To test this we compute Kendall's coefficient of concordance $W$ [19] over the three backbones' rankings of $T_a$. We obtain $W = 0.849$; a permutation test (5,000 random per-backbone rankings) places this far above the null (mean 0.334) with $p = 0.001$, so the agreement is highly significant. Three encoders that differ in pretraining objective and vision-language grounding nonetheless order the seven attributes' transfer shifts almost identically. Sign agreement is in fact unanimous -- DoF and Object are negative under all three backbones, the remaining five positive under all three. The photo-to-art transfer structure is therefore a property of the *domain pair*, not of any single representation. Because the unanimity rule and the set of backbones were fixed in advance, this concordance is a genuine test rather than a post-hoc selection. The practical import is that a reader need not worry which encoder we happened to choose: had we run the study with any one of the three alone, the qualitative conclusion -- which attributes are photo-specific, which reverse -- would have been the same. Figure 2 shows the shift for each backbone separately; the three columns share an almost identical red-to-blue pattern, which is precisely what the high concordance coefficient quantifies.

![Transfer shift per backbone](../results/fig2_heatmap.png)

**Figure 2.** Transfer shift $T_a$ per attribute (rows) and backbone (columns). The near-identical pattern across columns visualizes the rank agreement (Kendall's $W = 0.849$); sign is unanimous across backbones.

## 4.5 Model-Level Analysis: The Specialization Ladder

The probe analysis isolates individual attributes; we now ask whether the same photo-to-art gap appears at the level of complete, published aesthetic scorers. We assemble a grid of four models spanning the specialization spectrum and evaluate each on the artistic domain (BAID test): two photographic specialists -- **Q-Align** [14] and **Aesthetic-Predictor-V2.5** -- a multi-domain model, **ArtiMuse** [15] (trained on aesthetic data spanning photography, painting, design, and 3D), and an art specialist, **SAAN** [4], which we reproduce in-domain. Each model's art-domain SRCC carries a 95% bootstrap CI.

**Table 3.** Cross-domain aesthetic scoring. Photo columns are in-domain references; the art column (BAID) is the cross-domain test with 95% bootstrap CI. Ordering: photo-specialist < multi-domain < art-specialist.

| Model | Type | Photo (AVA) | Art (BAID) | 95% CI (art) |
|---|---|---|---|---|
| Aesthetic-Predictor-V2.5 | photo-specialist | 0.473 | 0.009 | [-0.013, 0.032] |
| Q-Align | photo-specialist | 0.817 | 0.126 | [0.102, 0.151] |
| **ArtiMuse** | **multi-domain** | 0.532 | **0.199** | [0.177, 0.223] |
| SAAN | art-specialist | -- | 0.472 | (in-domain) |

The grid yields a clean ordering on the artistic domain: the two photographic specialists score lowest (0.009 and 0.126), the multi-domain model recovers part of the gap (0.199), and the art specialist leads (0.472). The improvement from the best photographic specialist to the multi-domain model is statistically significant -- ArtiMuse exceeds Q-Align by a mean of +0.073, with $P(\text{ArtiMuse} \le \text{Q-Align}) < 0.001$ under a paired bootstrap and disjoint confidence intervals. The reading is twofold. Multi-domain training *does* partially recover artistic transfer relative to photo-only supervision, confirming that exposure to artistic data helps; yet the large remaining gap to the art specialist shows that broad multi-domain coverage is not a substitute for domain-specific adaptation. The absolute numbers sharpen the point. Q-Align is the strongest photographic model in the grid -- 0.817 on AVA, higher even than the art specialist's in-domain score -- yet on art it falls to 0.126, below a model (ArtiMuse) that is weaker on photographs but has seen artistic data. Raw photographic competence does not buy artistic competence; domain match does. Figure 3 makes the ladder visual: aesthetic competence on art rises monotonically with a model's exposure to artistic supervision, yet a substantial gap to the art specialist remains. We return to this in the Discussion, as it directly frames the target of a future domain-adaptation method.

![Aesthetic competence on art by specialization](../results/fig3_grid.png)

**Figure 3.** Aesthetic scoring on the artistic domain (BAID) for four models ordered by specialization. Photographic specialists score lowest, the multi-domain model recovers part of the gap, the art specialist leads.

## 4.6 Coverage of Readable Attributes

The per-attribute shifts of Section 4.3 raise a system-level question: taken together, how much of each domain's predictable aesthetic signal do the readable attributes account for? We define coverage as the ratio $R^2_{\text{attr}} / R^2_{\text{full}}$, where $R^2_{\text{full}}$ regresses the aesthetic score on the full 768-d representation (the achievable ceiling for that encoder) and $R^2_{\text{attr}}$ regresses it on the seven readable-attribute predictions. The ratio normalizes away the differing ceilings of the two domains, a necessary correction since art aesthetics is intrinsically harder to predict ($R^2_{\text{full}} \approx 0.10$ for BAID vs. $0.45$ for AVA, consistent with SAAN's in-domain ceiling of 0.472 SRCC).

**Table 4.** Coverage of readable attributes, per domain (mean over three backbones, 95% bootstrap CI).

| Domain | $R^2_{\text{full}}$ | $R^2_{\text{attr}}$ | Coverage | 95% CI |
|---|---|---|---|---|
| Photography (AVA) | 0.455 | 0.208 | **0.450** | [0.420, 0.477] |
| Art (BAID) | 0.095 | 0.007 | **0.030** | [-0.150, 0.163] |

In photography, the readable attributes account for roughly **45%** of the aesthetic signal the representation can predict. In art, they account for **essentially none (3%)**: the art-domain confidence interval includes zero, so the attribute predictions explain no more of the artistic aesthetic score than a constant would. The photographic aesthetic vocabulary does not merely weaken in art attribute by attribute; as a system it collapses, covering almost none of what makes an artwork aesthetically strong. That the art-domain interval includes zero is telling: we cannot reject the hypothesis that, taken together, the photographic aesthetic attributes explain *nothing* about what makes an artwork aesthetically strong. Against photography's tight 0.45 interval [0.420, 0.477], this is a difference in kind, not degree, and it motivates domain-adaptive modeling that targets the artistic medium directly rather than hoping a photographic vocabulary will stretch to cover it.

## 4.7 Robustness Analysis (Ablation)

Because the readable set (Section 4.2) underlies every subsequent result, we verify that it does not depend on our pre-registered hyperparameters. We vary the ridge regularization $\lambda$ over two orders of magnitude and the gate threshold $\tau$ around its registered value.

**Table 5.** Sensitivity of the readable set to $\lambda$ and $\tau$. The set is invariant to $\lambda$ and stable across $\tau$; only at $\tau = 0.25$ does one borderline attribute (BalancingElements, min-SRCC 0.266) enter.

| Setting | Readable count | Change from baseline |
|---|---|---|
| $\lambda = 1$ ($\tau = 0.30$) | 7 | identical set |
| $\lambda = 10$ ($\tau = 0.30$) -- ours | 7 | baseline |
| $\lambda = 100$ ($\tau = 0.30$) | 7 | identical set |
| $\tau = 0.25$ ($\lambda = 10$) | 8 | + BalancingElements |
| $\tau = 0.30$ ($\lambda = 10$) -- ours | 7 | baseline |
| $\tau = 0.35$ ($\lambda = 10$) | 7 | identical set |

The readable set is invariant to $\lambda \in \{1, 10, 100\}$ and identical for $\tau \in \{0.30, 0.35\}$; only lowering the threshold to $\tau = 0.25$ admits a single borderline attribute. Our pre-registered choices therefore do not drive the qualitative result.

---

# 5. Discussion

## 5.1 The Color Axis as the Locus of Transfer Failure

The clearest signal in our results is that the attributes which lose the most relevance in art are those governing color and tone: Light ($T=+0.32$), ColorHarmony ($T=+0.24$), and VividColor ($T=+0.16$) are the three largest positive shifts, and all collapse from moderate correlation in photography to near-zero or negative correlation in art. These three are not an arbitrary grouping. In photography, exposure, color harmony, and saturation are strong predictors of a well-received image because photographic practice treats them as technical objectives — a correctly exposed, color-balanced photograph is, all else equal, a better photograph. In painting and illustration, the same axes are compositional *choices* rather than technical targets: a deliberately desaturated palette, a high-key or low-key tonal scheme, or a dissonant color combination can be the very source of a work's aesthetic strength. Expressionist and much modern painting actively exploit tonal imbalance and jarring color as expressive devices, so a scoring direction that rewards conventional color harmony is not merely uninformative in art but can be actively misleading. This is what the sign reversals in Table 2 encode: a direction learned as "good" under photographic supervision points *against* aesthetic quality in the artistic domain for several color-tone attributes.

The coverage analysis (Section 4.6) shows that this is not confined to individual attributes. Taken as a set, the seven readable attributes account for roughly 45% of the predictable aesthetic signal in photography but essentially none of it in art, with the art-domain confidence interval spanning zero. The photographic aesthetic vocabulary, as a system, does not describe what makes an artwork good — and the failure is concentrated in the dimensions photography treats as most objective.

## 5.2 The Depth-of-Field and Object Reversals

The two attributes that behave in the opposite direction — depth of field ($T=-0.07$) and object emphasis ($T=-0.10$) — are the most informative results in the study, because they were not predicted by the "everything transfers worse" intuition. Both correlate *more* strongly with aesthetic quality in art than in photography, and both reversals are individually significant under the bootstrap (Table 2), so the effect is a genuine directional realignment rather than a threshold artifact.

We interpret this as a difference in how deliberately each attribute is constructed in each medium. In photography, depth of field and subject isolation are frequently a by-product of equipment and shooting conditions — a wide aperture blurs a background whether or not the photographer intended it as an aesthetic statement, and consumer cameras and phones produce shallow-depth effects automatically — so the correlation between the attribute and human aesthetic preference is diluted by incidental, unintentional occurrences. In painting and digital art, by contrast, there is no optical accident: every region of focus, every soft or sharp edge, every emphasized object is placed by deliberate authorial decision. An artist who renders a subject in crisp detail against a loosely brushed ground has made a compositional choice, and viewers respond to it as a mark of skill and intent. The attribute therefore carries *more* aesthetic signal in art, not less, because in art the signal is never noise.

This reversal has a conceptual consequence that reaches beyond the two attributes involved. It shows that "transfer failure" is not a uniform downward degradation of every axis, but an attribute-specific *realignment* of what each axis means as the medium changes. Some axes lose their meaning (color-tone), some retain it, and some become more meaningful (deliberate composition). A model that treats cross-domain transfer as a single scalar gap cannot represent this structure at all; only a per-attribute decomposition of the kind we introduce can. This reversal also bears directly on domain adaptation (Section 5.7): the attributes that strengthen in art are the ones an adaptation method must be careful *not* to disturb.

## 5.3 The Directionality of Specialization

The model grid (Section 4.5) makes the asymmetry concrete at the level of complete systems. The two photographic specialists collapse in the artistic domain — Q-Align falls from 0.817 on photographs to 0.126 on art, and Aesthetic-Predictor-V2.5 from 0.473 to 0.009 — while the art specialist, SAAN, reaches 0.472 on the same artworks. A model's aesthetic competence is thus not a general quantity that transfers across media; it is tied to the medium its supervision came from. Even though SAAN's absolute in-domain performance (0.472) is modest, it still outperforms the strongest photographic model on art by a factor of roughly 3.7. On artworks, a specialized-but-weaker art model beats a strong-but-mismatched photographic model decisively — a reminder that in-domain fit dominates raw model capability once the domain boundary is crossed.

The multi-domain model, ArtiMuse, trained on aesthetic data spanning photography, painting, 3D, and design, falls squarely between the two extremes on art: it reaches 0.199 on BAID, above both photographic specialists (0.126 and 0.009) yet far below the art specialist SAAN (0.472). The gap over the best photographic specialist is statistically significant (paired bootstrap $P<0.001$, disjoint confidence intervals), so the ordering photo-specialist $<$ multi-domain $<$ art-specialist is not an artifact of sampling. This ladder is the central finding of the grid, and it admits a precise reading: multi-domain training *does* partially recover artistic transfer relative to photo-only supervision, confirming that mere exposure to artistic data during training helps; yet the large residual gap to the art specialist shows that broad multi-domain coverage is not a substitute for domain-specific adaptation. Simply adding art to the training mix moves a model only part of the way. This is the regime a dedicated domain-adaptation method must improve upon — and, encouragingly, the fact that ArtiMuse already clears the photographic baseline shows the gap is closable rather than fundamental.

## 5.4 The Label Ceiling

Several of our absolute correlations are low even in-domain, and this must be read against the ceiling of the artistic benchmark itself. SAAN, an art-specialized model, reaches only 0.472 on BAID's own test set, and the full 768-dimensional representations achieve $R^2_{\text{full}} \approx 0.10$ on BAID versus $\approx 0.45$ on AVA. Art aesthetics is intrinsically harder to predict, in part because BAID's labels derive from online contest voting that reflects thematic fit, subject popularity, and viral appeal alongside aesthetic merit; the target itself is noisier than a controlled aesthetic rating would be. This is why we report transfer as a *shift* $T_a$ and coverage as a *ratio* rather than relying on raw correlation magnitudes: both quantities normalize away the differing ceilings of the two domains, so that our conclusions concern the *relative* behavior of each attribute across media rather than absolute predictability. The pre-registered decision to demote raw score differences in favor of shift, ratio, and rank-agreement statistics was made to avoid over-reading these low absolute numbers. It also means our central claims — the color-axis collapse, the two reversals, the cross-backbone agreement — are robust to the ceiling: they concern the *ordering and sign* of attribute relationships, which a low but non-degenerate signal is sufficient to establish.

## 5.5 Interpretation of the Unreadable Attributes

The four attributes that never pass the gate — RuleOfThirds, Symmetry, BalancingElements, and MotionBlur — are informative in their own right, and their exclusion is a finding rather than a nuisance. All four are spatial or compositional: they depend on *where* things are placed in the frame, not merely on *what* is present. Our backbones are read out through a single globally pooled vector, which by design summarizes the presence of features while discarding their spatial arrangement. That the layout-dependent attributes are the ones a pooled representation cannot linearly expose is strong internal evidence that the gate measures a real structural property of the representation, not arbitrary noise. It also delimits the scope of our transfer claims: we can speak to how *appearance-based* aesthetic attributes transfer from photography to art, but the composition-based ones lie outside what a pooled probe can see. A spatially resolved read-out — probing patch-level features or attention maps rather than the pooled vector — is the natural way to extend the analysis to them, and we return to this in the limitations below.

## 5.6 Course Learning Reflection

This study is a direct application of two modules of the course. From **Visual Intelligence**, we drew the method of *linear probing* on frozen representations — training a low-capacity linear model on fixed features to diagnose what information a representation linearly exposes, without altering the representation itself. From **Multimodal Intelligence**, we drew both the models (CLIP and SigLIP2, whose image–text contrastive pretraining aligns visual and linguistic representation) and the conceptual framing that a single visual representation may encode different notions of quality for different media. The exercise reinforced, in a concrete setting, a lesson the modules present in the abstract: a frozen representation is not a neutral container but carries the inductive biases of its training distribution, and those biases surface sharply the moment the representation is applied across a domain boundary. Working through the study also made concrete the practical machinery behind these modules that lectures necessarily compress — the feature-extraction pipeline, the sensitivity of results to preprocessing and resolution, the choice of rank-based over absolute metrics, and above all the discipline of pre-registering analysis decisions to keep an empirical study honest.

## 5.7 Limitations and Future Work

Our study has several limitations, each of which points to a concrete extension. First, the linear probe is deliberately low-capacity; a non-linear probe might recover some attribute signal that a linear read misses, though at the cost of the clean "present in the representation" interpretation that makes our gate meaningful. Second, as discussed in Section 5.5, the four unreadable attributes are invisible to our analysis not because they are unimportant to art but because globally pooled features discard spatial layout; a spatially-aware probe over patch tokens is the natural way to bring composition-based attributes into scope. Third, our artistic domain is a single dataset (BAID), and its contest-derived labels impose the ceiling discussed in Section 5.4; replication on other artistic benchmarks — for instance the attribute-annotated painting dataset APDD [5] — would both strengthen the conclusions and, because APDD carries its own art attributes, allow the diagnostic to be run in the art-native attribute space rather than only the photographic one.

The most direct continuation, however, is prescriptive rather than diagnostic. This paper establishes *what* fails to transfer (the color-tone axes), *what* strengthens in art (deliberate depth of field and object emphasis), *what* lies beyond a pooled read (the spatial-compositional attributes), and *how* to measure transfer (the shift statistic and coverage ratio). A domain-adaptation method for artistic aesthetic assessment can therefore be designed against an explicit specification: realign the non-transferring color axes toward the artistic target, preserve the axes that are already stronger in art, and verify the outcome with the very instruments developed here — the same probes and the same shift, measured before and after adaptation. The specialization ladder of Section 4.5 supplies both a baseline to beat (the multi-domain model's 0.199) and a target to approach (the art specialist's 0.472). The diagnosis reported in this paper is, in that sense, the specification for the treatment that follows.

---

# 6. Conclusion

We have presented a per-attribute diagnosis of how photographic aesthetic knowledge transfers — and fails to transfer — to art. By probing interpretable aesthetic attributes out of frozen representations and transporting them across a medium boundary, we converted the vague observation that "art is different" into a structured, quantitative account: color and tone attributes lose their relevance and sometimes reverse sign, depth of field and object emphasis strengthen, four spatial attributes are not linearly readable at all, and the whole photographic vocabulary covers almost none of what makes an artwork aesthetically strong. The pattern holds across three architecturally distinct encoders ($W = 0.849$) and is mirrored at the model level, where a clean specialization ordering separates photographic, multi-domain, and artistic scorers on the artistic domain.

Beyond these findings, the study is diagnostic by design: it identifies *what* a domain-adaptation method for artistic aesthetics must correct (the non-transferring color-tone axes), *what* it must preserve (the axes already stronger in art), and *how* its success should be measured (the same transfer shift and coverage ratio, before and after adaptation). Designing and evaluating that adaptation is the natural continuation of this work.

---

# References

[1] A. Radford, J. W. Kim, C. Hallacy, et al., "Learning Transferable Visual Models from Natural Language Supervision," in *Proc. Int. Conf. Machine Learning (ICML)*, 2021, pp. 8748-8763.

[2] M. Tschannen, A. Gritsenko, X. Wang, et al., "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features," *arXiv preprint arXiv:2502.14786*, 2025.

[3] N. Murray, L. Marchesotti, and F. Perronnin, "AVA: A Large-Scale Database for Aesthetic Visual Analysis," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2012, pp. 2408-2415.

[4] R. Yi, H. Tian, Z. Gu, Y.-K. Lai, and P. L. Rosin, "Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 22388-22397.

[5] X. Jin, Q. Qiao, Y. Lu, H. Wang, S. Gao, H. Huang, and G. Li, "Paintings and Drawings Aesthetics Assessment with Rich Attributes for Various Artistic Categories," in *Proc. Int. Joint Conf. Artificial Intelligence (IJCAI)*, 2024, pp. 7672-7680.

[6] S. Kong, X. Shen, Z. Lin, R. Mech, and C. Fowlkes, "Photo Aesthetics Ranking Network with Attributes and Content Adaptation," in *Proc. European Conf. Computer Vision (ECCV)*, 2016, pp. 662-679.

[7] T. Hayashi, H. Takahara, C. O. Mawalim, H. Narimatsu, A. Kimura, S. Kumano, and S. Okada, "XPASS-Vis: A Dataset for Cross-Domain Personalized Image Aesthetic Assessment," *arXiv preprint arXiv:2606.15629*, 2026.

[8] G. Alain and Y. Bengio, "Understanding Intermediate Layers Using Linear Classifier Probes," in *Proc. Int. Conf. Learning Representations (ICLR), Workshop Track*, 2017.

[9] M. Oquab, T. Darcet, T. Moutakanni, et al., "DINOv2: Learning Robust Visual Features without Supervision," *Transactions on Machine Learning Research (TMLR)*, 2024.

[10] R. Datta, D. Joshi, J. Li, and J. Z. Wang, "Studying Aesthetics in Photographic Images Using a Computational Approach," in *Proc. European Conf. Computer Vision (ECCV)*, 2006, pp. 288-301.

[11] Y. Ke, X. Tang, and F. Jing, "The Design of High-Level Features for Photo Quality Assessment," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2006, pp. 419-426.

[12] S. Dhar, V. Ordonez, and T. L. Berg, "High Level Describable Attributes for Predicting Aesthetics and Interestingness," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2011, pp. 1657-1664.

[13] H. Talebi and P. Milanfar, "NIMA: Neural Image Assessment," *IEEE Trans. Image Processing*, vol. 27, no. 8, pp. 3998-4011, 2018.

[14] H. Wu, Z. Zhang, W. Zhang, C. Chen, L. Liao, C. Li, et al., "Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels," in *Proc. Int. Conf. Machine Learning (ICML)*, 2024.

[15] S. Cao, N. Ma, J. Li, X. Li, et al., "ArtiMuse: Fine-Grained Image Aesthetics Assessment with Joint Scoring and Expert-Level Understanding," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2026.

[16] S. A. Amirshahi, G. U. Hayn-Leichsenring, J. Denzler, and C. Redies, "JenAesthetics Subjective Dataset: Analyzing Paintings by Subjective Scores," in *Proc. European Conf. Computer Vision (ECCV) Workshops*, 2014.

[17] A. Fekete, M. R. Pelowski, E. Specker, D. Brieber, R. Rosenberg, and H. Leder, "The Vienna Art Picture System (VAPS): A Data Set of 999 Paintings and Subjective Ratings for Art and Aesthetics Research," *Psychology of Aesthetics, Creativity, and the Arts*, 2022.

[18] Y. Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances," *Computational Linguistics*, vol. 48, no. 1, pp. 207-219, 2022.

[19] M. G. Kendall and B. Babington Smith, "The Problem of m Rankings," *Annals of Mathematical Statistics*, vol. 10, no. 3, pp. 275-287, 1939.

[20] S. Park, "Attribute-Level Analysis of Cross-Domain Aesthetic Transfer: Code and Reproducibility Materials," GitHub repository, 2026. https://github.com/seyeon-shijuan/aesthetic-transfer.

---

# Appendix A. Data Exclusion and Reproducibility Log

We record every sample excluded from analysis and every processing decision that
affects which items enter a correlation, so that all reported numbers can be
reproduced exactly. All code, feature-extraction scripts, and the exclusion logs
described below are available at https://github.com/seyeon-shijuan/aesthetic-transfer.

## A.1 Dataset exclusions

**AADB (attribute source).** Used in full: 8,458 train / 500 val / 1,000 test, as
released. Test filenames carry a score-prefix (e.g. `0.500_`) that we strip before
matching to images. No samples excluded.

**AVA (photographic domain).** From the standard `generic_test` list we drew a fixed
random sample of 5,000 (seed 42, fixed before any correlation was computed). Of these,
4,976 were scored; 24 were dropped because the image is no longer hosted. A small
number of AVA images are stored with truncated data; these were decoded partially
(PIL `LOAD_TRUNCATED_IMAGES`) rather than discarded, a standard handling for AVA. The
20,000-image AVA train sample (from `generic_ls_train`, seed 42) is used only to fit
the coverage ceiling and never enters a reported correlation.

**BAID (artistic domain).** 49 images failed to download from the host and were
excluded, logged with split and reason in `baid_dropped.csv`; the exclusions fall in
the training portion, leaving the 6,394-image official test split intact.

## A.2 Attribute exclusions (the readability gate)

Four of the eleven AADB attributes fail the readability gate (τ = 0.30, unanimous
across the three backbones) and are excluded from all transfer and coverage analysis:
RuleOfThirds, Symmetry, BalancingElements, and MotionBlur (Table 1). All four are
spatial or compositional attributes, consistent with global pooled features discarding
spatial layout. This exclusion is a *result*, not a preprocessing choice, and is stable
under the ablation of Section 4.7.

## A.3 Model-scoring parse log

The two MLLM-based scorers return a numeric score as generated text, which we parse.
Q-Align and Aesthetic-Predictor-V2.5 returned a valid number for all 4,976 AVA and
6,394 BAID images, so their scores align by position. ArtiMuse failed to yield a
parseable number for a small number of images (4 of 6,394 on BAID; 18 of 4,976 on
AVA), which were skipped. For ArtiMuse we therefore save the kept image identifiers
alongside its scores and compute correlations by matching on identifier rather than
position. This safeguard was added after a bootstrap cross-check revealed that
position-based matching mis-aligned the ArtiMuse scores; identifier-based matching
reproduces the direct evaluation exactly (BAID SRCC 0.199).

## A.4 Fixed choices (pre-registered)

The following were registered before any result was observed: readability threshold
τ = 0.30; the unanimity rule (readable iff gate SRCC ≥ τ for all three backbones);
ridge regularization λ = 10; the pooled 768-d feature for all backbones; and the AVA
sampling seed (42). The ablation in Section 4.7 confirms the readable set is
insensitive to τ and λ within the ranges tested.