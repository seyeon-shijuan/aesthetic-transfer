# 4. Experiments and Results

## 4.1 Experimental Setup

We evaluate on three aesthetic assessment datasets spanning two visual media. **AADB** [1] provides 8,458 / 500 / 1,000 (train / val / test) photographs, each annotated with an overall aesthetic score and eleven aesthetic attributes; it serves as the attribute source $\mathcal{D}_A$. **AVA** [3] supplies the photographic domain $\mathcal{D}_P$; following the standard `generic_test` split we sample 5,000 images (seed 42, fixed before any correlation was computed), of which 4,976 remain after accounting for images no longer hosted. **BAID** [2] supplies the artistic domain $\mathcal{D}_R$, with 6,394 test artworks after excluding 49 images (0.08%) unreachable from the original host; the excluded identifiers are logged for reproducibility.

As frozen backbones $\varphi$ we deliberately span three distinct pretraining paradigms, drawn from the course's **Visual Intelligence** and **Multimodal Intelligence** modules: **DINOv2** ViT-B/14 [4], trained by self-supervision without any language signal; **CLIP** ViT-B/16 [5], trained by contrastive image–text alignment; and **SigLIP2** ViT-B/16 [6], a more recent vision–language encoder with a sigmoid contrastive objective and improved dense features. This spread is intentional: if a self-supervised encoder and two differently trained vision–language encoders agree on the transfer structure (Section 4.4), that structure cannot be attributed to any single pretraining signal. All three are used at base scale to control for capacity, so that any observed difference reflects representational family rather than model size. We extract the native pooled representation (768-d for all three) and never fine-tune the backbone: only a linear probe is fit on top, so that a successful read indicates the target axis is *already present* in the representation rather than *constructed* by the probe.

For every attribute $a$ we fit a ridge probe $g_a$ on $\mathcal{D}_A$ train features and evaluate with Spearman's rank correlation coefficient (SRCC). All decisions governing the analysis — the readability threshold $\tau$, the unanimity rule, the feature definition, and the AVA sampling seed — were registered before any gate, correlation, or gap value was observed.

## 4.2 Which Attributes Are Linearly Readable? (Gate)

We first ask which of the eleven photographic attributes are linearly decodable from each frozen representation. An attribute is deemed **readable** only if its gate SRCC on $\mathcal{D}_A$ test reaches $\tau = 0.30$ for **all three** backbones; the unanimity rule fixes a common attribute support across backbones, which the later rank-agreement analysis (Section 4.4) requires.

Seven of eleven attributes pass (Table 1). Critically, the four that fail — RuleOfThirds, Symmetry, BalancingElements, MotionBlur — are precisely the *spatial and compositional* attributes, whereas all seven survivors concern *colour, tone, and content*. This is not an incidental threshold effect: a global pooled vector summarizes *what* is present in an image but discards *where* it is placed, so composition-dependent attributes are structurally harder to recover with a linear read. That even the failures are semantically coherent indicates the gate is measuring a real property of the representation rather than noise.

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

## 4.3 Do Readable Attributes Transfer From Photography to Art? (Transfer Shift)

For each of the seven readable attributes we apply the photo-trained probe $g_a$ — without any refitting — to both domains and measure how well it explains the human aesthetic score in each. The transfer shift

$$T_a = r_a(\mathcal{D}_P) - r_a(\mathcal{D}_R)$$

quantifies how much of an attribute's explanatory power is lost (or gained) when moving from photography to art, where $r_a(\mathcal{D}) = \mathrm{SRCC}(g_a(\varphi(\mathcal{D})),\ \text{score})$. Table 2 reports values averaged over the three backbones.

**Table 2.** Attribute correlation with aesthetic score in each domain, and transfer shift. Positive $T$ = photo-specific; negative $T$ = stronger in art.

| Attribute | $r$ (photo) | $r$ (art) | $T$ = shift |
|---|---|---|---|
| Light | 0.380 | 0.060 | **+0.321** |
| ColorHarmony | 0.226 | −0.014 | +0.240 |
| Content | 0.368 | 0.152 | +0.217 |
| VividColor | 0.179 | 0.019 | +0.160 |
| Repetition | 0.087 | −0.056 | +0.142 |
| DoF | 0.112 | 0.183 | −0.072 |
| Object | 0.143 | 0.244 | −0.101 |

Two patterns emerge. First, **five of seven attributes lose explanatory power in the artistic domain**, and the losses concentrate in the *colour and tone* family: Light collapses from 0.38 to 0.06, and ColorHarmony and VividColor both fall to near-zero or negative correlation. Axes that characterize a good photograph — balanced exposure, harmonious colour, saturation — do not characterize a good artwork; in several cases the correlation reverses sign, indicating that a direction learned as positive under photographic supervision runs *against* aesthetic quality in art.

Second, **two attributes reverse**: depth-of-field and object emphasis correlate more strongly with aesthetic quality in art than in photography. We report this observation here and defer its interpretation to the Discussion; we note only that it rules out a trivial "everything transfers worse" account — the shift is attribute-dependent in both magnitude and direction.

## 4.4 Is the Pattern Representation-Invariant? (Rank Agreement)

A natural objection is that the shifts in Table 2 might be an artifact of one particular encoder. To test this we compute Kendall's coefficient of concordance $W$ over the three backbones' rankings of $T_a$. We obtain $W = 0.849$, indicating strong agreement: three encoders that differ in pretraining objective and vision–language grounding nonetheless order the seven attributes' transfer shifts almost identically. Sign agreement is in fact unanimous — DoF and Object are negative under all three backbones, the remaining five positive under all three. The photo-to-art transfer structure is therefore a property of the *domain pair*, not of any single representation. Because the unanimity rule and the set of backbones were fixed in advance, this concordance is a genuine test rather than a post-hoc selection.

## 4.5 Coverage <!-- TODO: D4 — fill after R^2_full / R^2_attr computed -->

<!-- Placeholder. Report R^2_full (aesthetic score regressed on full feature)
     vs R^2_attr (score regressed on readable-attribute predictions), per domain.
     Headline: how much of achievable aesthetic prediction the readable attributes
     account for in photo vs art. -->
