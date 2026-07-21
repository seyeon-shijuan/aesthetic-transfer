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