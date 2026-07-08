# Pre-Inference Routing for Cost-Efficient Document Field Extraction

Route each document to a **cheap** or **expensive** extraction model *before* running
any extraction, based on interpretable, document-intrinsic features. The core question
is not "does routing work" but **when** it works: this repo builds a **diagnostic** for
deciding, before spending, whether pre-inference routing is useful, predictable, and
transferable on a given document genre.

> **TL;DR.** Pre-inference routing pays off only under two conditions: (1) **routable
> headroom** (the cheap model fails on a non-trivial fraction of documents) and (2)
> **feature-detectable difficulty** (that failure tracks observable document properties,
> not purely semantic content). We validate this across **five genres**. Where both hold,
> a calibrated router cuts cost by **31 to 33% on receipts** and **77% on degraded ad-buy
> forms** (a non-receipt genre) at quality within 0.02 F1 of always-large. Two genres fail,
> each violating exactly one condition (clean digital invoices, near-ceiling nutrition
> labels). A simple text router matches our engineered features on every genre, so the
> binding constraint is the genre, not the router; we keep the interpretable features
> because they explain *why* a genre is routable. F1 uses a field-type **canonical** matcher
> (money to numeric, date to date-component, text to normalized exact), with strict
> exact-match reported alongside.

---

## Why this is different

Most cost-saving methods are **cascades**: they run the cheap model first, look at its
output, then decide whether to retry with the expensive model, paying for a speculative
cheap call on *every* document. We decide **pre-inference**, from the document alone, and
pay for exactly one tier.

The reframing that makes this tractable: in structured extraction the schema (the "query")
is fixed and the **document** is what varies in difficulty, so the routing signal lives in
the document, not the request.

---

## The diagnostic (five genres)

*Headroom* is the fraction of documents the cheap tier fails (gap > tau). *AUC* is the
within-genre routing AUC (5-fold CV) achievable from pre-inference features. Routing helps
only where **both** are favorable.

| Genre | Capture | Headroom | AUC | Routes? |
|---|---|---|---|---|
| Receipts (CORD, SROIE) | photo / scan | 44% | 0.71 | yes |
| DeepForm (ad-buy forms) | fax | 41% | 0.91 | yes |
| Invoices (DocILE) | digital | 67% | 0.52 | no (not predictable) |
| Nutrition (POIE) | photo | 16% | 0.63 | no (no headroom) |
| VRDU (registration) | scan | 21% | 0.60 | marginal |

Invoices have ample headroom but are not predictable (semantic difficulty, invisible to
features); nutrition labels are predictable but have almost no headroom (the cheap tier is
near ceiling); VRDU is weak on both. A feature-signal-gated pilot predicted each outcome in
advance.

---

## Key results

**Receipts, within-genre** (pooled held-out test, N=447, tau=0.02; 95% CI over 2000 bootstraps):

| Router | ROC-AUC |
|---|---|
| Calibrated random forest (ours) | **0.706** [0.655, 0.757] |
| Logistic regression | 0.629 |
| Single feature (`ocr_conf`) | 0.551 |

Deployable no-peek savings within 0.02 F1 of always-large: **CORD 33%, SROIE 31%**.

**DeepForm, a non-receipt genre where routing works** (held-out test, two model pairs):

| | Haiku/Opus (5x) | Haiku/Sonnet (3x) |
|---|---|---|
| Held-out AUC | **0.916** | 0.845 |
| RF minus logistic | +0.062 | +0.069 |
| Large-required | 31% | 22% |
| No-peek saving (quality within 0.02) | **77%** | 65% |

**Cross-genre transfer** (rows train, columns test; diagonal is within-genre CV). Routing
does **not** transfer across genres, so the router is refit per genre (which is cheap and works):

| Train \ Test | Receipts | DeepForm | VRDU |
|---|---|---|---|
| Receipts | **0.71** | 0.62 | 0.51 |
| DeepForm | 0.54 | **0.91** | 0.56 |
| VRDU | 0.55 | 0.60 | **0.60** |

**The router can be simple.** A TF-IDF / LSA text router over the same OCR text matches the
engineered features on every genre (receipts 0.76 vs 0.71, DeepForm 0.91 vs 0.89, VRDU 0.59
vs 0.60). We report the interpretable features as the primary router for their explanatory
value, not because they route better.

**Cascades.** Pre-inference routing beats confidence cascades structurally: it pays for
exactly one tier, while a cascade always pays the cheap tier plus escalation. It beats a
same-signal cascade on both receipt datasets and beats even an *oracle* cascade on CORD (33%
vs 20%), where escalation is frequent. The textbook logprob-triggered cascade is not
implementable here at all: the Anthropic Messages API exposes no token logprobs.

All numbers above are reproduced by the scripts in `experiments/` and stored as CSVs in
`results/tables/`.

---

## Repository structure

```
src/
  data/         Loaders: CORD, SROIE, VRDU, DeepForm, DocILE, POIE (+ FUNSD loader, not used in results)
  features/     13 pre-inference features: OCR, image-quality, layout, content/structure
  extraction/   Claude Haiku / Opus / Sonnet tier wrappers + field-type canonical F1 evaluation
  routing/      Trivial baselines (always-small/large, random)
experiments/    Reproducible pipeline + analyses (see "Reproducing the pipeline")
  archive/      Superseded/exploratory scripts, not part of the pipeline (gitignored)
results/
  tables/       Oracle labels, feature table, all experiment output CSVs
  figures/      Paper figures; supplementary/ holds alternative-tau and VRDU plots
  tables_pair2/ Second-pair (Haiku/Sonnet) receipt results
tests/          Unit tests (loaders, features, evaluation, baselines)
```

> **Note.** `data/` (raw datasets, OCR cache, extraction cache) is **not committed**: the
> datasets are publicly available (see below) and the caches are regenerated by the pipeline.

---

## The 13 pre-inference features

Computed before any extraction call, from the image, OCR text, or OCR box geometry only
(never from ground-truth annotations), in four families:

| Family | Features |
|---|---|
| **OCR quality** | `ocr_conf`, `ocr_std`, `ocr_stage`, `short_token_ratio`, `inv_chars_per_word` |
| **Image quality** | `blur_score`, `image_contrast`, `word_height_cv` |
| **Layout** | `crowded_line_frac`, `line_density`, `aspect_ratio` |
| **Content / structure** | `item_density`, `tokens` |

An earlier 16-feature set also used three annotation-derived signals (`label_entropy`,
`label_diversity`, `section_count`); we removed them to eliminate label leakage, and found
it costs almost nothing (pooled AUC 0.707 clean vs 0.731 leaky). A composite
`complexity_score` from early experiments is also excluded (collinear with its components).

---

## Datasets (not included; download separately)

Place each under the path the loader expects (shown below), then run the pipeline. OCR uses
[Tesseract](https://github.com/tesseract-ocr/tesseract), which must be installed and on your
`PATH`.

| Dataset | Genre | Routes? | Source | Local path |
|---|---|---|---|---|
| **CORD** | Receipts | yes | https://github.com/clovaai/cord | `data/CORD/` |
| **SROIE** | Receipts | yes | https://rrc.cvc.uab.es/?ch=13 | `data/SROIE2019/` |
| **DeepForm** | Ad-buy forms | yes | DUE benchmark, https://duebenchmark.com/data | `data/deepform/` |
| **DocILE** | Invoices | no | https://github.com/rossumai/docile (or DUE) | `data/docile/` |
| **POIE** | Nutrition labels | no | POIE release (nutrition-facts v5) | `data/poie/nfv5/nfv5_3125/` |
| **VRDU** | Registration forms | marginal | https://github.com/google-research/google-research/tree/master/vrdu | `data/vrdu/registration-form/` |
| FUNSD | Forms (loader only) | n/a | https://guillaumejaume.github.io/FUNSD/ | `data/FUNSD/` |

DeepForm ships as page images (`page_pngs/`) plus per-split `document.jsonl` and
`documents_content.jsonl`; the loader also reads the source archive if colocated. All raw
downloads live under `data/` so nothing is referenced from outside the repo.

---

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."     # required for the extraction stages (see below)
```

## Model tiers

| Tier | Model | Price (in / out per 1M tok) |
|---|---|---|
| Small | `claude-haiku-4-5-20251001` | $1 / $5 |
| Large | `claude-opus-4-8` | $5 / $25 |
| Large (2nd pair) | `claude-sonnet-4-6` | $3 / $15 |

Haiku/Opus is a clean 5x cost ratio; the Haiku/Sonnet pair (3x) tests cross-pair
generalization. Both tiers use an identical schema-conditioned tool-use prompt, so F1
differences reflect model capability, not prompting.

---

## Reproducing the pipeline

Scripts are numbered by run order. Extraction stages make **API calls** (flagged and priced
below); everything else is local, deterministic, and free. Responses are cached per
`(tier, dataset, split, doc_id)`, so re-running an API stage is free. The number gaps
(05, 11, 14, 15, 18) are superseded or exploratory scripts moved to `experiments/archive/`.

**Stage A: receipts and VRDU (primary within-genre results)**

```bash
# 1. Pre-inference features for every document (local)
python experiments/exp_01_feature_analysis.py --full --split all

# 2. Oracle labels: run both tiers, score canonical F1, compute the gap  (API)
python experiments/exp_02_oracle_labeling.py --dataset cord  --split train
python experiments/exp_02_oracle_labeling.py --dataset cord  --split test
python experiments/exp_02_oracle_labeling.py --dataset sroie --split train
python experiments/exp_02_oracle_labeling.py --dataset sroie --split test
python experiments/exp_02_oracle_labeling.py --dataset vrdu  --split train
python experiments/exp_02_oracle_labeling.py --dataset vrdu  --split test

# 3. Train the router: AUC, savings, Pareto frontiers, tau sweep (local)
python experiments/exp_03_routing_model.py

# 4-9. Analyses (local, no API): gap histogram, threshold transfer, cascade
#      comparison (+ oracle upper bound), bootstrap CIs, feature ablation
python experiments/exp_04_gap_histogram.py
python experiments/exp_06_threshold_transfer.py
python experiments/exp_07_cascade_baseline.py
python experiments/exp_08_significance.py
python experiments/exp_09_feature_ablation.py

# 13. Text baseline (TF-IDF / LSA) + leakage ablation, on the cached labels (local)
python experiments/exp_13_text_baseline.py

# 12. Second model pair (Haiku vs Sonnet 4.6, 3x) on receipts  (API, ~$14)
python experiments/exp_12_second_pair.py

# 10. Optional: de-noise pilot, re-extract boundary docs k times to measure
#     decoding variance + label flip rate near tau  (API, pilot ~$8)
python experiments/exp_10_denoise.py --pilot 80 --control 20
```

**Stage B: the other genres (diagnostic), each self-contained (extract, label, route)**

```bash
python experiments/exp_16_docile_within_genre.py     # invoices     (API)
python experiments/exp_17_poie_pilot.py              # nutrition    (API, ~$2)
python experiments/exp_19_deepform_within_genre.py   # DeepForm     (API, ~$11)
python experiments/exp_21_deepform_second_pair.py    # DeepForm 3x  (API, ~$3, reuses cached Haiku)
```

**Stage C: cross-genre synthesis (local, reads the cached oracle-label CSVs)**

```bash
python experiments/exp_20_cross_genre_transfer.py    # transfer matrix + diagnostic table
```

## Tests

```bash
pytest tests/        # 33 unit tests: loaders, features, evaluation, baselines
```

---

## Output files

| File | Contents |
|---|---|
| `results/tables/feature_table.csv` | 13 features per document, all datasets/splits |
| `results/tables/oracle_labels_*.csv` | Per-doc canonical F1(small/large) + gap + tier label (receipts, VRDU, DeepForm, DocILE), with `*_strict` where applicable |
| `results/tables/poie_pilot.csv` | Nutrition (POIE) features + gap (pilot) |
| `results/tables/routing_predictions.csv` | Per-doc router probabilities (tau=0.02) |
| `results/tables/tau_sensitivity.csv` | AUC + savings across tau in {0.01, 0.02, 0.05} |
| `results/tables/exp0{6..9}_*.csv`, `exp13_*.csv` | Threshold transfer, cascade, CIs, feature ablation, text baseline + leakage |
| `results/tables/exp20_cross_genre_auc.csv` | Cross-genre transfer AUC matrix |
| `results/tables/feature_provenance.csv` | Provenance of each router feature (OCR / image / box only) |
| `results/tables/denoise_results.csv` | Per-doc decoding-variance + label flip rate (optional pilot) |
| `results/tables_pair2/*.csv` | Second-pair (Haiku/Sonnet) receipt oracle labels, predictions, tau-sensitivity |
| `results/figures/*.png` | Paper figures (Pareto frontiers, gap histogram); alternative-tau/VRDU plots in `supplementary/` |

---

## License

Code released under the MIT License (see `LICENSE`). Datasets retain their original licenses;
see each dataset's source above.
