# CAFA 6 — Protein Function Prediction

My work on **CAFA 6**: predicting **Gene Ontology (GO)** terms for proteins from sequence, embeddings, taxonomy and protein–protein interaction context.

The pipeline is built as **four independent prediction channels** that are blended into a single submission, with GO-graph propagation applied so the output is consistent with the ontology hierarchy. I deliberately built it this way — as composable channels rather than one monolithic model — so each source of evidence can be developed, inspected and weighted separately.

### Contents

| File | What it is |
|------|------------|
| `cafa6_multichannel_ensemble.ipynb` | The full pipeline: four channels, GO propagation, blender, submission writer (~1,400 lines) |
| `cnn1d_sequence_baseline.ipynb` | 1D CNN sequence-only baseline, used as the CNN channel |
| `build_human_ppi_network.py` | Human PPI network builder — merges STRING v12.0 and BioGRID onto UniProt accessions |

---

## 1) The problem

Given a protein, predict which GO terms describe it, with a confidence score for each. The difficulty is not any single model — it is that the label space is thousands of terms, hierarchically structured, extremely imbalanced, and that useful evidence comes from several unrelated sources at once.

CAFA scores with a **weighted F-measure** that uses information-accretion weights, so rare, specific terms matter more than common, vague ones. That changes what a good prediction looks like: a model that predicts only safe high-level terms scores poorly.

---

## 2) Architecture

Four channels, each producing scores in a common `(EntryID, term, score)` format, then blended:

```
CNN1D (sequence)        ──┐  0.30
ESM2 + MLP (embeddings)  ──┤  0.50
PPI neighbour prior      ──┼──►  weighted blend ──►  GO-graph propagation ──►  submission.tsv
Taxonomy prior           ──┘  0.15 / 0.05
```

### Channel 1 — CNN1D on raw sequence
A 1D convolutional network predicting GO terms directly from amino acid sequence. No external databases, so it generalises across species. Serves as the baseline and as one channel of the ensemble.

### Channel 2 — ESM2 embeddings + MLP
A lightweight MLP trained on precomputed **ESM2** protein language model embeddings. This is the strongest single channel and carries the largest blend weight (0.50). Predicts over the **full test superset**, not just the sample-submission subset.

### Channel 3 — PPI neighbour prior
Built from the merged STRING + BioGRID human interaction network (see `build_human_ppi_network.py`). Propagates GO terms from a protein's interaction partners — **guilt by association**: if A interacts with B and B is annotated, those annotations become more likely for A.

### Channel 4 — Taxonomy prior
Per-taxon GO frequency priors from labelled proteins. Falls back to a global prior when a test protein's taxon is unseen or missing. Small weight (0.05) — it is a prior, not a predictor.

### Blending and GO-graph propagation
The blender combines channels with fixed weights, then **propagates scores up the GO hierarchy** over all ancestors parsed from `go-basic.obo`. Propagation matters because the ontology is a DAG: if a protein is confidently assigned a specific term, every ancestor of that term must be at least as confident, or the prediction is internally inconsistent and the weighted F-measure penalises it.

The writer is **streaming** — it blends and writes per protein rather than materialising the full score matrix, because the dense channels run to multiple GB and will not fit in Kaggle's memory.

---

## 3) The PPI network builder (`build_human_ppi_network.py`)

Builds a human-only interaction network usable for label propagation.

**Sources:** STRING v12.0 (functional associations, confidence-scored) and BioGRID (curated experimental interactions).

**Steps:**
1. Parse CAFA train/test FASTA to extract the universe of UniProt accessions.
2. Build ID mappings from the STRING alias file — STRING protein ID → UniProt, and gene symbol → UniProt (needed for BioGRID).
3. STRING edges, filtered by `combined_score` ≥ 700, mapped to UniProt, restricted to CAFA proteins.
4. BioGRID edges, filtered to human–human (taxon 9606), gene symbols mapped via the STRING aliases, default confidence weight 0.9.
5. Merge: for duplicate edges keep the maximum weight and track which database it came from.
6. Build a neighbour index (`protein → [(neighbour, weight), ...]`) pickled for fast downstream lookup.

**Usage**

```bash
python build_human_ppi_network.py --min_score 700 --data_dir . --out_dir data_ppi
```

| Argument | Meaning |
|---|---|
| `--min_score` | STRING `combined_score` threshold, 0–1000 (default 700) |
| `--data_dir` | Directory containing the input files below |
| `--out_dir` | Output directory |

**Required inputs:** `train_sequences.fasta`, `testsuperset.fasta`, `9606.protein.aliases.v12.0.txt`, `9606.protein.links.detailed.v12.0.txt`, `BIOGRID-ORGANISM-Homo_sapiens-5.0.252.tab2.txt`

**Outputs:** `cafa_protein_ids.tsv`, `string_to_uniprot.tsv`, `ppi_string.tsv`, `ppi_biogrid.tsv`, `ppi_merged.tsv` (with source attribution), `ppi_neighbors.pkl`

The hardest part of this script was not the graph — it was **identifier reconciliation**. STRING, BioGRID and UniProt each name the same protein differently, and the network is worthless until they agree. Getting that mapping right was most of the work.

---

## 4) Output format

All prediction files are tab-separated:

```text
EntryID    term        score
A0A017SE81 GO:0004806  0.123456
A0A017SE81 GO:0005783  0.087000
```

Intermediate channel TSVs (`esm2.tsv`, `knn.tsv`, `taxo.tsv`) are not committed — the dense ones reach multiple GB, which is well past what Git should hold. They are regenerated by running the notebook.

---

## 5) What I would do next

- **Learned blend weights** rather than the fixed 0.30 / 0.50 / 0.15 / 0.05. The current weights were set by judgement, not fitted, and that is the most obvious thing to improve.
- **Calibration before blending** — isotonic regression or Platt scaling, so the four channels are on a comparable scale before they are averaged. Right now they are not, strictly speaking, comparable.
- **Per-ontology weighting** — BPO, MFO and CCO behave differently and almost certainly do not want the same weights.

---

## 6) What this project taught me

CAFA was a large step up from my earlier image and tabular work, and most of what I learned was not about model architecture:

- **The label space has structure, and ignoring it is a bug.** GO is a DAG, and predictions that are not propagated over ancestors are internally inconsistent. I did not understand this until I read how the weighted F-measure actually scores.
- **Heterogeneous evidence has to be made commensurable before it is combined.** Four channels producing numbers between 0 and 1 does not make those numbers comparable. This is the clearest unresolved weakness in my pipeline and I know why it matters.
- **Identifier reconciliation is real work.** Most of the PPI effort went into making three databases agree on protein names, not into the graph algorithm.
- **Memory shapes design.** The streaming blender exists because the non-streaming version cannot run. A pipeline that does not finish is not a pipeline.

Building this as composable channels rather than one model is the habit I most want to carry forward — it meant I could see which source of evidence was contributing what, instead of having one number and no way to interrogate it.
