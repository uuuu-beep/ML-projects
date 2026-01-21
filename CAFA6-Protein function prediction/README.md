# CAFA 6 Protein Function Prediction (Work in Progress)

This repo contains my **CAFA 6** work focused on building strong **single-model baselines** and **modular prediction "channels"** that can later be combined into an ensemble (the ensemble stage is in progress and not included here).

At the moment, this repo includes **two Kaggle notebooks** and a **PPI network builder script**:

- `CNN1D.ipynb` → **CNN1D sequence-only baseline** (no PPI).
- `ESM2 esmebled version 1.0.ipynb` → **ESM2-based pipeline** (MLP + kNN prior + taxonomy prior) producing channel outputs for the full test set.
- `Homo spaiens PPI .py` → **Human PPI network builder** that integrates STRING and BioGRID for neighbor-based label propagation.

---

## 1) What problem this solves

CAFA 6 is a protein function prediction task: given a protein, predict likely **Gene Ontology (GO)** terms with confidence scores.

A practical approach is to:
1) build strong independent predictors (sequence model, embedding model, priors),
2) export their outputs in a consistent format,
3) later combine them with calibrated weights and optional GO-graph propagation.

This repo covers **step (1) and (2)**.

---

## 2) Notebook: `CNN1D.ipynb` (CNN baseline)

### Goal
Train/infer a **1D convolutional neural network** that predicts GO terms **directly from sequence**.

### Key ideas
- Uses only sequence-derived features.
- Produces a single flat prediction file you can submit or reuse as a "channel" later.
- No PPI used here (so it remains broadly applicable across species).

### Output
- A Kaggle-style prediction TSV (commonly named something like):
  - `submission.tsv`  
  with columns:
  - `EntryID`, `term`, `score`

> This file can be used directly as a submission baseline, and later as a channel input for an ensemble.

---

## 3) Notebook: `ESM2 esmebled version 1.0.ipynb` (ESM2 channels for full test)

### Goal
Use precomputed **ESM2 embeddings** to generate multiple complementary prediction channels:

1) **ESM2 MLP predictions**  
2) **Embedding kNN neighbor prior** (propagates labels from similar proteins)  
3) **Taxonomy prior** (species/taxon-level GO frequency prior)

These are saved as separate TSVs so they can be combined later.

### Pipeline summary

**A) Build GO vocabulary**
- Loads CAFA train labels + external GOA annotations.
- Filters terms by a minimum frequency threshold to form a stable GO term set.

**B) ESM2 MLP channel**
- Trains a lightweight MLP on embeddings → GO multi-label probabilities.
- Predicts for the **full test superset** (not just the 10k "sample_submission" subset).

**C) kNN channel (embedding neighbor prior)**
- Finds nearest neighbors in embedding space.
- Converts neighbor labels into a normalized score per GO term.
- Uses FAISS when available (fast); falls back to sklearn if not.

**D) Taxonomy channel**
- Builds per-taxon GO priors from labeled proteins with taxonomy.
- If a test protein's taxon is unseen or missing → falls back to a global prior.

### Outputs (channel TSVs)
Generated for the **full test** set:

- `esm2.tsv`  → ESM2-MLP predictions  
- `knn.tsv`   → embedding kNN prior  
- `taxo.tsv`  → taxonomy prior  

All use the same format:
- `EntryID` (protein ID)
- `term` (GO:xxxxxxx)
- `score` (float, 0–1)

### Why some files are huge
- `esm2.tsv` and `taxo.tsv` are dense-ish channels (top-K terms per protein), so they can be multiple GB.
- `knn.tsv` is sparse (only terms supported by neighbor labels), so it's much smaller.

**Important note for GitHub:**  
These TSV outputs are **too large for normal Git**. If you need to share them, use:
- Kaggle Datasets / Kaggle output artifacts, or
- Git LFS (not recommended for multi-GB competition files unless necessary).

---

## 4) Script: `Homo spaiens PPI .py` (Human PPI Network Builder)

### Goal
Build a **human-only Protein-Protein Interaction (PPI) network** that can be used for neighbor-based label propagation in CAFA6. This script integrates two major PPI databases to create a unified, high-quality interaction network.

### Data sources
1. **STRING** (v12.0) — Functional association network with confidence scores
2. **BioGRID** (v5.0) — Curated experimental interaction data

### Pipeline summary

**Step 1: Load CAFA protein IDs**
- Parses CAFA train/test FASTA files to extract UniProt accessions.
- These become the "universe" of proteins we care about.

**Step 2: Build ID mappings**
- Parses STRING alias file to create:
  - STRING protein ID → UniProt accession mapping
  - Gene symbol → UniProt mapping (needed for BioGRID)

**Step 3: Build STRING PPI edges**
- Filters interactions by `combined_score` threshold (default ≥ 700).
- Maps STRING IDs to UniProt accessions.
- Keeps only edges between CAFA proteins.

**Step 4: Build BioGRID PPI edges**
- Filters for human–human interactions (taxon 9606).
- Maps gene symbols to UniProt via the STRING alias file.
- Assigns a default confidence weight (0.9) to BioGRID edges.

**Step 5: Merge networks**
- Combines STRING and BioGRID edges.
- For duplicate edges: keeps max weight, tracks source provenance.
- Outputs merged edge list with source attribution.

**Step 6: Build neighbor index**
- Creates a fast lookup: `protein → [(neighbor, weight), ...]`
- Saved as a pickle file for downstream use.

### Usage

```bash
python "Homo spaiens PPI .py" --min_score 700 --data_dir . --out_dir data_ppi
```

**Arguments:**
- `--min_score` : STRING combined_score threshold (0–1000). Default: 700
- `--data_dir` : Directory containing input files
- `--out_dir` : Output directory for PPI files

### Required input files
- `train_sequences.fasta` — CAFA training sequences
- `testsuperset.fasta` — CAFA test sequences
- `9606.protein.aliases.v12.0.txt` — STRING aliases (human)
- `9606.protein.links.detailed.v12.0.txt` — STRING interactions (human)
- `BIOGRID-ORGANISM-Homo_sapiens-5.0.252.tab2.txt` — BioGRID human interactions

### Outputs

| File | Description |
|------|-------------|
| `cafa_protein_ids.tsv` | All CAFA UniProt IDs |
| `string_to_uniprot.tsv` | STRING → UniProt mapping |
| `ppi_string.tsv` | STRING edges (prot_a, prot_b, weight, source) |
| `ppi_biogrid.tsv` | BioGRID edges (prot_a, prot_b, weight, source) |
| `ppi_merged.tsv` | Merged edges with source tracking |
| `ppi_neighbors.pkl` | Neighbor index for fast lookup |

### Why use PPI?

Protein-Protein Interactions encode **functional context**: if protein A interacts with protein B, and B has known GO terms, those terms are more likely to also apply to A. This is the basis of **guilt-by-association** methods.

The neighbor index enables:
- **PPI-based label propagation** as an additional prediction channel
- **Boosting confidence** for predictions supported by interacting neighbors
- **Discovering functions** for poorly annotated proteins via their well-annotated partners

---

## 5) Expected file formats (strict)

All prediction TSVs are written as:

```text
EntryID    term        score
A0A017SE81 GO:0004806  0.123456
A0A017SE81 GO:0005783  0.087000
...
```

Tab-separated, no header quoting, scores as floats.

---

## 6) Next steps (not in this repo yet)

- **Ensemble combination** — Weighted blending of channels with learned or fixed weights.
- **GO graph propagation** — Propagate predictions up the GO hierarchy for consistency.
- **PPI channel integration** — Use the merged PPI network for neighbor-based voting.
- **Calibration** — Isotonic regression or Platt scaling to align channel scores before blending.

---

## 7) Personal reflection

CAFA 6 has been a big step up in complexity compared to my earlier image classification and tabular ML work. The challenges here include:

- **Multi-label prediction at scale** — Thousands of possible GO terms, extreme class imbalance.
- **Heterogeneous data sources** — Sequences, embeddings, taxonomy, PPI — each requiring different preprocessing.
- **Evaluation quirks** — CAFA uses protein-centric and term-centric metrics that don't behave like standard accuracy.
- **Computational constraints** — Managing memory on Kaggle GPUs while processing 100k+ proteins.

Building the modular channel approach taught me to think about ML pipelines as **composable systems** rather than monolithic models. The PPI integration work also deepened my understanding of how biological context (interaction networks, taxonomy) can supplement sequence-based predictions.

This is ongoing work, and the ensemble/calibration stage is the next milestone.
