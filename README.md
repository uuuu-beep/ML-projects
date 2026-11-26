# ML-projects

This repository collects a selection of my machine learning projects, built while working through textbooks (*An Introduction to Statistical Learning*, *Hands-On Machine Learning*) and Kaggle competitions, and while preparing for postgraduate applications in computational biology.

I am a First-Class BSc (Hons) Biomedical Sciences graduate (University of Greenwich, 2025) moving toward computational biology. These projects are self-directed — my degree contained no programming or statistics modules, so everything here was built outside the curriculum.

Each subfolder has its own detailed README (technical + personal reflection).

---

## Projects

### 1. RSNA Intracranial Aneurysm Detection (3D medical imaging, PyTorch)

Folder: [`RSNA Intracranial Aneurysm Detection`](RSNA%20Intracranial%20Aneurysm%20Detection/)

- End-to-end pipeline for detecting intracranial aneurysms from **CT/CTA and MR/MRA head scans** (RSNA challenge).
- Work on **DICOM**, **3D → 2.5D representations**, CT windowing, MRI normalization, caching, and GPU-friendly processing on Kaggle.
- Focus on:
  - robust preprocessing and memory-safe MIP generation,
  - modality-specific pipelines (CT vs MRI),
  - loss functions and class imbalance,
  - threshold tuning, calibration, and clear evaluation.
- This project was my first serious step into **medical imaging + deep learning**.

See the project README in that folder for a full description and reflection.

---

### 2. Eid Al-Adha 2025: Sheep Classification Challenge (CNN, TensorFlow/Keras)

Folder: [`Sheep Classification Challenge`](Sheep%20Classification%20Challenge/)

- Kaggle image classification challenge for **7 Arabian sheep breeds**.
- Experiments organised as **Tests 1–5**, progressing from ResNet50 baselines to a final **MobileNetV2** pipeline.
- Key topics:
  - handling **class imbalance** (class weights, focal loss, upsampling),
  - building a strong **data augmentation** pipeline,
  - using **Grad-CAM** to visualise where CNNs focus,
  - comparing loss functions and training strategies (CE warm-up vs focal loss).
- Public leaderboard ≈ **0.615**, private ≈ **0.526** — not state-of-the-art, but an honest stopping point after a series of controlled experiments.

This project shows my early but serious work in **real-world image recognition**, interpretability, and practical training under constraints.

---

### 3. Titanic – Machine Learning from Disaster (tabular ML, scikit-learn)

Folder: [`Titanic - Machine Learning from Disaster`](Titanic%20-%20Machine%20Learning%20from%20Disaster/)

- Classic Kaggle competition on tabular data: predict passenger survival.
- Built after working through examples in **An Introduction to Statistical Learning (ISLR)**.
- Pipeline includes:
  - feature selection and cleaning (dropping noisy columns, handling missing data),
  - one-hot encoding of categorical variables,
  - **logistic regression** as an interpretable baseline,
  - **HistGradientBoostingClassifier** + `GridSearchCV` for a stronger non-linear model,
  - creation of a valid `submission.csv` for Kaggle and a public score around **0.75**.

This project connects textbook theory to a complete ML workflow on real tabular data.

---

### 4. CAFA 6 – Protein Function Prediction (bioinformatics, PyTorch, deep learning)

Folder: [`CAFA6-Protein function prediction`](CAFA6-Protein%20function%20prediction/)

- Kaggle competition for predicting **Gene Ontology (GO)** terms from protein sequences.
- A **four-channel weighted ensemble** with GO-graph propagation, built as composable channels rather than one model:
  - **CNN1D** (weight 0.30) — 1D convolutional network predicting GO terms directly from sequence.
  - **ESM2 + MLP** (weight 0.50) — MLP over pretrained protein language model embeddings, with kNN neighbour prior in embedding space.
  - **PPI neighbour prior** (weight 0.15) — label propagation over a human interaction network merging STRING v12.0 and BioGRID, mapped onto UniProt accessions.
  - **Taxonomy prior** (weight 0.05) — per-taxon GO frequency priors with global fallback.
- Blended scores are **propagated up the GO hierarchy** over all ancestors parsed from `go-basic.obo`, so predictions are consistent with the ontology DAG and are not penalised by CAFA's weighted F-measure.
- The blender is **streaming** — it writes per protein rather than materialising the full score matrix, since the dense channels run to multiple GB.
- Key challenges:
  - **Multi-label prediction at scale** — Thousands of GO terms with extreme class imbalance.
  - **Heterogeneous data fusion** — Combining sequence features, embeddings, taxonomy, and interaction networks.
  - **Bioinformatics workflows** — Working with FASTA files, UniProt IDs, and large-scale protein databases.

This project represents my first serious step into **computational biology** and **bioinformatics**, applying deep learning to a real-world protein function prediction challenge.

See the project README in that folder for full technical details.

---

## Ongoing work (not yet in this repo)

Alongside these completed projects, I'm currently working on:

- **FitMealFinder – AI-powered restaurant meal recommender**  
  A mobile app (React Native + Python backend) that helps users find restaurant meals matching their macro/micronutrient goals. This combines:
  - API integration (menus, geolocation),
  - an AI layer for reasoning about nutrient targets,
  - and a focus on explainable recommendations.  
  Code for this app currently lives in a separate private repo, but it is a big part of my ongoing work.

---

Together, these projects show my progression from **classical tabular ML**, to **natural image CNNs with interpretability**, to **3D medical imaging pipelines**, and how I am now extending this into **bioinformatics (CAFA 6)** and **applied AI systems (FitMealFinder)**. They also reflect how I learn: by iterating, documenting my reasoning, and tying practical decisions back to underlying statistical and ML concepts.