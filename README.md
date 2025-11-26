# ML-projects

This repository collects a selection of my machine learning projects that I completed while learning from textbooks (ISLR, deep learning courses) and Kaggle competitions, and while preparing for PhD applications.

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
- Includes the final **Kaggle submission (`finalresultgoat.csv`)** and leaderboard scores, plus an optional screenshot.

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

## Ongoing work (not yet in this repo)

Alongside these completed projects, I’m currently working on:

- **CAFA 6 – Protein Function Prediction**  
  Designing deep learning models (and baselines) to predict protein function from sequence, exploring sequence encoders, multi-label evaluation metrics, and large-scale bioinformatics workflows.

- **FitMealFinder – AI-powered restaurant meal recommender**  
  A mobile app (React Native + Python backend) that helps users find restaurant meals matching their macro/micronutrient goals. This combines:
  - API integration (menus, geolocation),
  - an AI layer for reasoning about nutrient targets,
  - and a focus on explainable recommendations.  
  Code for this app currently lives in a separate private repo, but it is a big part of my ongoing work.

---

Together, these projects show my progression from **classical tabular ML**, to **natural image CNNs with interpretability**, to **3D medical imaging pipelines**, and how I am now extending this into **bioinformatics (CAFA 6)** and **applied AI systems (FitMealFinder)**. They also reflect how I learn: by iterating, documenting my reasoning, and tying practical decisions back to underlying statistical and ML concepts.