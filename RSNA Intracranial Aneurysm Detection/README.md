## RSNA Intracranial Aneurysm Detection (Kaggle, PyTorch)

This repository contains my first end-to-end project on 3D medical imaging, using the RSNA Intracranial Aneurysm Detection challenge dataset. It was my introduction to working with **CT and MRI head scans**, **DICOM**, and **3D/2.5D representations** rather than simple 2D images.

Because the dataset is >200 GB and my local machine (MacBook M2, 256 GB SSD) cannot store or process it comfortably, **all experiments were run on Kaggle** with their provided GPU and RAM limits. A large part of the work was therefore about designing a **memory-safe, cache-friendly pipeline** that can actually run to completion on realistic hardware.

**Notebook:**  
[`RSNA Intracranial Aneurysm Detection Project.ipynb`](RSNA%20Intracranial%20Aneurysm%20Detection/RSNA%20Intracranial%20Aneurysm%20Detection%20Project.ipynb)

---

### Problem & data

The task is to predict:

- a **global label**: “aneurysm present / absent”, and  
- multiple **location labels** (e.g. specific arterial territories),

from 3D CT/CTA or MR/MRA volumes stored as **DICOM series**. This forced me to learn:

- how medical images are stored (DICOM headers, series, UIDs),  
- how **CT Hounsfield Units (HU)** and **windowing** work,  
- why **MRI intensities are arbitrary** and cannot be normalized like CT, and  
- how clinicians use **maximum intensity projections (MIPs)** in axial / coronal / sagittal views.

---

### Technical approach and iterative tests

The notebook is organised as a series of “Tests”, each one fixing a concrete failure or adding one new idea.

#### Test 1 – First tri-planar MIP baseline

- Built a single-file Kaggle pipeline.
- **Representation:**
  - MRI/MRA → **tri-planar MIPs** (axial, coronal, sagittal) stacked as 3 channels.
- Implemented basic DICOM reading and label loading.
- Naive initial MIP code tried to load too many slices into RAM, causing kernel crashes.
- Fix: re-wrote MIP computation to be **true on-the-fly reduction**, so only a few slices are in memory at once.
- Result: first stable end-to-end run (low performance, but the model trains and finishes).

#### Test 2 – Stability & “smoke test” pipeline

- Added a **“quick test decode”**: try `ds.pixel_array` on a single slice and fall back gracefully if decoding fails, instead of crashing the whole run.
- Introduced **“smoke limits”** (e.g. 600 train / 200 val studies) so I could iterate quickly on a subset while still respecting class balance.
- Improved logging and confusion matrices so I could see clearly when the model was just guessing.
- Outcome: I now had a **repeatable, debuggable training loop** that survives bad series and lets me test changes safely.

#### Test 3 – CT-specific processing & richer evaluation

- Recognised that CT/CTA and MR/MRA behave very differently.
- For CT/CTA, started moving towards **multi-window HU projections** and experimented with different image processing than for MRI.
- Expanded confusion matrices and metrics:
  - per-label ROC-AUC and PR-AUC,
  - per-label threshold tuning (best F1),
  - micro vs macro summaries.
- This made it obvious that:
  - global “aneurysm present” was learnable,
  - location labels were much more imbalanced and fragile.

#### Test 4 – Bone-aware highlight & imbalance handling (scripted baselines)

Implemented and ran patched baselines that:

- Use **multi-window axial MIPs for CT/CTA** (e.g. brain / soft-tissue / bone HU windows).
- Apply a **“highlight stretch”** idea:
  - My intuition was: as a human, I would look for *very bright vascular pixels* when scanning for aneurysms.
  - So I implemented a contrast boost on the brightest intensities to make potential aneurysm regions more visible to the model.
- Realised that **CTA skull bone is also very bright**, so highlight alone would also boost bone.
  - Added a **HU-based bone mask** (with simple morphology) and applied highlight **everywhere except bone** (bone-aware highlight).
- Trained with:
  - `BCEWithLogitsLoss` + **Focal term** and **per-class `pos_weight`** for imbalance,
  - a **class-balanced sampler** so rare positives appear frequently,
  - an optional hierarchy consistency penalty (location ≤ “aneurysm present”).
- These baselines are designed to be **Kaggle-friendly**: small batch size, caching to `/kaggle/working`, and clear “smoke” configs.

#### Test 5 – Modality-split + 2.5D MIL + ASL (designed / prototyped)

This test defines a more advanced architecture that is partly prototyped in the notebook and set up for future runs:

- **Modality-split:** train separate models for CT/CTA and MR/MRA, each with its own preprocessing.
- **2.5D Attention-MIL:**
  - Select top-K axial slices per study (by vesselness / intensity),
  - pass them through a 2D backbone,
  - aggregate with attention pooling to a study-level embedding.
- **CT/CTA pipeline:**
  - multi-window HU,
  - bone mask + morphology,
  - optional vesselness (Frangi / LoG) to focus on vasculature,
  - vessel-weighted highlight.
- **MR/MRA pipeline:**
  - per-slice percentile normalization (no HU),
  - axial slices used directly in the MIL stack.
- **Loss and heads:**
  - **Asymmetric Loss (ASL)** as default (with BCE+Focal available),
  - one “Present” head + 13 location heads,
  - location probabilities **gated by Present^α** at inference.
- **Calibration:**
  - per-label temperature scaling on validation logits,
  - thresholds chosen intentionally:
    - Present: smallest threshold achieving a target precision (e.g. 0.80),
    - Locations: per-label F1-tuned.
- Due to Kaggle resource limits, this remains a **designed / partially run pipeline**, and there is still work to do to reach the performance I would like.

---

### Resource constraints & engineering work

A big part of this project was simply **making things run** under constraints:

- Dataset > 200 GB → everything must run on Kaggle, not locally.
- RAM limits forced:
  - streaming DICOM reads,
  - careful MIP computation,
  - aggressive caching to disk.
- GPU limits meant:
  - small batch sizes,
  - “smoke” runs with 600/200 studies for fast feedback,
  - heavy focus on efficient preprocessing rather than huge 3D models.

---

### What I learned (personal reflection)

This is my favourite project so far because of how much I had to learn and connect:

- **Medical imaging fundamentals:** DICOM structure, HU vs arbitrary MRI intensities, CT windowing, axial/coronal/sagittal views, and why MIPs are used clinically.
- **3D vs 2.5D thinking:** understanding why full 3D CNNs are expensive, and how 2.5D (multiple 2D slices with smart aggregation) can capture much of the 3D context at lower cost.
- **Practical ML under constraints:** dealing with RAM crashes, caching, “smoke” subsets, and the fact that a beautiful idea is useless if it cannot run end-to-end.
- **Evaluation discipline:** reading per-label confusion matrices, distinguishing macro vs micro behaviour, and being careful with threshold tuning so that high “accuracy” does not hide degenerate behaviour.
- **Connecting clinical intuition to code:** the highlight-stretch idea (brightening likely vascular pixels), bone-aware masking, modality-specific preprocessing, and hierarchical heads were all motivated by how a human (or radiologist) would actually look at these scans.

The notebook is not the final word on aneurysm detection, but it represents a **real, iterative research process**: start with something simple, watch where it breaks, learn the underlying physics and statistics, and then redesign the pipeline step by step.