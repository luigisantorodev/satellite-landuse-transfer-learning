# Satellite Land Use Classification with Transfer Learning

## What This Project Does

This project classifies satellite imagery into 10 land use and land cover categories — forests, highways, residential areas, rivers, and more — by adapting a CNN that was never trained on satellite images in the first place. Instead of training a network from scratch, this project uses **transfer learning**: starting from a model pre-trained on millions of everyday photographs, and adapting it to a visually very different domain with a fraction of the data and compute a from-scratch approach would require.

---

## The Data

EuroSAT, a dataset of 27,000 RGB satellite images (64x64 pixels) captured by the European Space Agency's Sentinel-2 satellite, labeled into 10 land use classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake. The dataset was split 80/20 into training (21,600 images) and test (5,400 images) sets; class balance was verified explicitly afterward, confirming each class was represented at nearly identical proportions in both splits (e.g., the smallest class, Pasture, made up 7.4% of training and 7.3% of test).

---

## Step 1: Why Transfer Learning Instead of Training From Scratch

Training a CNN from scratch requires the network to learn everything from zero, including the most basic visual building blocks: edge detectors, texture filters, simple shape recognizers. This demands large amounts of data and compute to learn well.

The key insight behind transfer learning is that these low-level visual features are remarkably similar across very different problems — a filter that detects vertical edges is useful whether the task is recognizing cats or classifying satellite terrain. A model pre-trained on ImageNet (1.2 million photographs, 1,000 categories) has already learned excellent general-purpose visual features. Rather than re-learning them, this project reuses them directly.

---

## Step 2: Feature Extraction — Freezing Almost the Entire Network

A ResNet18 model, pre-trained on ImageNet, was loaded and **every one of its parameters was frozen** — set to not update during training. Its final classification layer (originally sized for ImageNet's 1,000 categories) was replaced with a new layer sized for EuroSAT's 10 classes. Only this new layer was trained.

The result: just **5,130 trainable parameters**, out of **11.2 million total** in the network — 0.046% of the model. This is the "feature extraction" strategy: the pre-trained network acts purely as a fixed feature extractor, and only a small classifier on top learns to map those features to the new set of classes.

A practical detail worth noting: EuroSAT's images are 64x64 pixels, while ResNet expects 224x224 inputs (the resolution it was trained on). Images were upscaled accordingly, and normalized using ImageNet's exact channel statistics — required for a pre-trained model to correctly interpret pixel values it wasn't specifically trained on.

---

## Step 3: Training and Results

Trained for just 5 epochs, training accuracy rose from 83% in the first epoch to 91.8% by the fifth, with loss decreasing consistently — fast convergence, consistent with training only a small fraction of the network's parameters.

On the held-out test set, the model reached **93.19% accuracy**, with macro and weighted F1-scores both at 0.93 — indicating fairly uniform performance across all 10 classes rather than a few classes propping up the average.

| Metric | Score |
|---|---|
| Test Accuracy | 93.19% |
| Macro avg F1 | 0.93 |
| Weighted avg F1 | 0.93 |
| Trainable parameters | 5,130 (0.046% of the network) |

---

## Step 4: Where the Model Struggles

Two classes stood out as harder to distinguish from each other: **Highway** and **River** were the model's weakest pair, with 27 highway images misclassified as river and 38 river images misclassified as highway — both scoring an F1 around 0.87, noticeably below the other eight classes.

A plausible explanation, rather than a modeling flaw: both highways and rivers tend to appear in satellite imagery as long, narrow, elongated features cutting across the frame — a geometric similarity that could reasonably confuse a model relying on general visual features rather than domain-specific cues (e.g., water reflectance patterns, road markings) that a model trained from scratch directly on satellite data might learn to exploit more precisely.

---

## Summary: What We Learned

Transfer learning let this project reach strong performance (93%+ accuracy across 10 classes) on a domain quite different from ImageNet's natural photographs, by training a tiny fraction of a much larger pre-trained network. This demonstrates the practical value of the approach: general visual features learned once, from a large and diverse dataset, transfer effectively even to domains as visually distinct as satellite imagery — without needing millions of labeled examples or extensive compute to relearn them from scratch.

---

## Tools Used

- **Python** with **PyTorch** and **torchvision** for the model, transfer learning, and data pipeline
- **scikit-learn** for evaluation metrics (classification report, confusion matrix)