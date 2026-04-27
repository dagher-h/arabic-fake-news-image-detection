# Detecting Arabic Fake News and Images Using Artificial Intelligence Techniques

A dual-track deep learning framework for multimodal misinformation detection in Arabic media — combining Natural Language Processing for textual credibility assessment and Computer Vision for image forgery detection.

---

## Overview

This project presents an independent dual-track framework designed to detect both fake Arabic news articles and digitally manipulated images. The system maintains full separation between the textual and visual tracks, enabling modularity, independent optimization, and flexible deployment.

A Streamlit interface was developed through which users can submit text, an image, or both together. The system automatically routes each component to its respective model for independent analysis.

This work was developed as a graduation project and has since been expanded into a full research article prepared for submission.

---

## Track 1 — Arabic Fake News Detection (NLP)

**Dataset:** Arabic Fake News Dataset (AFND) — 606,912 news articles from 134 Arabic news websites across 19 countries, labeled as Credible / Not Credible / Undecided.

| Experiment | Model | Data Strategy | Test Accuracy | 5-Fold CV |
|---|---|---|---|---|
| Exp 1 | LSTM | Undecided redistributed | 70.84% | 81% |
| Exp 2 | LSTM | Undecided removed | **85.72%** | **95%** |
| Exp 3 | GRU | Undecided removed | 84.05% | — |

**Best model:** LSTM with "Undecided" class removed — 95% accuracy (5-fold CV)

**Key finding:** Removing the ambiguous "Undecided" class had a greater impact on performance than any architectural change.

Architecture:
```
Input → Embedding (non-trainable) → LSTM(256) → LSTM(128) → Dense(64) → Dense(32) → Sigmoid
```

---

## Track 2 — Image Forgery Detection (Computer Vision)

**Dataset:** Combined CASIA v1 + v2 — 14,188 images (Authentic / Tampered), covering splicing and copy-move forgeries.

| Experiment | Architecture | Strategy | Final Accuracy | F1-Score |
|---|---|---|---|---|
| Exp 1 | ResNet50 + LBP + SRM + XGBoost | Hybrid baseline | 81.18% | 0.81 |
| Exp 2 | ELA + ResNet50 + XGBoost | ELA preprocessing | **92.89%** | **0.93** |
| Exp 3 | KD: ResNet101 → ResNet50 + SRM + LBP + ELA + Ensemble | Knowledge Distillation | 91.89% | 0.92 |
| Exp 4 | ViT-B/16 + Stacking Ensemble | Vision Transformer | 72.10% | 0.72 |

**Best model:** ELA + ResNet50 + XGBoost — 92.89% accuracy, F1 = 0.93

**Key finding:** Domain-specific preprocessing (ELA) outperformed architectural complexity, including Vision Transformers.

---

## Key Findings

- Preprocessing decisions consistently outweigh model architecture choices in both tracks.
- Eliminating label ambiguity in the text track boosted LSTM accuracy from 70% to 95%.
- ELA preprocessing in the image track elevated accuracy from 77% to 92.89% without changing the backbone.
- Vision Transformers underperformed due to the limited size of the CASIA dataset relative to their data requirements.
- The dual-track design proved more practical than a fused multimodal model for real-world deployment.

---

## Datasets

| Track | Dataset | Size | Link |
|---|---|---|---|
| Text | AFND — Arabic Fake News Dataset | 606,912 articles | [Kaggle](https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd) |
| Image | CASIA v1 + v2 | ~14,188 images | [Kaggle](https://www.kaggle.com/datasets/sophatvathana/casia-dataset) |

---

## Tech Stack

| Component | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras, PyTorch |
| Classical ML | XGBoost, LightGBM, Scikit-learn |
| NLP | LSTM, GRU, Embedding layers |
| Vision | ResNet50, ResNet101, ViT-B/16 |
| Forensic Features | ELA, LBP, SRM |
| Interface | Streamlit |
| Platform | Kaggle Notebooks / Google Colab |

---

## Project Structure

```
arabic-fake-news-image-detection/
│
├── text_track/
│   ├── exp1_lstm_redistribution.ipynb
│   ├── exp2_lstm_removal.ipynb           ← best text model
│   └── exp3_gru.ipynb
│
├── image_track/
│   ├── exp1_resnet50_xgboost.ipynb
│   ├── exp2_ela_resnet50.ipynb           ← best image model
│   ├── exp3_knowledge_distillation.ipynb
│   └── exp4_vit_stacking.ipynb
│
├── interface/
│   └── app.py                            ← Streamlit demo
│
└── README.md
```

---

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/dagher-h/arabic-fake-news-image-detection.git
   cd arabic-fake-news-image-detection
   ```

2. Install dependencies
   ```bash
   pip install tensorflow torch torchvision xgboost lightgbm scikit-learn opencv-python pillow numpy pandas matplotlib seaborn streamlit
   ```

3. Download the datasets from the Kaggle links above and place them in a `/data` folder.

4. Run notebooks independently — the text track and image track are fully separate.

5. To launch the interface
   ```bash
   streamlit run interface/app.py
   ```

---

## Academic Context

| | |
|---|---|
| University | Tishreen University, Syria |
| Faculty | Faculty of Informatics Engineering |
| Department | Artificial Intelligence |
| Year | 2024–2025 |

---

## Disclaimer

This system is intended for research purposes only. It should not be used as a standalone tool for fact-checking or content moderation without human oversight and further validation.
