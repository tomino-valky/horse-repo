# 🐴 Horse Colic Survival Analysis

> An end-to-end Python pipeline: EDA → dimensionality reduction → Random Forest survival prediction.
> Outputs a single self-contained HTML report — no server required to view it.

**[▶ View Live Report](https://tomasvalky.github.io/horse-colic-ml/report.html)**

---

## What this project does

Takes the [UCI Horse Colic dataset](https://archive.ics.uci.edu/dataset/47/horse+colic) (~300 clinical records)
and runs a reproducible analysis pipeline:

1. **Data cleaning** — standardises string columns, coerces numeric types, documents missingness
2. **EDA** — pulse vs outcome boxplot, correlation heatmap, PCV/protein scatter, pain frequency
3. **PCA** — scree plot + 2D projection coloured by outcome; shows why numeric vitals alone cannot linearly separate classes
4. **Random Forest classifier** — predicts Lived / Died / Euthanized using 8 features (vitals + categoricals)
5. **HTML report** — all figures and metrics exported into one portable file

### Key findings

- Median pulse in survivors (~50 bpm) is nearly half that of non-survivors (~90 bpm) — the strongest single biomarker
- PCV and pulse share the highest positive correlation (r ≈ 0.41), consistent with haemoconcentration-driven tachycardia
- PCA shows substantial overlap between outcome classes in vital-sign space, confirming that categorical features (pain level, surgery status) are essential for classification
- Random Forest achieves **~72% balanced accuracy** (5-fold CV); confusion is highest at the Died/Euthanized boundary — clinically meaningful, as these outcomes depend on veterinary judgement more than biomarkers alone

---

## Quickstart

```bash
git clone https://github.com/tomasvalky/horse-colic-ml.git
cd horse-colic-ml
pip install -r requirements.txt

# Download horse.csv from Kaggle and place it here, then:
python analyse.py

# Custom paths:
python analyse.py --csv data/horse.csv --out my_report.html
```

Open `report.html` in any browser.

---

## Project structure

```
horse-colic-ml/
├── analyse.py          # main script — runs the full pipeline
├── requirements.txt
├── report.html         # pre-generated output (view without running)
└── README.md
```

---

## Background

This project extends work done in a Data Visualization course (VMU, Kaunas, Lithuania, 2025)
where the same dataset was used for EDA assignments in R (ggplot2) and a Streamlit dashboard (HW7).
The goal here was to rebuild the pipeline in pure Python, move from descriptive to predictive analysis,
and produce a portable output that does not require a running server.

---

## Tech stack

| Tool | Purpose |
|------|---------|
| `pandas` | data loading and cleaning |
| `scikit-learn` | imputation, PCA, Random Forest, evaluation |
| `seaborn` / `matplotlib` | all plots |
| `base64` / stdlib | embedding figures into HTML |

No Streamlit, no Plotly, no external CDN — the output is a single self-contained `.html` file.

---

## Limitations

- ~30% average missingness across features; median imputation preserves sample size but understates variance
- Dataset is from the 1980s–1990s; treatment protocols have changed
- Not validated for clinical use

---

## Author

**Tomáš Války** — MSc Biotechnology, Masaryk University / Erasmus+ VMU Kaunas  
[LinkedIn](https://linkedin.com/in/tomasvalky) · [GitHub](https://github.com/tomasvalky)
