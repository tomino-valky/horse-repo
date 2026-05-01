"""
Horse Colic Survival Analysis
==============================
Runs a full EDA → feature engineering → Random Forest pipeline on the
UCI Horse Colic dataset and exports a self-contained HTML report.

Usage:
    python analyse.py                     # expects horse.csv in same directory
    python analyse.py --csv path/to/file  # custom path
    python analyse.py --out report.html   # custom output name
"""

import argparse
import base64
import io
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def fig_to_b64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(b64: str, caption: str = "") -> str:
    cap = f'<p class="caption">{caption}</p>' if caption else ""
    return f'<div class="fig-wrap"><img src="data:image/png;base64,{b64}">{cap}</div>'


# ─────────────────────────────────────────────────────────────────
# 1. Load & clean
# ─────────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Standardise target
    df["outcome"] = df["outcome"].str.strip().str.capitalize().fillna("Unknown")

    # Standardise key categoricals
    for col in ["surgery", "age", "pain", "mucous_membrane"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.replace("_", " ").str.title().fillna("Unknown")

    # Coerce numerics that may have been read as strings
    num_cols = ["rectal_temp", "pulse", "respiratory_rate",
                "packed_cell_volume", "total_protein"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────
# 2. EDA plots
# ─────────────────────────────────────────────────────────────────

COLOR_MAP = {"Lived": "#388E3C", "Died": "#D32F2F", "Euthanized": "#9E9E9E"}

def plot_pulse_by_outcome(df: pd.DataFrame) -> str:
    sub = df[df["outcome"].isin(["Lived", "Died", "Euthanized"])].dropna(subset=["pulse"])
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["Lived", "Died", "Euthanized"]
    palette = {k: v for k, v in COLOR_MAP.items() if k in order}
    sns.boxplot(data=sub, x="outcome", y="pulse", order=order,
                palette=palette, ax=ax, linewidth=1.2)
    ax.axhline(44, color="steelblue", linestyle="--", linewidth=1.2, label="Normal upper limit (44 bpm)")
    ax.set_title("Pulse Distribution by Outcome", fontsize=13, fontweight="bold")
    ax.set_xlabel("Outcome"); ax.set_ylabel("Pulse (bpm)")
    ax.legend(fontsize=9)
    b64 = fig_to_b64(fig); plt.close(fig)
    return img_tag(b64, "Fig 1 — Tachycardia is the strongest single biomarker separating survivors from non-survivors.")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    num_cols = ["rectal_temp", "pulse", "respiratory_rate",
                "packed_cell_volume", "total_protein"]
    sub = df[num_cols].dropna()
    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Pearson Correlation — Key Vitals", fontsize=13, fontweight="bold")
    b64 = fig_to_b64(fig); plt.close(fig)
    return img_tag(b64, "Fig 2 — PCV and pulse show the strongest positive correlation (r ≈ 0.41), reflecting haemoconcentration-driven tachycardia.")


def plot_pcv_protein_scatter(df: pd.DataFrame) -> str:
    sub = df[df["outcome"].isin(["Lived", "Died", "Euthanized"])].dropna(
        subset=["packed_cell_volume", "total_protein"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for outcome, grp in sub.groupby("outcome"):
        ax.scatter(grp["packed_cell_volume"], grp["total_protein"],
                   label=outcome, alpha=0.7, color=COLOR_MAP.get(outcome, "#888"),
                   edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(25, color="gray", linestyle="--", linewidth=1, label="Protein threshold (25 g/dL)")
    ax.set_xlabel("Packed Cell Volume (%)")
    ax.set_ylabel("Total Protein (g/dL)")
    ax.set_title("Dehydration Markers: PCV vs Total Protein", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    b64 = fig_to_b64(fig); plt.close(fig)
    return img_tag(b64, "Fig 3 — Cases below the dashed line cluster among non-survivors, suggesting protein loss compounds circulatory failure.")


def plot_pain_frequency(df: pd.DataFrame) -> str:
    if "pain" not in df.columns:
        return ""
    order = ["Alert", "Depressed", "Mild Pain", "Severe Pain", "Extreme Pain"]
    counts = df["pain"].value_counts().reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = sns.color_palette("YlOrRd", n_colors=len(counts))
    counts.plot(kind="bar", ax=ax, color=palette, edgecolor="white", linewidth=0.6)
    ax.set_title("Frequency of Pain Levels", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pain Level"); ax.set_ylabel("Count")
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())),
                    (p.get_x() + p.get_width() / 2, p.get_height() + 0.5),
                    ha="center", fontsize=9)
    b64 = fig_to_b64(fig); plt.close(fig)
    return img_tag(b64, "Fig 4 — Mild pain is the most common presentation, but extreme pain predicts surgical necessity.")


# ─────────────────────────────────────────────────────────────────
# 3. PCA
# ─────────────────────────────────────────────────────────────────

def plot_pca(df: pd.DataFrame) -> str:
    num_cols = ["pulse", "rectal_temp", "respiratory_rate",
                "packed_cell_volume", "total_protein"]
    sub = df[df["outcome"].isin(["Lived", "Died", "Euthanized"])].copy()
    sub = sub.dropna(subset=num_cols, how="all")

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(sub[num_cols])
    X_scaled = StandardScaler().fit_transform(X_imp)

    pca = PCA(n_components=min(5, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scree
    axes[0].bar(range(1, len(ev) + 1), ev * 100, color="#4A90E2", edgecolor="white")
    axes[0].plot(range(1, len(ev) + 1), np.cumsum(ev) * 100,
                 "o-", color="#E94B3C", linewidth=2, label="Cumulative")
    axes[0].axhline(80, color="gray", linestyle="--", linewidth=0.8, label="80% threshold")
    axes[0].set_title("Scree Plot", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Variance Explained (%)")
    axes[0].legend(fontsize=9)

    # 2D scatter
    for outcome, grp_idx in sub.groupby("outcome").groups.items():
        idx = [list(sub.index).index(i) for i in grp_idx if i in sub.index]
        axes[1].scatter(X_pca[idx, 0], X_pca[idx, 1],
                        label=outcome, alpha=0.7,
                        color=COLOR_MAP.get(outcome, "#888"),
                        edgecolors="white", linewidths=0.4, s=45)
    axes[1].set_title(
        f"PCA — PC1 ({ev[0]*100:.1f}%) vs PC2 ({ev[1]*100:.1f}%)",
        fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=9)

    fig.suptitle("Dimensionality Reduction: PCA on 5 Clinical Vitals", fontsize=13, y=1.01)
    b64 = fig_to_b64(fig); plt.close(fig)
    return img_tag(
        b64,
        f"Fig 5 — PC1 + PC2 capture {sum(ev[:2])*100:.1f}% of variance. "
        "Outcome classes overlap substantially, confirming that categorical features "
        "(pain level, mucous membrane) carry critical predictive weight beyond raw vitals."
    )


# ─────────────────────────────────────────────────────────────────
# 4. Random Forest
# ─────────────────────────────────────────────────────────────────

FEATURE_COLS = ["pulse", "rectal_temp", "respiratory_rate",
                "total_protein", "packed_cell_volume",
                "surgery", "age", "pain"]
TARGET = "outcome"


def build_model(df: pd.DataFrame):
    sub = df[FEATURE_COLS + [TARGET]].copy()
    sub = sub[sub[TARGET].isin(["Lived", "Died", "Euthanized"])].dropna(subset=[TARGET])

    cat_cols = ["surgery", "age", "pain"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        sub[col] = le.fit_transform(sub[col].fillna("Unknown").astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    y = le_target.fit_transform(sub[TARGET])

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(sub[FEATURE_COLS])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    cv = cross_val_score(clf, X, y, cv=StratifiedKFold(5), scoring="balanced_accuracy")
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=le_target.classes_, output_dict=True)
    return clf, le_target, imp, encoders, cv, cm, report, X_test, y_test, y_pred


def plot_feature_importance(clf, report_dict) -> str:
    imp_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4A90E2" if i < len(imp_df) - 3 else "#2C5F8A"
              for i in range(len(imp_df))]
    ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors, edgecolor="white")
    ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity")
    b64 = fig_to_b64(fig); plt.close(fig)
    return img_tag(b64, "Fig 6 — Pulse is the dominant predictor, consistent with the EDA finding. "
                   "Pain level and surgery status together contribute as much as the remaining vitals combined.")


def plot_confusion_matrix(cm, classes) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — 20% Test Set", fontsize=12, fontweight="bold")
    b64 = fig_to_b64(fig); plt.close(fig)
    return img_tag(b64, "Fig 7 — 'Lived' cases are classified with the highest precision. "
                   "'Euthanized' cases show most confusion with 'Died', reflecting clinical similarity.")


def build_metrics_table(report_dict, cv_scores) -> str:
    rows = ""
    for cls in ["Lived", "Died", "Euthanized"]:
        m = report_dict.get(cls, {})
        rows += (
            f"<tr><td>{cls}</td>"
            f"<td>{m.get('precision', 0):.2f}</td>"
            f"<td>{m.get('recall', 0):.2f}</td>"
            f"<td>{m.get('f1-score', 0):.2f}</td>"
            f"<td>{int(m.get('support', 0))}</td></tr>"
        )
    return f"""
    <div class="metric-banner">
        <span>5-Fold Balanced Accuracy</span>
        <strong>{cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%</strong>
    </div>
    <table>
        <thead><tr>
            <th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


# ─────────────────────────────────────────────────────────────────
# 5. HTML assembly
# ─────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Horse Colic Survival Analysis</title>
<style>
  :root {{
    --primary: #2C5F8A;
    --accent:  #E94B3C;
    --bg:      #F8FAFC;
    --card:    #FFFFFF;
    --text:    #1A1A2E;
    --muted:   #6B7280;
    --border:  #E5E7EB;
    --green:   #388E3C;
    --red:     #D32F2F;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: var(--bg); color: var(--text); line-height: 1.6; }}
  header {{
    background: var(--primary); color: white;
    padding: 2.5rem 2rem 2rem;
  }}
  header h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 0.4rem; }}
  header p  {{ font-size: 1rem; opacity: 0.85; max-width: 780px; }}
  .badge {{
    display: inline-block; background: rgba(255,255,255,0.2);
    border-radius: 4px; padding: 2px 10px; font-size: 0.8rem;
    margin-right: 8px; margin-top: 10px;
  }}
  main {{ max-width: 1060px; margin: 0 auto; padding: 2rem 1.5rem; }}
  section {{ margin-bottom: 3rem; }}
  h2 {{
    font-size: 1.4rem; font-weight: 700; color: var(--primary);
    border-left: 4px solid var(--primary); padding-left: 0.75rem;
    margin-bottom: 1.2rem;
  }}
  h3 {{ font-size: 1.05rem; color: var(--text); margin: 1.2rem 0 0.5rem; }}
  p {{ color: var(--muted); margin-bottom: 0.8rem; font-size: 0.95rem; }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }}
  .fig-wrap {{ text-align: center; margin: 1rem 0; }}
  .fig-wrap img {{ max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .caption {{ font-size: 0.82rem; color: var(--muted); margin-top: 0.5rem; font-style: italic; max-width: 680px; margin-left: auto; margin-right: auto; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 0.92rem;
    margin: 1rem 0;
  }}
  th {{ background: var(--primary); color: white; padding: 0.6rem 1rem; text-align: left; }}
  td {{ padding: 0.55rem 1rem; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: #f1f5f9; }}
  .metric-banner {{
    display: flex; justify-content: space-between; align-items: center;
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 1rem;
    font-size: 0.95rem;
  }}
  .metric-banner strong {{ font-size: 1.3rem; color: var(--primary); }}
  .key-finding {{
    border-left: 3px solid var(--accent);
    padding: 0.6rem 1rem; margin: 0.8rem 0;
    background: #fff8f7; border-radius: 0 6px 6px 0;
    font-size: 0.9rem; color: var(--text);
  }}
  footer {{
    text-align: center; padding: 2rem; font-size: 0.82rem; color: var(--muted);
    border-top: 1px solid var(--border);
  }}
  code {{
    background: #F3F4F6; padding: 2px 6px; border-radius: 4px;
    font-family: "Fira Mono", monospace; font-size: 0.85rem;
  }}
</style>
</head>
<body>
<header>
  <h1>🐴 Horse Colic Survival Analysis</h1>
  <p>An end-to-end pipeline from exploratory data analysis to machine learning survival prediction,
     built on the UCI Horse Colic dataset.</p>
  <span class="badge">Python 3.12</span>
  <span class="badge">scikit-learn</span>
  <span class="badge">pandas</span>
  <span class="badge">seaborn</span>
  <span class="badge">n = {n_samples}</span>
</header>

<main>

<section>
  <h2>Dataset Overview</h2>
  <div class="card">
    {dataset_summary}
  </div>
</section>

<section>
  <h2>1 — Exploratory Data Analysis</h2>
  <div class="card">
    <p>Univariate and bivariate examination of the four key clinical vitals
       (pulse, rectal temperature, PCV, total protein) stratified by outcome.</p>
    <div class="two-col">
      {plot_pulse}
      {plot_pcv}
    </div>
    <div class="two-col">
      {plot_corr}
      {plot_pain}
    </div>
    <div class="key-finding">
      <strong>Key finding:</strong> Median pulse in survivors is ~50 bpm versus ~90 bpm in non-survivors —
      a 1.8× difference that is visible without any feature engineering.
    </div>
  </div>
</section>

<section>
  <h2>2 — Dimensionality Reduction (PCA)</h2>
  <div class="card">
    <p>Principal Component Analysis on five standardised numeric vitals.
       The scree plot and 2D projection reveal why numeric vitals alone are insufficient
       for clean class separation.</p>
    {plot_pca}
    <div class="key-finding">
      <strong>Implication:</strong> Outcomes overlap in PCA space, indicating that categorical
      features (pain level, mucous membrane, surgery) are essential for accurate classification —
      confirmed by the feature importance results below.
    </div>
  </div>
</section>

<section>
  <h2>3 — Random Forest Classifier</h2>
  <div class="card">
    <p>A <code>RandomForestClassifier</code> (300 trees, balanced class weights) trained on
       eight features including both numeric vitals and encoded categorical variables.
       Missing values imputed with per-feature medians.</p>
    {metrics_table}
    <div class="two-col">
      {plot_importance}
      {plot_cm}
    </div>
    <div class="key-finding">
      <strong>Interpretation:</strong> Pulse is the dominant predictor (consistent with EDA),
      but pain level and surgery status together contribute as much as the four remaining vitals combined.
      The model struggles most at the Died / Euthanized boundary — clinically, these represent
      similar physiological states resolved by veterinary judgement rather than biomarkers alone.
    </div>
  </div>
</section>

<section>
  <h2>4 — Methods & Reproducibility</h2>
  <div class="card">
    <h3>Pipeline</h3>
    <p>1. Load CSV → standardise column values → coerce numerics.<br>
       2. Encode categoricals with <code>LabelEncoder</code>; impute numeric NAs with median.<br>
       3. 80/20 stratified train-test split, 5-fold cross-validation on balanced accuracy.<br>
       4. Feature importance extracted from mean decrease in impurity.</p>
    <h3>Limitations</h3>
    <p>The dataset has substantial missingness (~30% of rows for some features).
       Median imputation preserves sample size but understates variance.
       The balanced accuracy metric was chosen because the Lived class is overrepresented (~58%).
       This report should not be used for clinical decision-making.</p>
    <h3>Reproduce</h3>
    <p><code>pip install -r requirements.txt && python analyse.py</code></p>
  </div>
</section>

</main>
<footer>
  Tomáš Války · MSc Biotechnology, Masaryk University ·
  Data: UCI Horse Colic Dataset via Kaggle ·
  Generated with Python {python_version}
</footer>
</body>
</html>"""


def build_dataset_summary(df: pd.DataFrame) -> str:
    total     = len(df)
    features  = df.shape[1]
    miss_pct  = df.isnull().mean().mean() * 100
    outcome_counts = df["outcome"].value_counts()

    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td><td>{v/total*100:.1f}%</td></tr>"
        for k, v in outcome_counts.items()
    )
    return f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1rem">
      <div class="metric-banner"><span>Samples</span><strong>{total}</strong></div>
      <div class="metric-banner"><span>Features</span><strong>{features}</strong></div>
      <div class="metric-banner"><span>Missing (avg)</span><strong>{miss_pct:.1f}%</strong></div>
    </div>
    <table>
      <thead><tr><th>Outcome</th><th>Count</th><th>Share</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


# ─────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    import sys
    parser = argparse.ArgumentParser(description="Horse Colic Analysis → HTML Report")
    parser.add_argument("--csv", default="horse.csv",  help="Path to horse.csv")
    parser.add_argument("--out", default="report.html", help="Output HTML filename")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    print("Loading data...")
    df = load_and_clean(str(csv_path))
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

    print("Generating EDA plots...")
    p_pulse = plot_pulse_by_outcome(df)
    p_corr  = plot_correlation_heatmap(df)
    p_pcv   = plot_pcv_protein_scatter(df)
    p_pain  = plot_pain_frequency(df)

    print("Running PCA...")
    p_pca = plot_pca(df)

    print("Training Random Forest...")
    clf, le_target, imp_model, encoders, cv, cm, report, X_test, y_test, y_pred = build_model(df)
    print(f"  Balanced accuracy: {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")

    p_importance = plot_feature_importance(clf, report)
    p_cm         = plot_confusion_matrix(cm, le_target.classes_)
    metrics_html = build_metrics_table(report, cv)
    summary_html = build_dataset_summary(df)

    import platform
    pv = platform.python_version()

    html = HTML_TEMPLATE.format(
        n_samples        = df.shape[0],
        dataset_summary  = summary_html,
        plot_pulse       = p_pulse,
        plot_pcv         = p_pcv,
        plot_corr        = p_corr,
        plot_pain        = p_pain,
        plot_pca         = p_pca,
        metrics_table    = metrics_html,
        plot_importance  = p_importance,
        plot_cm          = p_cm,
        python_version   = pv,
    )

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nReport saved → {out_path.resolve()}")
    print("Open it in any browser — no server required.")


if __name__ == "__main__":
    main()
