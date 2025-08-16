
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------
# Config
# -------------------------
CSV_FILE = "WineQT.csv"        # put this file next to this script
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load & quick info
# -------------------------
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Can't find {CSV_FILE}. Put it in this folder and re-run.")

df = pd.read_csv(CSV_FILE)
print("\n--- First 5 rows ---")
print(df.head().to_string(index=False))
print("\n--- Shape ---", df.shape)
print("\n--- Columns ---")
print(df.columns.tolist())
print("\n--- Missing values (per column) ---")
print(df.isnull().sum())
print("\n--- Dtypes ---")
print(df.dtypes)
print("\n--- Describe (numeric) ---")
print(df.describe().T)

# Save a CSV summary
df.describe().to_csv(os.path.join(OUTPUT_DIR, "describe_numeric.csv"))

# -------------------------
# Prepare numeric cols
# -------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric columns detected: {numeric_cols}")

# store figures so we can export to a single PDF later
figures = []

# -------------------------
# 1) Histograms (one per numeric column)
# -------------------------
for col in numeric_cols:
    data = df[col].dropna()
    if data.empty:
        continue
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data, bins=30, kde=True, ax=ax)
    ax.set_title(f"Histogram — {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fn = os.path.join(OUTPUT_DIR, f"hist_{col}.png")
    fig.savefig(fn, dpi=150)
    print(f"Saved: {fn}")
    figures.append(fig)

# -------------------------
# 2) Boxplots (one per numeric column)
# -------------------------
for col in numeric_cols:
    data = df[col].dropna()
    if data.empty:
        continue
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=data, ax=ax)
    ax.set_title(f"Boxplot — {col}")
    ax.set_xlabel(col)
    fig.tight_layout()
    fn = os.path.join(OUTPUT_DIR, f"box_{col}.png")
    fig.savefig(fn, dpi=150)
    print(f"Saved: {fn}")
    figures.append(fig)

# -------------------------
# 3) Correlation heatmap
# -------------------------
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fn = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(fn, dpi=150)
    print(f"Saved: {fn}")
    figures.append(fig)
else:
    print("Not enough numeric columns for a correlation heatmap.")

# -------------------------
# 4) Scatter plot (alcohol vs pH preferred, else first two numeric cols)
# -------------------------
scatter_done = False
if 'alcohol' in df.columns and 'pH' in df.columns:
    xcol, ycol = 'alcohol', 'pH'
    scatter_done = True
elif len(numeric_cols) >= 2:
    xcol, ycol = numeric_cols[0], numeric_cols[1]
    scatter_done = True

if scatter_done:
    fig, ax = plt.subplots(figsize=(7, 5))
    if 'quality' in df.columns and pd.api.types.is_numeric_dtype(df['quality']):
        sc = ax.scatter(df[xcol], df[ycol], c=df['quality'], cmap='viridis', alpha=0.8)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('quality')
    else:
        ax.scatter(df[xcol], df[ycol], alpha=0.7)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"Scatter: {xcol} vs {ycol}")
    fig.tight_layout()
    fn = os.path.join(OUTPUT_DIR, f"scatter_{xcol}_vs_{ycol}.png")
    fig.savefig(fn, dpi=150)
    print(f"Saved: {fn}")
    figures.append(fig)

# -------------------------
# 5) Quality distribution (if present)
# -------------------------
if 'quality' in df.columns:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='quality', data=df, ax=ax, palette='Set2')
    ax.set_title("Quality Distribution")
    ax.set_xlabel("Quality")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fn = os.path.join(OUTPUT_DIR, "quality_distribution.png")
    fig.savefig(fn, dpi=150)
    print(f"Saved: {fn}")
    figures.append(fig)

# -------------------------
# 6) Save all figures into a single PDF
# -------------------------
pdf_path = os.path.join(OUTPUT_DIR, "visual_report.pdf")
with PdfPages(pdf_path) as pdf:
    for fig in figures:
        pdf.savefig(fig)
    # add a text page with brief summary
    summary_fig = plt.figure(figsize=(8.5, 11))
    summary_fig.clf()
    text = [
        "Visual Report — Summary",
        f"Dataset: {CSV_FILE}",
        f"Rows: {df.shape[0]}, Columns: {df.shape[1]}",
        f"Numeric columns: {', '.join(numeric_cols)}",
        "",
        "Generated charts: histograms, boxplots, correlation heatmap, scatter, quality distribution (if present)."
    ]
    summary_fig.text(0.1, 0.9, "\n".join(text), fontsize=11, va='top')
    pdf.savefig(summary_fig)
    plt.close(summary_fig)

print(f"\nAll visuals saved in folder: {os.path.abspath(OUTPUT_DIR)}")
print(f"Combined PDF: {pdf_path}")

# Close all created figures
plt.close('all')
