"""
Production parity-plot figures for the LLM-based polymer erosion-yield predictor.

Generates two figures:
  Figure 1 (Section 1): Fine-tuned GPT-4o only -- random split (by orientation)
                        alongside restricted-group split (by split).
  Figure 2 (Section 2): 2x2 comparison of fine-tuned GPT-4o vs Gaussian Process
                        Regression across random and restricted-group splits.

Set the *_BASE_DIR / *_CSV path placeholders in each USER SETTINGS block before running.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, NullLocator, NullFormatter, FuncFormatter

# =====================================================
# === USER SETTINGS ==================================
# =====================================================

# --- RANDOM paths/settings (by orientation) ---
RANDOM_BASE_DIR = "/path/to/random_split_results"
RANDOM_SPLIT_COUNT = 10

# --- RG paths/settings (by split) ---
RG_BASE_DIR = "/path/to/restricted_group_results"
RG_SPLIT_COUNT = 5
RG_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
RG_MARKERS = ["o", "s", "D", "^", "v"]

# === Export ===
EXPORT_FMT = "svg"
EXPORT_PATH = os.path.join(RG_BASE_DIR, "combined_random_RG_figure.svg")

# === Range limits ===
FULL_MIN, FULL_MAX = 0.001, 100
ZOOM_MIN, ZOOM_MAX = 0.1, 10

# === Titles ===
RANDOM_MAIN_TITLE = "Random Split"
RG_MAIN_TITLE = "Restricted Group Split"
CUSTOM_XLABEL = r"True Value ($\mathrm{\AA^3/atom}$)"
CUSTOM_YLABEL = r"Predicted Value ($\mathrm{\AA^3/atom}$)"

# === Orientation color / marker map ===
ORIENT_STYLE = {
    "ram":    {"color": "#8B246C", "marker": "o"},
    "nadir":  {"color": "#934f06", "marker": "s"},
    "wake":   {"color": "#17becf", "marker": "D"},
    "zenith": {"color": "#bcbd22", "marker": "^"},
}

# === Fonts and style ===
MAIN_TITLE_FONT = 30
SUB_TITLE_FONT = 24
AXIS_LABEL_FONT = 22
TICK_LABEL_FONT = 18
LEGEND_FONT = 18
METRIC_FONT = 18
SUBPLOT_LABEL_FONT = 28


# =====================================================
# === METRIC FUNCTIONS ================================
# =====================================================

def calculate_ome(y_true_log, y_pred_log):
    return np.mean(np.abs(y_true_log - y_pred_log))

def calculate_log_r2(y_true_log, y_pred_log):
    ss_res = np.sum(np.square(y_true_log - y_pred_log))
    ss_tot = np.sum(np.square(y_true_log - np.mean(y_true_log)))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


# =====================================================
# === DATA LOADING ====================================
# =====================================================
_EPS = 1e-300

def load_clean_combined(path):
    df = pd.read_csv(path)
    if "mean pred" in df.columns and "meanpred" not in df.columns:
        df = df.rename(columns={"mean pred": "meanpred"})
    for col in ["true", "meanpred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "errbar" in df.columns:
        df["errbar"] = pd.to_numeric(df["errbar"], errors="coerce")
        df = df.dropna(subset=["true", "meanpred", "errbar"]).copy()
        df["errbar"] = df["errbar"].clip(lower=0)
    elif "stdpred" in df.columns or "std pred" in df.columns:
        std_col = "stdpred" if "stdpred" in df.columns else "std pred"
        df = df.rename(columns={std_col: "errbar"})
        df["errbar"] = pd.to_numeric(df["errbar"], errors="coerce")
        df = df.dropna(subset=["true", "meanpred", "errbar"]).copy()
        df["errbar"] = df["errbar"].clip(lower=0)
    else:
        df = df.dropna(subset=["true", "meanpred"]).copy()
        df["errbar"] = 0.0

    df["true_lin"] = np.power(10.0, df["true"])
    df["meanpred_lin"] = np.power(10.0, df["meanpred"])
    df["errbar_lin"] = df["meanpred_lin"] * np.log(10) * df["errbar"]
    bad = (df["meanpred_lin"] - df["errbar_lin"]) <= 0
    df.loc[bad, "errbar_lin"] = 0.0
    df["x_true"] = df["true_lin"].clip(lower=_EPS)
    df["y_pred"] = df["meanpred_lin"].clip(lower=_EPS)
    df["yerr"] = df["errbar_lin"]
    return df


# ----------------------------------------------------------
# RANDOM: load all splits, combine, tag by orientation
# ----------------------------------------------------------
random_all = []
print("Loading Random splits...")
for i in range(1, RANDOM_SPLIT_COUNT + 1):
    path = f"{RANDOM_BASE_DIR}/split_0{i}/combined/Rlogall_4o_split{i}_combined_test.csv"
    if not os.path.exists(path):
        print(f"  Warning: Missing {path}")
        continue
    df = load_clean_combined(path)
    random_all.append(df)

if not random_all:
    raise RuntimeError("No Random files found.")

random_df = pd.concat(random_all, ignore_index=True)
print(f"Random: combined {len(random_all)} files, {len(random_df)} rows.")

def extract_orientation(q):
    q_lower = str(q).lower()
    for orient in ["ram", "nadir", "wake", "zenith"]:
        if orient in q_lower:
            return orient
    return "unknown"

random_df["orientation"] = random_df["question"].apply(extract_orientation)

df_ram = random_df[random_df["orientation"] == "ram"]
df_other = random_df[random_df["orientation"] != "ram"]

ome_ram = calculate_ome(df_ram["true"], df_ram["meanpred"])
lr2_ram = calculate_log_r2(df_ram["true"], df_ram["meanpred"])
ome_other = calculate_ome(df_other["true"], df_other["meanpred"])
lr2_other = calculate_log_r2(df_other["true"], df_other["meanpred"])

print(f"  RAM rows: {len(df_ram)}  |  Other rows: {len(df_other)}")
print(f"  RAM  -> OME = {ome_ram:.3f}, LogR\u00b2 = {lr2_ram:.3f}")
print(f"  Other -> OME = {ome_other:.3f}, LogR\u00b2 = {lr2_other:.3f}")

METRICS_RAM = (
    f"Ram-Facing:\n"
    f"OME = {ome_ram:.3f}\n"
    f"LogR\u00b2 = {lr2_ram:.3f}"
)
METRICS_OTHER = (
    f"Other Orientations:\n"
    f"OME = {ome_other:.3f}\n"
    f"LogR\u00b2 = {lr2_other:.3f}"
)

# ----------------------------------------------------------
# RG: load splits individually
# ----------------------------------------------------------
rg_splits = []
rg_raw_for_metrics = []
print("\nLoading RG splits...")
for i in range(1, RG_SPLIT_COUNT + 1):
    path = f"{RG_BASE_DIR}/split_0{i}/combined/RGlogall_4o_split{i}_combined_test.csv"
    if not os.path.exists(path):
        print(f"  Warning: Missing {path}")
        continue
    df = load_clean_combined(path)
    df["split"] = f"Split {i}"
    rg_splits.append(df)
    raw = pd.read_csv(path)
    if "mean pred" in raw.columns and "meanpred" not in raw.columns:
        raw = raw.rename(columns={"mean pred": "meanpred"})
    raw["true"] = pd.to_numeric(raw["true"], errors="coerce")
    raw["meanpred"] = pd.to_numeric(raw["meanpred"], errors="coerce")
    raw = raw.dropna(subset=["true", "meanpred"])
    rg_raw_for_metrics.append(raw)

if rg_raw_for_metrics:
    rg_comb = pd.concat(rg_raw_for_metrics, ignore_index=True)
    rg_ome = calculate_ome(rg_comb["true"], rg_comb["meanpred"])
    rg_lr2 = calculate_log_r2(rg_comb["true"], rg_comb["meanpred"])
    rg_metrics = f"OME = {rg_ome:.3f}\nLogR\u00b2 = {rg_lr2:.3f}"
else:
    rg_metrics = "No data"
    rg_ome = np.nan
    rg_lr2 = np.nan
print(f"RG: loaded {len(rg_splits)} splits")
print(rg_metrics)


# =====================================================
# === HELPER: format axes =============================
# =====================================================

def format_ax(ax, xmin, xmax, subtitle=None, show_ylabel=False, show_xlabel=False):
    ax.plot([xmin, xmax], [xmin, xmax], "--", color="k", alpha=0.7, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect("equal", adjustable="box")

    plain = FuncFormatter(lambda v, pos: f"{v:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT)
    ax.grid(True, which="major", linestyle=":")

    if subtitle:
        ax.set_title(subtitle, fontsize=SUB_TITLE_FONT, pad=10)
    if show_ylabel:
        ax.set_ylabel(CUSTOM_YLABEL, fontsize=AXIS_LABEL_FONT)
    if show_xlabel:
        ax.set_xlabel(CUSTOM_XLABEL, fontsize=AXIS_LABEL_FONT)


# =====================================================
# === COMBINED FIGURE =================================
# =====================================================

fig = plt.figure(figsize=(16, 7))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 0.35, 1], wspace=0.05)

ax_rand = fig.add_subplot(gs[0, 0])
ax_rg = fig.add_subplot(gs[0, 2])

# ----- LEFT: Random by orientation (full range) -----
for orient, style in ORIENT_STYLE.items():
    subset = random_df[random_df["orientation"] == orient]
    if subset.empty:
        continue
    ax_rand.errorbar(
        subset["x_true"], subset["y_pred"], yerr=subset["yerr"],
        fmt=style["marker"], markersize=7, linewidth=1.2,
        color=style["color"], alpha=0.9, label=orient,
        zorder=3 if orient != "ram" else 2, capsize=3
    )

format_ax(ax_rand, FULL_MIN, FULL_MAX,
          subtitle=RANDOM_MAIN_TITLE, show_ylabel=True, show_xlabel=True)

# Orientation legend — centered between plots (upper)
orient_handles = []
for orient, style in ORIENT_STYLE.items():
    orient_handles.append(
        mlines.Line2D([], [], color=style["color"], marker=style["marker"],
                      linestyle="None", markersize=8, label=orient)
    )
fig.legend(
    orient_handles, [h.get_label() for h in orient_handles],
    loc="center",
    bbox_to_anchor=(0.5, 0.72),
    fontsize=LEGEND_FONT,
    frameon=True,
)

# Random metrics boxes (original style: two separate boxes)
metrics_ram_text = (
    f"Ram-Facing:\n"
    f"OME = {ome_ram:.3f}\n"
    f"LogR\u00b2 = {lr2_ram:.3f}"
)
metrics_other_text = (
    f"Other Orientations:\n"
    f"OME = {ome_other:.3f}\n"
    f"LogR\u00b2 = {lr2_other:.3f}"
)
ax_rand.text(0.97, 0.03, metrics_other_text, transform=ax_rand.transAxes,
             fontsize=METRIC_FONT - 4, verticalalignment="bottom",
             horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="gray", alpha=0.9))
ax_rand.text(0.97, 0.22, metrics_ram_text, transform=ax_rand.transAxes,
             fontsize=METRIC_FONT - 4, verticalalignment="bottom",
             horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="gray", alpha=0.9))

# (a) label
ax_rand.text(-0.15, 1.05, "(a)", transform=ax_rand.transAxes,
             fontsize=SUBPLOT_LABEL_FONT, fontweight="bold")

# ----- RIGHT: RG by split (zoomed) -----
rg_handles, rg_labels = [], []
for i, df in enumerate(rg_splits):
    h = ax_rg.errorbar(
        df["x_true"], df["y_pred"], yerr=df["yerr"],
        fmt=RG_MARKERS[i], markersize=7, linewidth=1.2,
        color=RG_COLORS[i], alpha=0.9, label=f"Split {i+1}",
        zorder=3, capsize=3
    )
    rg_handles.append(h)
    rg_labels.append(f"Split {i+1}")

format_ax(ax_rg, ZOOM_MIN, ZOOM_MAX,
          subtitle=RG_MAIN_TITLE, show_ylabel=False, show_xlabel=True)

# RG split legend — centered between plots (lower)
fig.legend(
    rg_handles, rg_labels,
    loc="center",
    bbox_to_anchor=(0.5, 0.28),
    fontsize=LEGEND_FONT,
    frameon=True,
)

# RG metrics box
ax_rg.text(0.97, 0.03, rg_metrics, transform=ax_rg.transAxes,
           fontsize=METRIC_FONT, verticalalignment="bottom",
           horizontalalignment="right",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor="gray", alpha=0.9))

# (b) label
ax_rg.text(-0.15, 1.05, "(b)", transform=ax_rg.transAxes,
           fontsize=SUBPLOT_LABEL_FONT, fontweight="bold")

plt.tight_layout()
plt.show()

# =====================================================
# === SAVE OPTION ====================================
# =====================================================
resp = input("Save combined figure? [y/N]: ").strip().lower()
if resp.startswith("y"):
    fig.savefig(EXPORT_PATH, format=EXPORT_FMT, bbox_inches="tight")
    print(f"Saved: {EXPORT_PATH}")
    eps_path = os.path.splitext(EXPORT_PATH)[0] + ".eps"
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"Saved: {eps_path}")
else:
    print("Skipped saving.")


# %% =================================================================
# %% SECTION 2: GPT-4o vs GPR 2x2 comparison figure
# %% =================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, NullLocator, NullFormatter, FuncFormatter

# =====================================================
# === USER SETTINGS ===================================
# =====================================================

# --- GPT-4o paths ---
GPT_RANDOM_BASE_DIR = "/path/to/random_split_results"
GPT_RG_BASE_DIR     = "/path/to/restricted_group_results"
GPT_RANDOM_SPLIT_COUNT = 10
GPT_RG_SPLIT_COUNT     = 5

# --- GPR paths ---
GPR_RANDOM_CSV = "/path/to/gpr_random_split.csv"
GPR_RG_CSV     = "/path/to/gpr_restricted_group_results.csv"
GPR_RG_SPLIT_COUNT = 5

# --- GPR column mapping ---
# Random wide-format file:
GPR_RANDOM_N_SPLITS = 10
GPR_ORIENT_COL      = "cat_orientation"
GPR_TRUE_COL        = "target_e_y (A3/atom): Input"   # linear Ey (random file)
GPR_PRED_FMT        = "random split {i}: prediction"
GPR_UNC_FMT         = "random split {i}: uncertainty (log10)"

# RG long-format file: all predictions in a single prediction column,
# with 'split' column identifying which split each row belongs to.
GPR_RG_TRUE_COL       = "e_y (A3/atom): Input"
GPR_RG_PRED_COL       = "LEO RG split 1: prediction"
GPR_RG_UNC_COL        = "LEO RG split 1: uncertainty (log10)"
GPR_RG_SPLIT_ID_COL   = "split"

# === Export ===
EXPORT_FMT = "svg"
EXPORT_PATH = os.path.join(GPT_RG_BASE_DIR, "combined_2x2_gpt4o_gpr.svg")

# === Range limits ===
FULL_MIN, FULL_MAX = 0.001, 100
ZOOM_MIN, ZOOM_MAX = 0.1, 10

# === Titles ===
RANDOM_TITLE = "Random Split"
RG_TITLE     = "Restricted Group Split"
ROW_LABEL_GPT = "Fine-tuned GPT-4o"
ROW_LABEL_GPR = "Gaussian Process Regression"
CUSTOM_XLABEL = r"True Value ($\mathrm{\AA^3/atom}$)"
CUSTOM_YLABEL = r"Predicted Value ($\mathrm{\AA^3/atom}$)"

# === Orientation style (used for both random panels) ===
ORIENT_STYLE = {
    "ram":    {"color": "#8B246C", "marker": "o"},
    "nadir":  {"color": "#06930b", "marker": "s"},
    "wake":   {"color": "#17becf", "marker": "D"},
    "zenith": {"color": "#bdb822", "marker": "^"},
}

# === RG split colors/markers (used for both RG panels) ===
RG_COLORS  = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
RG_MARKERS = ["o", "s", "D", "^", "v"]

# === Fonts and style ===
MAIN_TITLE_FONT     = 28
ROW_LABEL_FONT      = 22
COL_TITLE_FONT      = 22
AXIS_LABEL_FONT     = 20
TICK_LABEL_FONT     = 16
LEGEND_FONT         = 16
METRIC_FONT         = 14
SUBPLOT_LABEL_FONT  = 24

FIGSIZE_IN = (16, 14)

_EPS = 1e-300


# =====================================================
# === METRICS =========================================
# =====================================================
def calculate_ome(y_true_log, y_pred_log):
    return np.mean(np.abs(y_true_log - y_pred_log))

def calculate_log_r2(y_true_log, y_pred_log):
    ss_res = np.sum(np.square(y_true_log - y_pred_log))
    ss_tot = np.sum(np.square(y_true_log - np.mean(y_true_log)))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


# =====================================================
# === GPT-4o DATA LOADING =============================
# =====================================================
def load_clean_combined_gpt(path):
    """Load GPT-4o combined CSV (log10 true/meanpred, log10 errbar)."""
    df = pd.read_csv(path)
    if "mean pred" in df.columns and "meanpred" not in df.columns:
        df = df.rename(columns={"mean pred": "meanpred"})
    for col in ["true", "meanpred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "errbar" in df.columns:
        df["errbar"] = pd.to_numeric(df["errbar"], errors="coerce")
        df = df.dropna(subset=["true", "meanpred", "errbar"]).copy()
        df["errbar"] = df["errbar"].clip(lower=0)
    elif "stdpred" in df.columns or "std pred" in df.columns:
        std_col = "stdpred" if "stdpred" in df.columns else "std pred"
        df = df.rename(columns={std_col: "errbar"})
        df["errbar"] = pd.to_numeric(df["errbar"], errors="coerce")
        df = df.dropna(subset=["true", "meanpred", "errbar"]).copy()
        df["errbar"] = df["errbar"].clip(lower=0)
    else:
        df = df.dropna(subset=["true", "meanpred"]).copy()
        df["errbar"] = 0.0

    df["true_lin"]     = np.power(10.0, df["true"])
    df["meanpred_lin"] = np.power(10.0, df["meanpred"])
    df["errbar_lin"]   = df["meanpred_lin"] * np.log(10) * df["errbar"]
    bad = (df["meanpred_lin"] - df["errbar_lin"]) <= 0
    df.loc[bad, "errbar_lin"] = 0.0
    df["x_true"] = df["true_lin"].clip(lower=_EPS)
    df["y_pred"] = df["meanpred_lin"].clip(lower=_EPS)
    df["yerr"]   = df["errbar_lin"]
    return df


def extract_orientation(q):
    q_lower = str(q).lower()
    for orient in ["ram", "nadir", "wake", "zenith"]:
        if orient in q_lower:
            return orient
    return "unknown"


# ---- GPT-4o Random (all splits combined, by orientation) ----
# ---- GPT-4o Random (single combined CSV with orientation column) ----
GPT_RANDOM_CSV = f"{GPT_RANDOM_BASE_DIR}/output_with_orientation.csv"

def load_gpt_random():
    if not os.path.exists(GPT_RANDOM_CSV):
        raise RuntimeError(f"GPT-4o random CSV not found: {GPT_RANDOM_CSV}")
    df = pd.read_csv(GPT_RANDOM_CSV)
    # Columns: question, true, pred1, pred2, meanpred, errbar, lin_true, lin_pred, orientation
    # 'true' and 'meanpred' are log10; reuse the standard cleaner for consistent x_true/y_pred/yerr
    df = load_clean_combined_gpt_from_df(df)
    if "orientation" in df.columns:
        df["orientation"] = df["orientation"].astype(str).str.lower()
    else:
        df["orientation"] = "unknown"
    return df


def load_clean_combined_gpt_from_df(df):
    """Same cleaning as load_clean_combined_gpt but takes an already-loaded df."""
    if "mean pred" in df.columns and "meanpred" not in df.columns:
        df = df.rename(columns={"mean pred": "meanpred"})
    for col in ["true", "meanpred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "errbar" in df.columns:
        df["errbar"] = pd.to_numeric(df["errbar"], errors="coerce")
        df = df.dropna(subset=["true", "meanpred", "errbar"]).copy()
        df["errbar"] = df["errbar"].clip(lower=0)
    else:
        df = df.dropna(subset=["true", "meanpred"]).copy()
        df["errbar"] = 0.0
    df["true_lin"]     = np.power(10.0, df["true"])
    df["meanpred_lin"] = np.power(10.0, df["meanpred"])
    df["errbar_lin"]   = df["meanpred_lin"] * np.log(10) * df["errbar"]
    bad = (df["meanpred_lin"] - df["errbar_lin"]) <= 0
    df.loc[bad, "errbar_lin"] = 0.0
    df["x_true"] = df["true_lin"].clip(lower=_EPS)
    df["y_pred"] = df["meanpred_lin"].clip(lower=_EPS)
    df["yerr"]   = df["errbar_lin"]
    return df


# ---- GPT-4o RG (per-split list) ----
def load_gpt_rg():
    splits = []
    raw_all = []
    for i in range(1, GPT_RG_SPLIT_COUNT + 1):
        p = f"{GPT_RG_BASE_DIR}/split_{i:02d}/combined/RGlogall_4o_split{i}_combined_test.csv"
        if not os.path.exists(p):
            print(f"  [GPT RG] missing: {p}")
            continue
        df = load_clean_combined_gpt(p)
        df["split"] = f"Split {i}"
        splits.append(df)
        raw = pd.read_csv(p)
        if "mean pred" in raw.columns and "meanpred" not in raw.columns:
            raw = raw.rename(columns={"mean pred": "meanpred"})
        raw["true"]     = pd.to_numeric(raw["true"], errors="coerce")
        raw["meanpred"] = pd.to_numeric(raw["meanpred"], errors="coerce")
        raw_all.append(raw.dropna(subset=["true", "meanpred"]))
    return splits, raw_all


# =====================================================
# === GPR DATA LOADING ================================
# =====================================================
def load_gpr_random_wide(path):
    """
    Load GPR random-split wide-format CSV with columns like
    'random split {i}: prediction' and 'random split {i}: uncertainty (log10)'.
    Stacks all N splits into a long dataframe with columns:
      x_true, y_pred, yerr, orientation, true_log, pred_log, split
    """
    df = pd.read_csv(path)
    if GPR_ORIENT_COL not in df.columns:
        raise RuntimeError(f"GPR random file missing '{GPR_ORIENT_COL}' column.")
    if GPR_TRUE_COL not in df.columns:
        raise RuntimeError(f"GPR random file missing '{GPR_TRUE_COL}' column.")

    long_rows = []
    for i in range(1, GPR_RANDOM_N_SPLITS + 1):
        pred_col = GPR_PRED_FMT.format(i=i)
        unc_col  = GPR_UNC_FMT.format(i=i)
        if pred_col not in df.columns:
            print(f"  [GPR rand] missing column: {pred_col}")
            continue
        sub = pd.DataFrame({
            "x_true":      pd.to_numeric(df[GPR_TRUE_COL], errors="coerce"),
            "y_pred":      pd.to_numeric(df[pred_col], errors="coerce"),
            "unc_log10":   pd.to_numeric(df[unc_col], errors="coerce") if unc_col in df.columns else 0.0,
            "orientation": df[GPR_ORIENT_COL].astype(str).str.lower(),
            "split":       f"Split {i}",
        })
        sub = sub.dropna(subset=["x_true", "y_pred"]).copy()
        sub["unc_log10"] = sub["unc_log10"].fillna(0).clip(lower=0)
        sub["yerr"] = sub["y_pred"] * np.log(10) * sub["unc_log10"]
        bad = (sub["y_pred"] - sub["yerr"]) <= 0
        sub.loc[bad, "yerr"] = 0.0
        sub["x_true"] = sub["x_true"].clip(lower=_EPS)
        sub["y_pred"] = sub["y_pred"].clip(lower=_EPS)
        sub["true_log"] = np.log10(sub["x_true"])
        sub["pred_log"] = np.log10(sub["y_pred"])
        long_rows.append(sub)

    if not long_rows:
        raise RuntimeError("No GPR random predictions could be parsed.")
    out = pd.concat(long_rows, ignore_index=True)
    # Normalize orientation labels to the known set
    out["orientation"] = out["orientation"].apply(
        lambda s: next((o for o in ["ram", "nadir", "wake", "zenith"] if o in s), "unknown")
    )
    return out


def load_gpr_rg(path):
    """
    Load GPR RG long-format CSV. All rows share one prediction column
    ('LEO RG split 1: prediction'), and the 'split' column (1-5) tells us
    which split each row belongs to.
    Returns (splits_list, raw_for_metrics_list) mirroring load_gpt_rg.
    """
    df = pd.read_csv(path)
    if GPR_RG_TRUE_COL not in df.columns:
        raise RuntimeError(f"GPR RG file missing '{GPR_RG_TRUE_COL}' column.")
    if "split" not in df.columns:
        raise RuntimeError("GPR RG file missing 'split' column.")

    pred_col = GPR_RG_PRED_COL
    unc_col  = GPR_RG_UNC_COL
    if pred_col not in df.columns:
        raise RuntimeError(f"GPR RG file missing prediction column '{pred_col}'.")

    splits = []
    raw_all = []
    unique_splits = sorted(df["split"].dropna().unique())
    for i, split_val in enumerate(unique_splits, start=1):
        sub_raw = df[df["split"] == split_val].copy()
        sub = pd.DataFrame({
            "x_true":    pd.to_numeric(sub_raw[GPR_RG_TRUE_COL], errors="coerce"),
            "y_pred":    pd.to_numeric(sub_raw[pred_col], errors="coerce"),
            "unc_log10": pd.to_numeric(sub_raw[unc_col], errors="coerce") if unc_col in sub_raw.columns else 0.0,
        })
        sub = sub.dropna(subset=["x_true", "y_pred"]).copy()
        sub["unc_log10"] = sub["unc_log10"].fillna(0).clip(lower=0)
        sub["yerr"] = sub["y_pred"] * np.log(10) * sub["unc_log10"]
        bad = (sub["y_pred"] - sub["yerr"]) <= 0
        sub.loc[bad, "yerr"] = 0.0
        sub["x_true"] = sub["x_true"].clip(lower=_EPS)
        sub["y_pred"] = sub["y_pred"].clip(lower=_EPS)
        sub["split"]  = f"Split {i}"
        splits.append(sub)

        raw_clean = pd.DataFrame({
            "true_log": np.log10(sub["x_true"]),
            "pred_log": np.log10(sub["y_pred"]),
        })
        raw_all.append(raw_clean)
    return splits, raw_all


# =====================================================
# === AXIS FORMATTER ==================================
# =====================================================
def format_ax(ax, xmin, xmax, subtitle=None, show_ylabel=False, show_xlabel=False):
    ax.plot([xmin, xmax], [xmin, xmax], "--", color="k", alpha=0.7, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect("equal", adjustable="box")

    plain = FuncFormatter(lambda v, pos: f"{v:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT)
    ax.grid(True, which="major", linestyle=":")

    if subtitle:
        ax.set_title(subtitle, fontsize=COL_TITLE_FONT, pad=10)
    if show_ylabel:
        ax.set_ylabel(CUSTOM_YLABEL, fontsize=AXIS_LABEL_FONT)
    if show_xlabel:
        ax.set_xlabel(CUSTOM_XLABEL, fontsize=AXIS_LABEL_FONT)


# =====================================================
# === PLOTTING HELPERS ================================
# =====================================================
def plot_random_by_orientation(ax, df, xmin, xmax, subtitle, show_ylabel, show_xlabel):
    for orient, style in ORIENT_STYLE.items():
        sub = df[df["orientation"] == orient]
        if sub.empty:
            continue
        ax.errorbar(
            sub["x_true"], sub["y_pred"], yerr=sub.get("yerr", 0),
            fmt=style["marker"], markersize=6.5, linewidth=1.1,
            color=style["color"], alpha=0.9,
            zorder=3 if orient != "ram" else 2, capsize=3,
        )
    format_ax(ax, xmin, xmax, subtitle=subtitle,
              show_ylabel=show_ylabel, show_xlabel=show_xlabel)


def plot_rg_by_split(ax, split_dfs, xmin, xmax, subtitle, show_ylabel, show_xlabel):
    for i, df in enumerate(split_dfs):
        ax.errorbar(
            df["x_true"], df["y_pred"], yerr=df.get("yerr", 0),
            fmt=RG_MARKERS[i % len(RG_MARKERS)], markersize=6.5, linewidth=1.1,
            color=RG_COLORS[i % len(RG_COLORS)], alpha=0.9,
            zorder=3, capsize=3,
        )
    format_ax(ax, xmin, xmax, subtitle=subtitle,
              show_ylabel=show_ylabel, show_xlabel=show_xlabel)


def add_orientation_metrics(ax, df, true_log_col=None, pred_log_col=None):
    """Add two metric boxes (ram vs other) for an orientation-colored panel.
    Expects df to have log-space columns available; if not, derives from x_true/y_pred.
    """
    if true_log_col is None or pred_log_col is None:
        tlog = np.log10(df["x_true"])
        plog = np.log10(df["y_pred"])
    else:
        tlog = df[true_log_col]
        plog = df[pred_log_col]
    mask_ram = df["orientation"] == "ram"
    mask_oth = ~mask_ram
    if mask_ram.any():
        ome_r = calculate_ome(tlog[mask_ram], plog[mask_ram])
        lr_r  = calculate_log_r2(tlog[mask_ram], plog[mask_ram])
    else:
        ome_r, lr_r = np.nan, np.nan
    if mask_oth.any():
        ome_o = calculate_ome(tlog[mask_oth], plog[mask_oth])
        lr_o  = calculate_log_r2(tlog[mask_oth], plog[mask_oth])
    else:
        ome_o, lr_o = np.nan, np.nan

    ax.text(0.97, 0.03,
            f"Other Orientations:\nOME = {ome_o:.3f}\nLogR\u00b2 = {lr_o:.3f}",
            transform=ax.transAxes, fontsize=METRIC_FONT,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9))
    ax.text(0.97, 0.26,
            f"Ram-Facing:\nOME = {ome_r:.3f}\nLogR\u00b2 = {lr_r:.3f}",
            transform=ax.transAxes, fontsize=METRIC_FONT,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9))


def add_rg_metrics(ax, raw_all):
    if not raw_all:
        return
    comb = pd.concat(raw_all, ignore_index=True)
    tlog = comb["true_log"] if "true_log" in comb.columns else comb["true"]
    plog = comb["pred_log"] if "pred_log" in comb.columns else comb["meanpred"]
    ome = calculate_ome(tlog, plog)
    lr2 = calculate_log_r2(tlog, plog)
    ax.text(0.97, 0.03,
            f"OME = {ome:.3f}\nLogR\u00b2 = {lr2:.3f}",
            transform=ax.transAxes, fontsize=METRIC_FONT,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9))


# =====================================================
# === LOAD ALL DATA ===================================
# =====================================================
print("Loading GPT-4o random...")
gpt_random_df = load_gpt_random()

print("Loading GPT-4o RG...")
gpt_rg_splits, gpt_rg_raw = load_gpt_rg()

print("Loading GPR random...")
gpr_random_df = load_gpr_random_wide(GPR_RANDOM_CSV)
# Derive log columns for metrics
gpr_random_df["true_log"] = np.log10(gpr_random_df["x_true"])
gpr_random_df["pred_log"] = np.log10(gpr_random_df["y_pred"])

print("Loading GPR RG...")
gpr_rg_splits, gpr_rg_raw = load_gpr_rg(GPR_RG_CSV)


# =====================================================
# === BUILD FIGURE ====================================
# =====================================================
fig = plt.figure(figsize=FIGSIZE_IN)
gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.30)

# --- Top-left: GPT-4o Random (full range, by orientation) ---
ax_tl = fig.add_subplot(gs[0, 0])
plot_random_by_orientation(ax_tl, gpt_random_df, FULL_MIN, FULL_MAX,
                           subtitle=RANDOM_TITLE, show_ylabel=True, show_xlabel=False)
add_orientation_metrics(ax_tl, gpt_random_df,
                        true_log_col="true" if "true" in gpt_random_df.columns else None,
                        pred_log_col="meanpred" if "meanpred" in gpt_random_df.columns else None)
ax_tl.text(-0.18, 1.05, "(a)", transform=ax_tl.transAxes,
           fontsize=SUBPLOT_LABEL_FONT, fontweight="bold")

# --- Top-right: GPT-4o RG (zoomed, by split) ---
ax_tr = fig.add_subplot(gs[0, 1])
plot_rg_by_split(ax_tr, gpt_rg_splits, ZOOM_MIN, ZOOM_MAX,
                 subtitle=RG_TITLE, show_ylabel=False, show_xlabel=False)
add_rg_metrics(ax_tr, gpt_rg_raw)
ax_tr.text(-0.18, 1.05, "(b)", transform=ax_tr.transAxes,
           fontsize=SUBPLOT_LABEL_FONT, fontweight="bold")

# --- Bottom-left: GPR Random (full range, by orientation) ---
ax_bl = fig.add_subplot(gs[1, 0])
plot_random_by_orientation(ax_bl, gpr_random_df, FULL_MIN, FULL_MAX,
                           subtitle=None, show_ylabel=True, show_xlabel=True)
add_orientation_metrics(ax_bl, gpr_random_df,
                        true_log_col="true_log", pred_log_col="pred_log")
ax_bl.text(-0.18, 1.05, "(c)", transform=ax_bl.transAxes,
           fontsize=SUBPLOT_LABEL_FONT, fontweight="bold")

# --- Bottom-right: GPR RG (zoomed, by split) ---
ax_br = fig.add_subplot(gs[1, 1])
plot_rg_by_split(ax_br, gpr_rg_splits, ZOOM_MIN, ZOOM_MAX,
                 subtitle=None, show_ylabel=False, show_xlabel=True)
add_rg_metrics(ax_br, gpr_rg_raw)
ax_br.text(-0.18, 1.05, "(d)", transform=ax_br.transAxes,
           fontsize=SUBPLOT_LABEL_FONT, fontweight="bold")


# =====================================================
# === ROW LABELS (GPT-4o / GPR) on far left ===========
# =====================================================
# Vertical row labels on the far left, centered with each row
fig.text(0.04, 0.74, ROW_LABEL_GPT, rotation=90, ha="center", va="center",
         fontsize=ROW_LABEL_FONT, fontweight="bold")
fig.text(0.04, 0.28, ROW_LABEL_GPR, rotation=90, ha="center", va="center",
         fontsize=ROW_LABEL_FONT, fontweight="bold")


# =====================================================
# === LEGENDS =========================================
# =====================================================
# Orientation legend — inside the random column (top-left axis, upper-left corner)
orient_handles = [
    mlines.Line2D([], [], color=st["color"], marker=st["marker"],
                  linestyle="None", markersize=8, label=name)
    for name, st in ORIENT_STYLE.items()
]
ax_tl.legend(handles=orient_handles, loc="upper left",
             fontsize=LEGEND_FONT, frameon=True, title="Orientation",
             title_fontsize=LEGEND_FONT)

# RG split legend — inside the RG column (top-right axis, upper-left corner)
n_rg = max(len(gpt_rg_splits), len(gpr_rg_splits))
rg_handles = [
    mlines.Line2D([], [], color=RG_COLORS[i % len(RG_COLORS)],
                  marker=RG_MARKERS[i % len(RG_MARKERS)],
                  linestyle="None", markersize=8, label=f"Split {i+1}")
    for i in range(n_rg)
]
ax_tr.legend(handles=rg_handles, loc="upper left",
             fontsize=LEGEND_FONT, frameon=True, title="Split",
             title_fontsize=LEGEND_FONT)


# =====================================================
# === MAIN TITLE ======================================
# =====================================================
fig.suptitle("Fine-tuned GPT-4o vs GPR — Random and Restricted Group Splits",
             fontsize=MAIN_TITLE_FONT, y=0.98)

fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.07)

plt.show()


# =====================================================
# === SAVE ============================================
# =====================================================
resp = input("Save combined 2x2 figure? [y/N]: ").strip().lower()
if resp.startswith("y"):
    fig.savefig(EXPORT_PATH, format=EXPORT_FMT, bbox_inches="tight")
    print(f"Saved: {EXPORT_PATH}")
    eps_path = os.path.splitext(EXPORT_PATH)[0] + ".eps"
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"Saved: {eps_path}")
else:
    print("Skipped saving.")
