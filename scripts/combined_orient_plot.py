import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator, NullFormatter, FuncFormatter
import matplotlib.lines as mlines

# =====================================================
# === USER SETTINGS ==================================
# =====================================================

BASE_DIR = "/path/to/results"
SPLIT_COUNT = 10
FIGSIZE_IN = (14, 7)
EXPORT_FMT = "svg"
EXPORT_PATH = os.path.join(BASE_DIR, "combined_all_splits_by_orientation.svg")

# === Manual figure text entries ===
MAIN_TITLE = "All Missions: All Features (Random Split) - By Orientation"
SUBTITLE_LEFT = "Full Range (0.001–100)"
SUBTITLE_RIGHT = "Zoomed Range (0.1–10)"
CUSTOM_XLABEL = r"True Value (Å$\mathregular{^3}$/atom)"
CUSTOM_YLABEL = r"Predicted Value (Å$\mathregular{^3}$/atom)"

# === Fonts and style ===
MAIN_TITLE_FONT = 30
SUB_TITLE_FONT = 24
AXIS_LABEL_FONT = 22
TICK_LABEL_FONT = 18
LEGEND_FONT = 18
METRIC_FONT = 18

# === Layout / spacing tweaks ===
FULL_MIN, FULL_MAX = 0.001, 100
ZOOM_MIN, ZOOM_MAX = 0.1, 10
MAIN_TITLE_Y = 0.93
X_LABEL_Y = 0.035
LEGEND_X = 0.85
SPLIT_LEGEND_Y = 0.60
METRIC_BOX_Y_RAM = 0.33
METRIC_BOX_Y_OTHER = 0.18

# === Orientation color / marker map ===
ORIENT_STYLE = {
    "ram":    {"color": "#1f77b4", "marker": "o"},
    "nadir":  {"color": "#d62728", "marker": "s"},
    "wake":   {"color": "#2ca02c", "marker": "D"},
    "zenith": {"color": "#9467bd", "marker": "^"},
}

# =====================================================
# === LOAD AND CLEAN DATA =============================
# =====================================================
_EPS = 1e-300

def load_clean_combined(path):
    df = pd.read_csv(path)
    if 'mean pred' in df.columns and 'meanpred' not in df.columns:
        df = df.rename(columns={'mean pred': 'meanpred'})
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

all_dfs = []
print("Loading files...")
for i in range(1, SPLIT_COUNT + 1):
    path = f"{BASE_DIR}/split_0{i}/combined/v3AM-AF_Rlogall_4o_split{i}_combined_test.csv"
    if not os.path.exists(path):
        print(f"Warning: Missing {path}")
        continue
    df = load_clean_combined(path)
    all_dfs.append(df)

if not all_dfs:
    print("Error: No files found.")
else:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined {len(all_dfs)} files, {len(combined_df)} rows.")

    # --- Extract orientation from question column ---
    def extract_orientation(q):
        q_lower = str(q).lower()
        for orient in ["ram", "nadir", "wake", "zenith"]:
            if orient in q_lower:
                return orient
        return "unknown"

    combined_df["orientation"] = combined_df["question"].apply(extract_orientation)

    # --- Metric functions ---
    def calculate_ome(y_true_log, y_pred_log):
        return np.mean(np.abs(y_true_log - y_pred_log))

    def calculate_log_r2(y_true_log, y_pred_log):
        ss_res = np.sum(np.square(y_true_log - y_pred_log))
        ss_tot = np.sum(np.square(y_true_log - np.mean(y_true_log)))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - (ss_res / ss_tot)

    # --- Calculate RAM vs Other metrics ---
    df_ram = combined_df[combined_df["orientation"] == "ram"]
    df_other = combined_df[combined_df["orientation"] != "ram"]

    ome_ram = calculate_ome(df_ram["true"], df_ram["meanpred"])
    log_r2_ram = calculate_log_r2(df_ram["true"], df_ram["meanpred"])
    ome_other = calculate_ome(df_other["true"], df_other["meanpred"])
    log_r2_other = calculate_log_r2(df_other["true"], df_other["meanpred"])

    print(f"\nRAM rows: {len(df_ram)}  |  Other rows: {len(df_other)}")
    print(f"RAM  -> OME = {ome_ram:.3f}, LogR² = {log_r2_ram:.3f}")
    print(f"Other -> OME = {ome_other:.3f}, LogR² = {log_r2_other:.3f}")

    METRICS_RAM = (
        f"Ram-Facing:\n"
        f"OME = {ome_ram:.3f}\n"
        f"LogR² = {log_r2_ram:.3f}"
    )
    METRICS_OTHER = (
        f"Other Orientations:\n"
        f"OME = {ome_other:.3f}\n"
        f"LogR² = {log_r2_other:.3f}"
    )

    # --- Save combined CSV ---
    COMBINED_FILE_PATH = os.path.join(BASE_DIR, "ALL_SPLITS_combined_test.csv")
    try:
        combined_df.to_csv(COMBINED_FILE_PATH, index=False)
        print(f"Saved combined data to: {COMBINED_FILE_PATH}")
    except Exception as e:
        print(f"Warning: Could not save CSV. Error: {e}")

    # =====================================================
    # === HELPER FUNCTION TO DRAW PLOTS ===================
    # =====================================================
    def draw_parity_plot(ax, xmin, xmax, subtitle, show_ylabel=False):
        handles, labels = [], []
        for orient, style in ORIENT_STYLE.items():
            subset = combined_df[combined_df["orientation"] == orient]
            if subset.empty:
                continue
            h = ax.errorbar(
                subset["x_true"], subset["y_pred"], yerr=subset["yerr"],
                fmt=style["marker"], markersize=7, linewidth=1.2,
                color=style["color"], alpha=0.9, label=orient,
                zorder=3 if orient != "ram" else 2, capsize=3
            )
            handles.append(h)
            labels.append(orient)

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

        ax.set_title(subtitle, fontsize=SUB_TITLE_FONT, pad=10)
        if show_ylabel:
            ax.set_ylabel(CUSTOM_YLABEL, fontsize=AXIS_LABEL_FONT)
        return handles, labels

    # =====================================================
    # === COMBINED FIGURE =================================
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_IN, sharey=False)

    handles, labels = draw_parity_plot(ax1, FULL_MIN, FULL_MAX, SUBTITLE_LEFT, show_ylabel=True)
    draw_parity_plot(ax2, ZOOM_MIN, ZOOM_MAX, SUBTITLE_RIGHT, show_ylabel=False)

    # Shared X label
    fig.text(0.5, X_LABEL_Y, CUSTOM_XLABEL, ha="center", va="center", fontsize=AXIS_LABEL_FONT)

    # Main title
    fig.suptitle(MAIN_TITLE, fontsize=MAIN_TITLE_FONT, y=MAIN_TITLE_Y)

    # Orientation legend
    orient_handles = []
    for orient, style in ORIENT_STYLE.items():
        orient_handles.append(mlines.Line2D([], [], color=style["color"], marker=style["marker"],
                                             linestyle="None", markersize=8, label=orient))
    fig.legend(
        orient_handles, [h.get_label() for h in orient_handles],
        loc="center left",
        bbox_to_anchor=(LEGEND_X, SPLIT_LEGEND_Y),
        fontsize=LEGEND_FONT,
        frameon=True
    )

    # Metrics boxes
    fig.text(
        LEGEND_X, METRIC_BOX_Y_RAM, METRICS_RAM,
        va="center", ha="left",
        fontsize=METRIC_FONT,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9)
    )
    fig.text(
        LEGEND_X, METRIC_BOX_Y_OTHER, METRICS_OTHER,
        va="center", ha="left",
        fontsize=METRIC_FONT,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9)
    )

    # Adjust subplot spacing
    fig.subplots_adjust(wspace=0.25, right=0.85, top=0.88, bottom=0.15)
    plt.tight_layout(rect=[0, 0.08, 0.85, 0.88])

    plt.show()

    # =====================================================
    # === SAVE OPTION ====================================
    # =====================================================
    resp = input(f"Save combined figure? [y/N]: ").strip().lower()
    if resp.startswith("y"):
        fig.savefig(EXPORT_PATH, format=EXPORT_FMT, bbox_inches="tight")
        print(f"Saved: {EXPORT_PATH}")
        eps_path = os.path.splitext(EXPORT_PATH)[0] + ".eps"
        fig.savefig(eps_path, format="eps", bbox_inches="tight")
        print(f"Saved: {eps_path}")
    else:
        print("Skipped saving.")
