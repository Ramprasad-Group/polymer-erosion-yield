#!/usr/bin/env python3
"""
Unified LEO erosion-yield pipeline: LLM (fine-tuned GPT-4o) and GPR, evaluated on
the identical three split strategies so they are directly comparable. This is a
single self-contained script; all shared logic (splits, features, kernel,
metrics) lives in the "shared core" section below.

MODES
  LLM (OpenAI fine-tuning; DEFAULT_TEMPERATURE is only a fallback -- the tuned
  value in runs/llm_temp.json wins, and it is 0.25):
    temp-tune    sweep temperature {0.1,0.15,0.2,0.25,0.3} x RG 5-fold (reuses one
                 model per fold; temp is inference-only). Writes llm_temp.json.
                 Rerunning is free (every temperature is cached in the llm_cv
                 row store) and rescores with bootstrap CIs.
    epoch-tune   fine-tune EPOCH_GRID x RG 5-fold; the arm matching the epochs
                 llm-cv actually trained its RG models at (read from llm_cv's
                 manifest) is REUSED, so {5,10} costs 5 models, not 10. Picks best
                 epochs by by-row OME/log-R2. Inference on test only,
                 INFER_REPEATS repeats/point.
    llm-cv       final CV: winning epochs on RG(5)+random(10)+variable(10) = 25
                 models, INFER_REPEATS repeats per test point (test only).
    llm-report   read-only: score whatever llm-cv models are already finished in
                 the manifest (partial results while llm-cv still runs). Runs
                 inference on completed splits only; makes API calls.
    llm-prod     1 model on all 201, INFER_REPEATS repeats per point, one parity.
    llm-ablation layers/thickness ablation (baseline / +layers / +thickness),
                 matched-control rows. Built for completeness; costs money to run.

  GPR (predictive sigma; deterministic, no repeats):
    gpr-opt      sweep FP bits {1024,2048} x radius {2,3} x kernel
                 {tanimoto_rbf,tanimoto,rbf}; score by pooled
                 OME/log-R2 on RG 5-fold (variable split reported as secondary).
                 Winner written to gpr_best.json.
    gpr-cv       winning GPR config on RG(5)+random(10)+variable(10) = 25 models,
                 each with sigma; per-split + pooled metrics + separate parity
                 plots (one per split strategy).
    gpr-prod     1 GPR on all 201 + sigma, parity, 10-fold CV generalization.
    gpr-ablation layers/thickness ablation with GPR (matched-control rows).

  Reporting / figures (no API calls, no fitting -- safe to run any number of times):
    plot-only     redraw one saved *_predictions.csv (--pred_csv PATH).
    validation-fig  combined 2 x N figure of the CROSS-VALIDATION results (LLM
                  top row, GPR bottom, one column per split), from saved
                  predictions; --fig_splits selects the columns. This is the
                  validation figure -- NOT production: production is llm-prod /
                  gpr-prod, a single model fit on all 201 rows.
    data-fig      dataset-description figure: per-chemistry sample counts, and AO
                  fluence / erosion yield stacked by orientation.
    tuning-fig    2 x 2: parity for each epoch arm (shared axis range) over the
                  temperature sweep of log-R2 and OME. Requires completed
                  temp-tune and epoch-tune artifacts; refuses to substitute the
                  defaults or fall back to the llm_cv predictions. No API calls.
    gpr-ablation-fig  the same three panels for the GPR ablation, styled
                  identically but as a SEPARATE figure. Reads the per-point
                  predictions gpr-ablation writes; no refitting, no API calls.
    ablation-fig  three parity panels (baseline / +layers / +thickness) for the
                  LLM ablation, styled exactly like the restricted-group panel.
                  Rebuilt from the saved raw repeats; no API calls.
    rebuild-bands recompute each cached row's error band as
                  1 sigma (sd, ddof=1) from the saved raw/rep*.csv repeats, and
                  patch it into the row store and the saved prediction CSVs.
                  Only `band` is written -- truth/pred, and therefore OME and
                  log-R2, are untouched. Reads no dataset, makes no API calls.

Every mode takes --data_csv (the 201-row canonical CSV) and, where restricted-
group is involved, --rg_dir (your provided RG split_* folders).
"""

import os, sys, json, time, argparse, glob, csv, hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Figure fonts: Computer Modern to match the LaTeX manuscript. Family is given
# as a list (not font.family="serif") so matplotlib falls back per glyph --
# cmr10's TTF is missing a few glyphs (angstrom, en/em dash, unicode minus).
# Angstroms go through mathtext (\mathrm{\AA}) and unicode_minus is off so
# those never hit the fallback. fonttype 42 embeds TrueType; publishers reject
# the default Type 3.
# ---------------------------------------------------------------------------
def _font_ladder():
    """Family list filtered to installed fonts, so matplotlib doesn't spam
    findfont warnings for absent families."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    ladder = [n for n in ("CMU Serif", "cmr10", "DejaVu Serif") if n in have]
    return ladder or ["DejaVu Serif"]

PLOT_FONT = {
    "font.family":                 _font_ladder(),
    "mathtext.fontset":            "cm",
    "mathtext.rm":                 "cmr10",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus":          False,
    "pdf.fonttype":                42,   # embed TrueType; Type 3 (the default) is
    "ps.fonttype":                 42,   # rejected by many publishers
}
plt.rcParams.update(PLOT_FONT)


# ================================ shared core ================================

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, WhiteKernel
from sklearn.metrics import r2_score

# ------------------------------------------------------------------- config ---
SEED          = 42
N_RANDOM      = 10          # random 10-fold CV (KFold)
N_VARIABLE    = 10          # variable target-stratified 10-fold CV (StratifiedKFold)
VAR_BINS      = 5           # target-stratification bins for the variable split
COAT_RARE_MIN = 3           # coatings with < this many pool rows -> "other"

# GPR fingerprint / kernel defaults (overridable by the optimization sweep)
FP_BITS       = 2048
FP_RADIUS     = 2
KERNEL        = "tanimoto_rbf"   # tanimoto_rbf | tanimoto | rbf
N_RESTARTS    = 4               # GPR hyperparameter restarts

NUM_COLS      = ["mission_time (yr)", "solar (esh)", "fluence"]   # logged + scaled
ORIENT_LEVELS = ["ram", "zenith", "nadir", "wake", "unknown"]

# ---------------------------------------------------------------------------
# PLOT STYLING -- edit any of these to tune a single text element in isolation.
# Sizes are matplotlib font sizes (points). Titles use {model} and {split}.
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    # font sizes, per element
    "title_size":        18,
    "axis_label_size":   16,
    "tick_label_size":   13,
    "legend_size":       13,
    "legend_title_size": 13,
    "metrics_size":      12,    # the OME / LogR2 / n box
    "colorbar_label_size": 13,
    "colorbar_tick_size":  12,
    # human-readable names shown in titles/labels (not the internal keys)
    "split_display": {"restricted-group": "Restricted-Group Split",
                      "random": "Random Split",
                      "variable": "Variable (Target-Stratified) Split"},
    "model_display": {"llm": "Fine-Tuned GPT-4o", "gpr": "Gaussian Process Regression"},
    # title template: {model} and {split} are filled in; set to "" for no title
    "title_template":  "{model} | {split}",
    "prod_title_template": "{model} | Production (All Data)",
    "xlabel": r"True $E_y$ ($\mathrm{\AA^3/atom}$)",
    "ylabel": r"Predicted $E_y$ ($\mathrm{\AA^3/atom}$)",
    "title_pad": 10,           # gap between title and axes (points)
    # figure-level title used by ablation-fig / tuning-fig / gpr-ablation-fig
    "suptitle_size": 22,
    "suptitle_y":    0.965,    # 1.0 = flush to the top edge; lower = bigger gap
    "suptitle_rect_top": 0.99, # tight_layout reserves above this for the suptitle;
                               # raise it to close the gap under the suptitle
}

# ---------------------------------------------------------------------------
# Font scaling: LaTeX shrinks a figure by textwidth/fig_width_in, so equal
# point sizes render unequal on the page. scaled_style() multiplies every size
# in PLOT_STYLE by fig_width_in/CANON_FIG_WIDTH_IN, which cancels that out --
# text lands the same physical size for any figure width, assuming inclusion at
# width=\textwidth (pass inclusion_fraction otherwise). Only text sizes scale;
# markers/lines are left alone since scaling them changes how the data reads.
# ---------------------------------------------------------------------------
CANON_FIG_WIDTH_IN = 13.0

# ---------------------------------------------------------------------------
# Every multi-panel parity figure is built from the same cell (one column =
# PARITY_PANEL_W_IN x PARITY_PANEL_H_IN, plus a suptitle strip where needed)
# with fonts from scaled_style(fig_w), so at width=\textwidth all figures put
# identical text sizes and square panels on the page. Panels of figures with
# different column counts necessarily differ in size; equal fonts is what is
# enforced.
# ---------------------------------------------------------------------------
PARITY_PANEL_W_IN   = 6.33   # one parity column, inches (matches the reference
                             # validation figure, whose panels are the target look)
PARITY_PANEL_H_IN   = 5.5    # one parity row, inches
PARITY_SUPTITLE_IN  = 0.55   # height of the reserved suptitle strip, inches --
                             # just over the width-scaled suptitle's own text
                             # height (~0.45in), so it clears the middle panel's
                             # title without a dead gap
PARITY_SPLIT_SHORT = {"restricted-group": "RG Split", "random": "Random Split",
                      "variable": "Variable Split"}   # gpr-ablation-fig titles only
# Ablation figures are sized from the AXES out: every panel is a square of
# side PARITY_AXES_IN, col 0 gets extra width for the shared ylabel
# (width_ratios), and rows get title/xlabel allowances (height_ratios) --
# so the panels FILL the columns and the only inter-panel gap left is the
# neighbour's own tick labels.
PARITY_AXES_IN      = 5.0    # square axes side, ablation figures
PARITY_YLAB_IN      = 1.15   # ylabel + y-tick margin, col 0
PARITY_TICK_IN      = 0.50   # y-tick margin, cols 1..2
PARITY_TITLE_IN     = 0.70   # title allowance per row
PARITY_XLAB_IN      = 0.80   # xlabel allowance, bottom row
PARITY_LABEL_PAD_IN = 0.35   # extra per-figure height headroom for the width-
                             # scaled titles/xlabels around the square axes;
                             # without it tight_layout (which mis-measures
                             # set_box_aspect axes) lets the xlabel clip

_SCALED_KEYS = ("title_size", "axis_label_size", "tick_label_size", "legend_size",
                "legend_title_size", "metrics_size", "colorbar_label_size",
                "colorbar_tick_size", "suptitle_size", "title_pad")

def scaled_style(fig_width_in, inclusion_fraction=1.0):
    """PLOT_STYLE with every text size scaled so this figure's text renders at
    the same physical size as a CANON_FIG_WIDTH_IN figure. Returns a NEW dict --
    the global is never mutated, so one figure can never change another's fonts."""
    k = fig_width_in / (CANON_FIG_WIDTH_IN * inclusion_fraction)
    st = dict(PLOT_STYLE)
    for key in _SCALED_KEYS:
        st[key] = PLOT_STYLE[key] * k
    return st

# Restricted-group panels: one marker shape AND one colour per split, so the
# splits stay separable in greyscale and to colour-blind readers. Matches the
# reference figure.
# RG-split panels use CIRCLES for every split (colour alone separates them):
# marker shape is reserved for orientation in the random/variable panels.
SPLIT_MARKERS = ["o"]

def _pretty_title(tag, strategy):
    """Build a readable title from a tag like 'llm_restricted-group' or
    'gpr_random' and a strategy key."""
    model_key = "llm" if tag.startswith("llm") else ("gpr" if tag.startswith("gpr") else "")
    model = PLOT_STYLE["model_display"].get(model_key, model_key.upper())
    split = PLOT_STYLE["split_display"].get(strategy, strategy)
    tmpl = PLOT_STYLE["title_template"]
    return tmpl.format(model=model, split=split) if tmpl else ""


# canonical-CSV -> internal column names
MASTER_REN = {
    "psmiles": "smiles", "e_y (A3/atom)": "e_y (A3/atom)",
    "mission time (yr)": "mission_time (yr)",
    "solar exposure (esh)": "solar (esh)",
    "ao fluence (atoms/cm2)": "fluence",
    "mission name": "mission", "orientation": "orientation",
    "coating name": "coating name", "polymer name": "polymer name",
    "layers": "layers", "thickness (mm)": "thickness (mm)",
}
# RG-export (on-disk) header aliases -> internal names
RG_REN = {
    "smiles1": "smiles", "smiles": "smiles",
    "num_mission_time (yr)": "mission_time (yr)", "mission_time (yr)": "mission_time (yr)",
    "num_ram_solar_exposure (esh)": "solar (esh)", "ram_solar_exposure (esh)": "solar (esh)",
    "num_ram_ao_fluence (atoms/cm2)": "fluence", "ram_ao_fluence (atoms/cm2)": "fluence",
    "cat_mission": "mission", "mission": "mission",
    "cat_orientation": "orientation", "orientation": "orientation",
    "e_y (A3/atom)": "e_y (A3/atom)", "coating name": "coating name",
    "polymer name": "polymer name",
}

# ---------------------------------------------------------------- data load ---
def load_master_csv(path):
    """Load the canonical 201-row dataset; apply the one real data fix (an
    ITO-coated-silver-Teflon row had its coating in the polymer name with the
    coating cell blank). No rows dropped/merged/averaged."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=MASTER_REN)
    df["e_y (A3/atom)"] = pd.to_numeric(df.get("e_y (A3/atom)"), errors="coerce")
    ito = (df["polymer name"].str.contains("indium tin oxide coated silver",
                                            case=False, na=False)
           & df["coating name"].isna())
    df.loc[ito, "coating name"] = "indium tin oxide coated silver"
    return df.reset_index(drop=True)

def load_rg_dir(rg_dir):
    """Return [(train_df, test_df)] for the provided restricted-group split_*
    folders, header-normalized to the internal schema."""
    def _load(p):
        d = pd.read_csv(p)
        d.columns = [c.strip() for c in d.columns]
        # apply master-schema aliases first (final-dataset headers: psmiles,
        # 'mission time (yr)', ...), then RG-export aliases (smiles1, num_*),
        # so RG files written in EITHER schema normalize to internal names.
        ren = {**MASTER_REN, **RG_REN}
        return d.rename(columns={k: v for k, v in ren.items() if k in d.columns})
    out = []
    for sd in sorted(glob.glob(os.path.join(rg_dir, "split_*"))):
        tr = glob.glob(os.path.join(sd, "train*.csv"))
        te = glob.glob(os.path.join(sd, "test*.csv"))
        if tr and te:
            out.append((_load(tr[0]), _load(te[0])))
    if not out:
        raise FileNotFoundError(f"no split_* folders with train*/test* under {rg_dir}")
    return out

def report_collisions(pool, delta=0.15):
    """Informational only: print rows sharing a feature vector and their target
    spread. Never modifies pool or any split."""
    y = target(pool)
    c = _canon_coating_series(pool["coating name"], pool["polymer name"])
    key = pd.DataFrame({
        "smiles": pool["smiles"],
        "lflu": np.log10(pd.to_numeric(pool["fluence"], errors="coerce")).round(6),
        "lsol": np.log10(pd.to_numeric(pool["solar (esh)"], errors="coerce")).round(6),
        "lmt":  np.log10(pd.to_numeric(pool["mission_time (yr)"], errors="coerce")).round(6),
        "orient": pool["orientation"].astype(str).str.strip().str.lower(),
        "coat": c,
    })
    groups = [(k, idx) for k, idx in key.groupby(list(key.columns)).groups.items()
              if len(idx) > 1]
    n_rows = sum(len(i) for _, i in groups)
    print(f"collision report: {len(groups)} groups share a feature vector "
          f"({n_rows} of {len(pool)} rows). Informational only; no rows changed.")
    worst = sorted(((y[idx].max() - y[idx].min(), len(idx),
                     pool.loc[idx, 'polymer name'].tolist()) for _, idx in groups),
                   reverse=True)[:5]
    for spread, n, names in worst:
        flag = " <-- large spread" if spread > delta else ""
        print(f"  spread={spread:.3f}  n={n}  {names}{flag}")

# ------------------------------------------------------------------ target ----
def target(df):
    return np.log10(pd.to_numeric(df["e_y (A3/atom)"], errors="coerce").to_numpy(float))

# ------------------------------------------------------------------ splits ----
def random_splits(pool, n=N_RANDOM):
    # K-fold: every row is in exactly ONE test fold -> pooled n == len(pool).
    kf = KFold(n_splits=n, shuffle=True, random_state=SEED)
    return [(pool.iloc[tr], pool.iloc[te]) for tr, te in kf.split(pool)]

def variable_splits(pool, n=N_VARIABLE, bins=VAR_BINS):
    # Stratified K-fold on target quantile bins: balanced target per fold AND
    # every row tested exactly once.
    y = target(pool)
    b = pd.qcut(y, bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED)
    return [(pool.iloc[tr], pool.iloc[te]) for tr, te in skf.split(pool, b)]



def _dataset_hash(data_csv):
    """md5 of the runtime dataset file -- ties cached results to exact data."""
    try:
        return hashlib.md5(open(data_csv, "rb").read()).hexdigest()
    except Exception:
        return None

def _membership_hash(out_dir, strategy):
    """md5 of the frozen split-membership file for this strategy (or None)."""
    p = os.path.join(out_dir, f"{strategy}_split_membership.csv")
    if not os.path.exists(p):
        return None
    return hashlib.md5(open(p, "rb").read()).hexdigest()

def _rg_split_hash(rg_dir):
    """Stable md5 over BOTH train and test files of every RG split folder
    (sorted), including each file's relative name and byte length as delimiters,
    so RG cached GPR results are tied to the exact five-fold RG definition --
    training data included, since GPR predictions depend on it. None if absent."""
    try:
        h = hashlib.md5()
        folders = sorted(glob.glob(os.path.join(rg_dir, "split_*")))
        got_any = False
        for sd in folders:
            for pat in ("train*.csv", "test*.csv"):
                for f in sorted(glob.glob(os.path.join(sd, pat))):
                    data = open(f, "rb").read()
                    rel = os.path.relpath(f, rg_dir).encode()
                    h.update(rel); h.update(b"\x00")
                    h.update(str(len(data)).encode()); h.update(b"\x00")
                    h.update(data); h.update(b"\x00")
                    got_any = True
        return h.hexdigest() if got_any else None
    except Exception:
        return None

def _gpr_cache_valid(run_dir, strategy, data_csv, out_dir, bits, radius, kernel,
                     rg_dir=None):
    """True only if saved gpr_<strategy>_predictions.csv is a valid current
    result: exists, provenance matches the ACTUAL optimized GPR config, dataset
    hash and membership/RG-split hash match, expected fold count, per-fold row
    counts, and strategy+fold IDs all match. Any error or mismatch -> False so
    the caller safely recomputes (never crashes on a malformed cache)."""
    pred = os.path.join(run_dir, f"gpr_{strategy}_predictions.csv")
    prov = os.path.join(run_dir, f"gpr_{strategy}_provenance.json")
    if not (os.path.exists(pred) and os.path.exists(prov)):
        return False
    try:
        pv = json.load(open(prov)); df = pd.read_csv(pred)
        # all required columns must be present
        required = ["strategy", "split", "truth", "pred", "band", "orientation", "fluence"]
        if any(c not in df.columns for c in required):
            return False
        if df.empty:
            return False
        # truth/pred/band must be numeric and finite
        for c in ("truth", "pred", "band"):
            vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
            if not np.all(np.isfinite(vals)):
                return False
        # config match (actual optimized bits/radius/kernel)
        if (pv.get("fp_bits") != bits or pv.get("fp_radius") != radius
                or pv.get("kernel") != kernel):
            return False
        # dataset hash
        if pv.get("data_hash") != _dataset_hash(data_csv):
            return False
        # split-definition hash: membership file (random/variable) or RG files (rg)
        if strategy in ("random", "variable"):
            if pv.get("membership_hash") != _membership_hash(out_dir, strategy):
                return False
        elif strategy == "restricted-group":
            if pv.get("rg_split_hash") != _rg_split_hash(rg_dir):
                return False
        # strategy present and only itself
        if set(str(x) for x in df["strategy"].unique()) != {strategy}:
            return False
        # expected fold IDs 0..n-1
        expected = {"restricted-group": 5, "random": N_RANDOM,
                    "variable": N_VARIABLE}[strategy]
        folds = sorted(int(x) for x in df["split"].unique())
        if folds != list(range(expected)):
            return False
        # per-fold row counts
        got_counts = {int(k): int(v) for k, v in df.groupby("split").size().to_dict().items()}
        if strategy in ("random", "variable"):
            mem_path = os.path.join(out_dir, f"{strategy}_split_membership.csv")
            if os.path.exists(mem_path):
                mem = pd.read_csv(mem_path)
                exp_counts = {int(k): int(v) for k, v in mem.groupby("fold").size().to_dict().items()}
                if exp_counts != got_counts:
                    return False
        elif strategy == "restricted-group" and rg_dir:
            rg = load_rg_dir(rg_dir)
            exp_counts = {k: len(te) for k, (_, te) in enumerate(rg)}
            if exp_counts != got_counts:
                return False
    except Exception:
        return False
    return True

def _split_store_path(out_dir, strategy):
    return os.path.join(out_dir, f"{strategy}_split_membership.csv")

def _save_split_membership(splits, pool, out_dir, strategy):
    """Freeze a split to disk as (fold, row_index) using the exact master-row
    position, so it is reproducible on every future run with zero identity
    ambiguity -- no dependence on sklearn versions or on smiles+target keys."""
    os.makedirs(out_dir, exist_ok=True)
    pool = pool.reset_index(drop=True)
    # map each row's identity back to its master index
    idx_of = {id(pool.iloc[i]): i for i in range(len(pool))}
    rows = []
    for k, (_, te) in enumerate(splits):
        for pos in te.index:
            rows.append({"fold": k, "row_index": int(pos)})
    pd.DataFrame(rows).to_csv(_split_store_path(out_dir, strategy), index=False)

def _load_split_membership(pool, out_dir, strategy):
    """Rebuild splits from a frozen membership file that stores the exact
    row_index of every test row per fold. row_index refers to the position in
    the master dataset, so identity is exact (no smiles+target ambiguity for
    rows that share those values). Returns list of (train_df, test_df) or None."""
    path = _split_store_path(out_dir, strategy)
    if not os.path.exists(path):
        return None
    mem = pd.read_csv(path)
    pool = pool.reset_index(drop=True)
    n_folds = int(mem["fold"].max()) + 1
    splits = []
    for k in range(n_folds):
        te_idx = mem[mem["fold"] == k]["row_index"].astype(int).tolist()
        tr_idx = [i for i in range(len(pool)) if i not in set(te_idx)]
        splits.append((pool.iloc[tr_idx], pool.iloc[te_idx]))
    return splits

def generate_rg_splits(pool, n=5):
    """Rebuild the restricted-group folds from the dataset alone.

    The RG test set is the set of SINGLETON chemistries -- those with exactly one
    row -- so every test chemistry is entirely unseen in training. They are
    shuffled with seed SEED and dealt into n contiguous groups:

        singles = chemistries with exactly one row      (29 of 55 here)
        perm    = np.random.default_rng(SEED).permutation(singles)
        folds   = np.array_split(perm, n)               (6/6/6/6/5)

    Verified to reproduce the provided rg/split_* folders EXACTLY -- all five
    folds, in order, at the row level. This makes RG reproducible from the CSV
    the same way random/variable are, so the pipeline can be re-run end to end
    from the dataset alone; the rg/ files remain authoritative when present."""
    counts = pool["smiles"].value_counts()
    singles = list(counts[counts == 1].index)
    perm = np.random.default_rng(SEED).permutation(singles)
    out = []
    for chunk in np.array_split(perm, n):
        te = pool[pool["smiles"].isin(set(chunk))]
        tr = pool[~pool["smiles"].isin(set(chunk))]
        out.append((tr, te))
    return out

def get_splits(strategy, pool, rg_dir=None, out_dir="runs"):
    if strategy == "restricted-group":
        # the provided rg/ folders are authoritative when present (they are what
        # the trained models used); regenerate from the pool only if absent.
        if rg_dir and glob.glob(os.path.join(rg_dir, "split_*")):
            return load_rg_dir(rg_dir)
        print("restricted-group: no rg/split_* folders found -- regenerating the "
              "folds from the dataset (singleton chemistries, seed 42).")
        return generate_rg_splits(pool)
    # random / variable: if a frozen membership file exists, load it (exact,
    # reproducible, version-independent). Otherwise generate deterministically
    # and freeze it to disk so all future runs are identical.
    frozen = _load_split_membership(pool, out_dir, strategy)
    if frozen is not None:
        return frozen
    if strategy == "random":
        splits = random_splits(pool)
    elif strategy == "variable":
        splits = variable_splits(pool)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    _save_split_membership(splits, pool, out_dir, strategy)
    return splits

STRATEGIES = ["restricted-group", "random", "variable"]

# ---------------------------------------------------------------- coating -----
def _canon_coating_series(coat_raw, polymer_name):
    c = coat_raw.fillna("none").astype(str).str.strip().str.lower()
    c = c.replace({"?": "none", "": "none", "nan": "none"})
    ito = polymer_name.str.contains("indium tin oxide coated silver",
                                    case=False, na=False) & (c == "none")
    c = c.where(~ito, "indium tin oxide coated silver")
    c = (c.str.replace("coated-silver", "coated silver", regex=False)
           .str.replace("paint*", "paint", regex=False)
           .str.replace("back-surface", "back surface", regex=False))
    merge = {
        "indium tin oxide coated": "indium tin oxide",
        "back surface carbon-painted": "back surface carbon paint",
        "back surface spray painted with bbq black (carbon) paint": "back surface carbon paint",
        "back surface silver": "back surface silver-based",
        "back surface silver/niobium": "back surface silver-based",
    }
    return c.replace(merge)

def coating_levels(pool):
    c = _canon_coating_series(pool["coating name"], pool["polymer name"])
    vc = c.value_counts()
    return sorted(set(vc[vc >= COAT_RARE_MIN].index)) + ["other"]

def coating_onehot(df, levels):
    c = _canon_coating_series(df["coating name"], df["polymer name"])
    c = c.where(c.isin(levels), "other")
    oh = np.zeros((len(df), len(levels)), np.float32)
    idx = {l: j for j, l in enumerate(levels)}
    for i, v in enumerate(c):
        oh[i, idx[v]] = 1.0
    return oh

def orient_onehot(df):
    o = df["orientation"].astype(str).str.strip().str.lower()
    o = o.where(o.isin(ORIENT_LEVELS), "unknown")
    oh = np.zeros((len(df), len(ORIENT_LEVELS)), np.float32)
    for i, v in enumerate(o):
        oh[i, ORIENT_LEVELS.index(v)] = 1.0
    return oh

# ------------------------------------------------------------ fingerprints ----
def _loop_mol(s):
    from rdkit import Chem
    s = str(s)
    if s == "[*]C[*]":
        s = "[*]CCC[*]"
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    dummy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy) != 2:
        return mol
    rw = Chem.RWMol(mol)
    nbr = []
    for d in dummy:
        ns = [n.GetIdx() for n in rw.GetAtomWithIdx(d).GetNeighbors()]
        if len(ns) != 1:
            return None
        nbr.append(ns[0])
    if rw.GetBondBetweenAtoms(nbr[0], nbr[1]) is None:
        rw.AddBond(nbr[0], nbr[1], Chem.BondType.SINGLE)
    for d in sorted(dummy, reverse=True):
        rw.RemoveAtom(d)
    looped = rw.GetMol()
    try:
        Chem.SanitizeMol(looped)
    except Exception:
        return None
    return looped

def morgan_fp(smiles, bits=None, radius=None):
    """(n, bits) float32 loop-closed Morgan fingerprints; NaN row = invalid mol."""
    from rdkit.Chem import rdFingerprintGenerator
    bits = FP_BITS if bits is None else bits
    radius = FP_RADIUS if radius is None else radius
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    X = np.full((len(smiles), bits), np.nan, np.float32)
    for i, s in enumerate(smiles):
        m = _loop_mol(s)
        if m is not None:
            X[i] = gen.GetFingerprintAsNumPy(m)
    return X

# ------------------------------------------------------------- feature build --
def build_blocks(df, coat_lvls, bits=None, radius=None):
    """Return (fp, num_raw, cat) blocks. num_raw = log10 numerics (unscaled);
    scaling is fit per-split downstream to avoid leakage."""
    fp = morgan_fp(df["smiles"].tolist(), bits=bits, radius=radius)
    num = np.empty((len(df), len(NUM_COLS)), float)
    for j, c in enumerate(NUM_COLS):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if (v <= 0).any() or not np.isfinite(v).all():
            bad = np.where(~np.isfinite(v) | (v <= 0))[0].tolist()
            raise ValueError(f"{c}: non-positive/non-finite rows {bad}")
        num[:, j] = np.log10(v)
    cat = np.hstack([coating_onehot(df, coat_lvls), orient_onehot(df)])
    return fp, num, cat

# ----------------------------------------------------------------- kernel -----
class TanimotoRBF(Kernel):
    """fp_amp * Tanimoto(X_fp) + rbf_amp * RBF_l(X_num). First n_fp columns are
    the fingerprint (Tanimoto); the rest are RBF inputs. Set mode to drop a term:
    'both' (default), 'tanimoto' (fp only), 'rbf' (numeric+onehot only)."""
    def __init__(self, n_fp, mode="both", fp_amp=1.0, rbf_amp=1.0, length_scale=1.0,
                 fp_amp_bounds=(1e-3, 1e3), rbf_amp_bounds=(1e-3, 1e3),
                 length_scale_bounds=(1e-2, 1e2)):
        self.n_fp = n_fp; self.mode = mode
        self.fp_amp = fp_amp; self.rbf_amp = rbf_amp; self.length_scale = length_scale
        self.fp_amp_bounds = fp_amp_bounds
        self.rbf_amp_bounds = rbf_amp_bounds
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_fp_amp(self):
        return Hyperparameter("fp_amp", "numeric", self.fp_amp_bounds)
    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)
    @property
    def hyperparameter_rbf_amp(self):
        return Hyperparameter("rbf_amp", "numeric", self.rbf_amp_bounds)

    def _split(self, X):
        return np.asarray(X[:, :self.n_fp], float), np.asarray(X[:, self.n_fp:], float)

    def _tanimoto(self, A, B):
        AB = A @ B.T
        sa = np.einsum("ij,ij->i", A, A); sb = np.einsum("ij,ij->i", B, B)
        den = sa[:, None] + sb[None, :] - AB
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, AB / np.where(den == 0, 1.0, den), 0.0)

    def __call__(self, X, Y=None, eval_gradient=False):
        Xf, Xn = self._split(X)
        if Y is None:
            Yf, Yn = Xf, Xn
        else:
            if eval_gradient:
                raise ValueError("gradient only for Y is None")
            Yf, Yn = self._split(Y)
        use_fp = self.mode in ("both", "tanimoto")
        use_rbf = self.mode in ("both", "rbf")
        Kt = self._tanimoto(Xf, Yf) if use_fp else np.zeros((len(Xf), len(Yf)))
        if use_rbf:
            an = np.einsum("ij,ij->i", Xn, Xn); bn = np.einsum("ij,ij->i", Yn, Yn)
            D2 = np.maximum(an[:, None] + bn[None, :] - 2.0 * Xn @ Yn.T, 0.0)
            Kr = np.exp(-0.5 * D2 / self.length_scale ** 2)
        else:
            D2 = np.zeros((len(Xn), len(Yn))); Kr = np.zeros_like(D2)
        K = self.fp_amp * Kt + self.rbf_amp * Kr
        if not eval_gradient:
            return K
        g_fp = (self.fp_amp * Kt)[:, :, None]
        g_l = (self.rbf_amp * Kr * (D2 / self.length_scale ** 2))[:, :, None]
        g_rbf = (self.rbf_amp * Kr)[:, :, None]
        return K, np.dstack([g_fp, g_l, g_rbf])

    def diag(self, X):
        Xf, _ = self._split(X)
        sa = np.einsum("ij,ij->i", Xf, Xf)
        d = np.zeros(len(X))
        if self.mode in ("both", "tanimoto"):
            d = d + self.fp_amp * (sa > 0).astype(float)
        if self.mode in ("both", "rbf"):
            d = d + self.rbf_amp * 1.0
        return d

    def is_stationary(self):
        return False

def make_gpr(n_fp, kernel=KERNEL, n_restarts=N_RESTARTS):
    mode = {"tanimoto_rbf": "both", "tanimoto": "tanimoto", "rbf": "rbf"}[kernel]
    k = TanimotoRBF(n_fp, mode=mode) + WhiteKernel(noise_level=0.1,
                                                   noise_level_bounds=(1e-5, 1e1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=n_restarts,
                                    alpha=1e-8, random_state=SEED)

# ----------------------------------------------------------------- metrics ----
def metrics(y_true, y_pred):
    """OME (order-of-magnitude error == MAE on log10 target) + log-R2."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ome = float(np.abs(y_true - y_pred).mean())
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return ome, r2

# ---- row-bootstrap uncertainty on the panel metrics --------------------------
# Both OME and log-R2 get a percentile CI from the same row resamples, computed
# identically for both models. The intervals are per-panel and NOT paired
# (aggregate prediction CSVs carry no row identifier, so pairing can't be
# verified) -- overlapping CIs are not a test of no difference between models.
# Per-point error bars (LLM decode sd, GPR posterior sd) are separate from this;
# only row-level mean predictions are resampled.
METRIC_CI = {
    "B":                10000,   # resamples
    "seed":             42,
    "level":            95,      # percentile interval
    "max_invalid_frac": 0.01,    # refuse if more than this fraction degenerate
}

def _boot_idx(n):
    """Resample indices for a panel of n rows. Deterministic in (n, seed)."""
    return np.random.default_rng(METRIC_CI["seed"]).integers(
        0, n, (METRIC_CI["B"], n))

def metric_ci(y_true, y_pred):
    """Point estimates + row-bootstrap percentile CIs for OME and log-R2.
    Returns (ome, (ome_lo, ome_hi), r2, (r2_lo, r2_hi), row_err_sd)."""
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    n = len(yt)
    ome, r2 = metrics(yt, yp)
    err = np.abs(yt - yp)
    sd = float(np.std(err, ddof=1)) if n > 1 else 0.0
    if n < 3:
        return ome, (np.nan, np.nan), r2, (np.nan, np.nan), sd
    idx = _boot_idx(n)
    T = yt[idx]; P = yp[idx]
    b_ome = np.abs(T - P).mean(1)
    ss_res = ((T - P) ** 2).sum(1)
    ss_tot = ((T - T.mean(1, keepdims=True)) ** 2).sum(1)
    # a resample can draw rows with (near) zero spread in truth -> SStot ~ 0 ->
    # R2 explodes. Drop those; refuse the interval if too many.
    ok = ss_tot > 1e-12
    frac_bad = 1.0 - ok.mean()
    if frac_bad > METRIC_CI["max_invalid_frac"]:
        sys.exit(f"metric_ci: {frac_bad:.1%} of bootstrap resamples had ~zero variance "
                 f"in truth (n={n}); the CI would be meaningless. Refusing.")
    b_r2 = 1.0 - ss_res[ok] / ss_tot[ok]
    lo, hi = (100 - METRIC_CI["level"]) / 2, 100 - (100 - METRIC_CI["level"]) / 2
    return (ome, (float(np.percentile(b_ome, lo)), float(np.percentile(b_ome, hi))),
            r2, (float(np.percentile(b_r2, lo)), float(np.percentile(b_r2, hi))), sd)

def metric_row(y_true, y_pred, **extra):
    """One metric_ci() result as a CSV row, in the schema every reported metrics
    file in this pipeline uses (ablation_results.csv, *_cv_summary.csv, the
    tuning CSVs, production). Sec 1.2: every reported metric carries a row-
    bootstrap 95% percentile CI plus the bootstrap metadata needed to reproduce
    it. OME_std_rows rides along as a diagnostic -- it is the dispersion of the
    individual row errors, NOT an uncertainty of the estimate, and nothing plots
    it beside one."""
    ome, ome_ci, r2, r2_ci, sd = metric_ci(y_true, y_pred)
    row = dict(OME=ome, OME_ci_lo=ome_ci[0], OME_ci_hi=ome_ci[1],
               logR2=r2, logR2_ci_lo=r2_ci[0], logR2_ci_hi=r2_ci[1],
               n_rows=int(len(np.asarray(y_true))), OME_std_rows=sd,
               boot_B=METRIC_CI["B"], boot_seed=METRIC_CI["seed"],
               ci_level=METRIC_CI["level"])
    row.update(extra)
    return row

def metric_box(ome, ome_ci, r2, r2_ci, n, prefix=""):
    """The metrics-box text. Brackets, not +/-: percentile CIs are asymmetric."""
    return (f"{prefix}OME = {ome:.3f} [{ome_ci[0]:.3f}, {ome_ci[1]:.3f}]\n"
            f"{prefix}LogR$^2$ = {r2:.3f} [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]\n"
            f"n = {n} rows")

# =================== end shared core ===================


# ------------------------------------------------------------------- config ---
DEFAULT_TEMPERATURE = 0.2       # FALLBACK ONLY. The temp-tune winner is 0.25 and
                                # lives in runs/llm_temp.json, which _best_temp prefers.
TEMP_GRID     = [0.1, 0.15, 0.2, 0.25, 0.3]
EPOCH_GRID    = [5, 10]
DEFAULT_EPOCHS = 5              # used by llm-cv/prod/ablation when epoch-tune not run
INFER_REPEATS = 10
BASE_MODEL    = "gpt-4o-2024-08-06"
MAX_ACTIVE_JOBS = 3
POLL_SECS     = 60
# Retry budgets for fine-tune submission. 0 = unlimited (keep trying until the
# endpoint accepts) -- useful when the daily fine-tune quota is shared and you
# want to claim a slot the moment one frees up. See the backoff ceilings below.
MAX_TRANSIENT_TRIES = 8    # 5xx / connection / timeout
MAX_RATELIMIT_TRIES = 20   # cap-hit on submit (whether or not jobs are active)
# Backoff ceilings. Both ramp 60s -> 120 -> 240 -> ... then hold at the cap
# indefinitely (with --retry_tries 0). A 5xx usually clears in minutes, so its
# ceiling is 10 min. A daily job-quota takes hours to reset, so its ceiling is
# 4 h -- the ramp probes often early in case the cap-hit is momentary, then
# settles to a 4 h cadence for as long as it takes.
TRANSIENT_BACKOFF_CAP = 600       # 10 min
RATELIMIT_BACKOFF_CAP = 4 * 3600  # 4 h
_TERMINAL     = {"succeeded", "failed", "cancelled"}

GPR_BITS_GRID   = [1024, 2048]
GPR_RADIUS_GRID = [2, 3]
GPR_KERNEL_GRID = ["tanimoto_rbf", "tanimoto", "rbf"]
GPR_CV_FOLDS    = 10        # production-model generalization CV

# ===========================================================================
# ===  LLM PROMPTING  =======================================================
# ===========================================================================
_RULES = (
    " Rules: 1) Use only the provided input fields. 2) Keep reasoning internal; "
    "do not explain your steps. 3) Do not include text, labels, or units. 4) Output "
    "only the final numeric value with 3 significant figures. 5) Negative values are "
    "allowed. 6) If prediction is not possible, output exactly null."
)
_PREDICT = (" predict the base-10 logarithm of the atomic oxygen erosion yield in "
            "(Angstroms^3/atom).")
# --- system/user heads: base (no layers/thickness), and the two ablation adds ---
_SYS_HEAD = ("You are a materials scientist specializing in low Earth orbit atomic "
             "oxygen interactions with polymers. Given a structured input describing a "
             "polymer sample, including the polymer name, the SMILES string, a "
             "description of the coating, ")
_SYS_TAIL = ("the orientation of the sample during space exposure, the mission time "
             "(years of direct space exposure while attached to the ISS), the solar "
             "exposure (equivalent sun hours) and the atomic oxygen fluence "
             "(atoms/cm^2),")
SYSTEM = {
    "base":      _SYS_HEAD + _SYS_TAIL + _PREDICT + _RULES,
    "layers":    _SYS_HEAD + "the number of stacked thin-film layers, " + _SYS_TAIL + _PREDICT + _RULES,
    "thickness": _SYS_HEAD + "the film thickness in millimeters, " + _SYS_TAIL + _PREDICT + _RULES,
}
USER = {
    "base": ("What is the base-10 logarithm of the atomic oxygen erosion yield of the "
             "polymer {} represented by SMILES {}, with {} coating, oriented in the {} "
             "direction for a mission time of {} years, subjected to a solar exposure of "
             "{} equivalent sun hours and an atomic oxygen fluence of {} atom/cm^2?"),
    "layers": ("What is the base-10 logarithm of the atomic oxygen erosion yield of the "
               "polymer {} represented by SMILES {}, with {} coating, composed of {} "
               "stacked thin-film layers, oriented in the {} direction for a mission time "
               "of {} years, subjected to a solar exposure of {} equivalent sun hours and "
               "an atomic oxygen fluence of {} atom/cm^2?"),
    "thickness": ("What is the base-10 logarithm of the atomic oxygen erosion yield of the "
                  "polymer {} represented by SMILES {}, with {} coating, with a film "
                  "thickness of {} mm, oriented in the {} direction for a mission time of "
                  "{} years, subjected to a solar exposure of {} equivalent sun hours and "
                  "an atomic oxygen fluence of {} atom/cm^2?"),
}

def _coat_text(row):
    c = row.get("coating name", None)
    if c is None or (isinstance(c, float) and pd.isna(c)) or str(c).strip() == "":
        return "no"
    return str(c)

def row_to_record(row, variant="base"):
    name = row.get("polymer name", "unknown polymer")
    smiles, orient = row["smiles"], row["orientation"]
    mtime = row["mission_time (yr)"]
    solar = round(float(row["solar (esh)"]))
    flu = f"{float(row['fluence']):.3e}"
    y = round(float(target(pd.DataFrame([row]))[0]), 3)
    coat = _coat_text(row)
    if variant == "layers":
        lay = row.get("layers", None)
        lay = "an unspecified number of" if pd.isna(lay) else str(int(float(lay)))
        user = USER["layers"].format(name, smiles, coat, lay, orient, mtime, solar, flu)
    elif variant == "thickness":
        thk = row.get("thickness (mm)", None)
        thk = "unspecified" if pd.isna(thk) else f"{float(thk):g}"
        user = USER["thickness"].format(name, smiles, coat, thk, orient, mtime, solar, flu)
    else:
        user = USER["base"].format(name, smiles, coat, orient, mtime, solar, flu)
    return {"messages": [
        {"role": "system", "content": SYSTEM[variant]},
        {"role": "user", "content": user},
        {"role": "assistant", "content": str(y)},
    ]}

def write_jsonl(df, path, variant="base"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(row_to_record(r, variant)) + "\n")

# ===========================================================================
# ===  OpenAI fine-tune orchestration  ======================================
# ===========================================================================
def _resolve_api_key():
    if os.path.exists("openai_api_key.txt"):
        k = open("openai_api_key.txt").read().strip()
        if k:
            return k
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    raise RuntimeError("No OpenAI API key (openai_api_key.txt or OPENAI_API_KEY).")

def _client():
    from openai import OpenAI
    return OpenAI(api_key=_resolve_api_key())

def _manifest_path(run_dir):
    return os.path.join(run_dir, "manifest.json")

def _load_manifest(run_dir):
    p = _manifest_path(run_dir)
    return json.load(open(p)) if os.path.exists(p) else {"units": {}}

def _save_manifest(run_dir, m):
    os.makedirs(run_dir, exist_ok=True)
    json.dump(m, open(_manifest_path(run_dir), "w"), indent=2)

def _submit(client, run_dir, m, unit, epochs):
    from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
    tj = os.path.join(run_dir, unit, "train.jsonl")
    if not os.path.exists(tj):
        print(f"  {unit}: missing {tj}"); return False
    rec = m["units"].setdefault(unit, {})
    try:
        # reuse a previously-uploaded training file if we already have one
        file_id = rec.get("train_file_id")
        if not file_id:
            up = client.files.create(file=open(tj, "rb"), purpose="fine-tune")
            file_id = up.id
            rec["train_file_id"] = file_id
            _save_manifest(run_dir, m)
        job = client.fine_tuning.jobs.create(
            training_file=file_id, model=BASE_MODEL,
            method={"type": "supervised",
                    "supervised": {"hyperparameters": {"n_epochs": epochs}}})
    except RateLimitError:
        print(f"  {unit}: cap hit; retry as slots free."); return "ratelimit"
    except (APIConnectionError, APITimeoutError, InternalServerError) as e:
        # transient: connection/timeout/5xx -> caller may retry with backoff
        print(f"  {unit}: transient submit error ({type(e).__name__}); will retry.")
        return "transient"
    except Exception as e:
        # non-transient (4xx / invalid request / auth / config): fail fast
        print(f"  {unit}: submit failed permanently ({type(e).__name__}: {e}).")
        return "error"
    rec.update({"train_file_id": file_id, "job_id": job.id, "status": job.status, "epochs": epochs})
    _save_manifest(run_dir, m)
    print(f"  {unit}: submitted {job.id}")
    return True

def _poll(client, run_dir, m, unit):
    rec = m["units"][unit]
    job = client.fine_tuning.jobs.retrieve(rec["job_id"])
    rec["status"] = job.status
    if job.status == "succeeded":
        rec["fine_tuned_model"] = job.fine_tuned_model
        print(f"  {unit}: SUCCEEDED -> {job.fine_tuned_model}")
    elif job.status in ("failed", "cancelled"):
        print(f"  {unit}: {job.status.upper()} {getattr(job, 'error', '')}")
    _save_manifest(run_dir, m)
    return job.status

def orchestrate(run_dir, units_epochs, wait=True):
    """units_epochs: dict unit_name -> epochs. Cap-aware submit+poll loop.
    Transient submit failures (5xx/connection/timeout) retry with bounded
    exponential backoff; permanent (4xx/config/auth) failures stop immediately.
    The uploaded training_file_id is reused across retries."""
    client = _client()
    m = _load_manifest(run_dir)
    units = list(units_epochs)
    def done(u):
        r = m["units"].get(u, {})
        return bool(r.get("fine_tuned_model")) or r.get("status") in _TERMINAL
    transient_tries = {u: 0 for u in units}
    ratelimit_tries = 0
    while True:
        for u in units:
            r = m["units"].get(u, {})
            if r.get("job_id") and not r.get("fine_tuned_model") and r.get("status") not in _TERMINAL:
                _poll(client, run_dir, m, u)
        active = sum(1 for u in units if m["units"].get(u, {}).get("job_id") and not done(u))
        todo = [u for u in units if not m["units"].get(u, {}).get("job_id")]
        permanent_err = False
        transient_backoff = 0
        ratelimit_hit = False
        for u in todo:
            if active >= MAX_ACTIVE_JOBS:
                break
            res = _submit(client, run_dir, m, u, units_epochs[u])
            if res == "ratelimit":
                ratelimit_hit = True; break
            if res == "error":
                permanent_err = True; break
            if res == "transient":
                transient_tries[u] += 1
                if MAX_TRANSIENT_TRIES and transient_tries[u] >= MAX_TRANSIENT_TRIES:
                    print(f"  {u}: still failing after {MAX_TRANSIENT_TRIES} transient "
                          f"retries; stopping (rerun later to resume).")
                    permanent_err = True; break
                transient_backoff = min(60 * (2 ** (transient_tries[u] - 1)),
                                       TRANSIENT_BACKOFF_CAP)
                break
            if res is True:
                active += 1
        if all(done(u) for u in units) or not wait:
            break
        if permanent_err and active == 0:
            print("  cannot submit and nothing in flight; stopping (rerun to resume)."); break
        if ratelimit_hit:
            ratelimit_tries += 1
            if MAX_RATELIMIT_TRIES and ratelimit_tries >= MAX_RATELIMIT_TRIES:
                print(f"  rate-limited {MAX_RATELIMIT_TRIES}x; stopping "
                      f"(rerun later to resume).")
                break
            rl_backoff = min(60 * (2 ** (ratelimit_tries - 1)), RATELIMIT_BACKOFF_CAP)
            _w = (f"{rl_backoff/3600:.1f}h" if rl_backoff >= 3600 else f"{rl_backoff}s")
            print(f"  rate limit / daily cap; retrying in {_w} (attempt {ratelimit_tries}"
                  + ("" if not MAX_RATELIMIT_TRIES else f"/{MAX_RATELIMIT_TRIES}") + ")")
            time.sleep(rl_backoff); continue
        if transient_backoff > 0:
            # Back off even while other jobs are in flight. The old
            # `and active == 0` guard meant a 5xx retried every POLL_SECS
            # whenever anything else was running, bypassing the backoff
            # entirely -- the same defect already fixed on the rate-limit
            # branch above. TRANSIENT_BACKOFF_CAP is 10 min, so the longest
            # an in-flight job goes unpolled is 10 min.
            print(f"  transient error; backing off {transient_backoff}s before retry")
            time.sleep(transient_backoff); continue
        if POLL_SECS > 0:
            print(f"  {sum(1 for u in units if not done(u))} outstanding; sleep {POLL_SECS}s")
            time.sleep(POLL_SECS)
        else:
            break
    return m

def infer_repeats(client, model, test_df, temperature, variant="base", repeats=INFER_REPEATS,
                  label="", raw_dir=None):
    """Run `repeats` inference passes per test row. Returns
    (truth, mean, sd, nval, stats). Never touches train.

    sd is the SAMPLE STANDARD DEVIATION (ddof=1) of the valid repeats for each
    row: one sigma, so LLM and GPR share the SAME 1-SIGMA ERROR-BAR CONVENTION.
    They are NOT the same quantity: this is empirical scatter across stochastic
    decodes (sampling variability only), whereas GPR's return_std is a posterior
    predictive standard deviation. Same convention, different meaning -- say so
    in any caption that shows both.
    Rows with a single valid repeat get sd = 0 (no spread estimable).

    If raw_dir is given, writes rep1.csv .. repN.csv there: every individual
    call's raw model output (question, raw pred string, truth) is preserved so
    nothing is lost and results are fully reproducible/inspectable. stats tracks
    invalids at BOTH the call level (any single failed parse) and the row level
    (a row is invalid only if every repeat failed)."""
    recs = [row_to_record(r, variant) for _, r in test_df.iterrows()]
    questions = [r["messages"][1]["content"] for r in recs]
    truth_str = [r["messages"][2]["content"] for r in recs]
    preds = np.full((len(recs), repeats), np.nan)
    raw_txt = [["" for _ in range(repeats)] for _ in recs]
    truth = np.array([float(t) for t in truth_str])
    total = len(recs) * repeats
    done = 0; invalid_calls = 0
    tag = f"[{label}] " if label else ""
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)
    print(f"    {tag}inferring {len(recs)} rows x {repeats} repeats = {total} calls",
          flush=True)
    for j in range(repeats):
        for i, rec in enumerate(recs):
            resp = client.chat.completions.create(
                model=model, temperature=temperature,
                messages=[rec["messages"][0], rec["messages"][1]])
            raw = resp.choices[0].message.content
            raw_txt[i][j] = raw
            try:
                preds[i, j] = float(str(raw).strip())
            except (ValueError, TypeError):
                preds[i, j] = np.nan
                invalid_calls += 1
            done += 1
        if raw_dir:
            pd.DataFrame({"question": questions,
                          "pred": [raw_txt[i][j] for i in range(len(recs))],
                          "truth": truth_str}).to_csv(
                os.path.join(raw_dir, f"rep{j+1}.csv"),
                index=False, quoting=csv.QUOTE_ALL, escapechar="\\")
        n_bad = int(np.isnan(preds[:, j]).sum())
        print(f"    {tag}repeat {j+1}/{repeats} done "
              f"({done}/{total} calls" + (f", {n_bad} invalid this pass" if n_bad else "") + ")",
              flush=True)
    mean = np.nanmean(preds, axis=1)
    nval = np.sum(~np.isnan(preds), axis=1)
    # one sigma across the valid repeats (ddof=1). Same 1-sigma convention as
    # GPR's predictive sd (not the same quantity). nval<2 -> 0.
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(preds, axis=1, ddof=1)
    sd = np.where(nval > 1, sd, 0.0)
    stats = {"total_calls": total, "invalid_calls": invalid_calls,
             "invalid_call_rate": (invalid_calls / total) if total else None,
             "invalid_rows": int((nval == 0).sum()), "n_rows": len(recs)}
    return truth, mean, sd, nval, stats

# ===========================================================================
# ===  LLM MODES  ===========================================================
# ===========================================================================
def _unit(strategy, k):
    return f"{strategy}__split_{k:02d}"

def _best_temp(args):
    """Resolve inference temperature: --temperature override > tuned llm_temp.json
    > DEFAULT_TEMPERATURE (the value you already selected)."""
    if getattr(args, "temperature", None) is not None:
        return args.temperature
    p = os.path.join(args.out_dir, "llm_temp.json")
    if os.path.exists(p):
        return json.load(open(p))["best_temperature"]
    return DEFAULT_TEMPERATURE

def _cv_rg_models(args):
    """The already-trained restricted-group models from llm_cv, keyed by fold.
    Temperature is inference-only, so tuning never needs to re-train these."""
    m = _load_manifest(os.path.join(args.out_dir, "llm_cv"))
    out = {}
    for k in range(5):
        mdl = m["units"].get(_unit("restricted-group", k), {}).get("fine_tuned_model")
        if mdl:
            out[k] = mdl
    return out

def _cv_rg_epochs(args):
    """The n_epochs llm_cv's RG models were ACTUALLY trained at, read from its
    manifest (_submit records it at submission time). Returns the single value if
    all five agree, else None (disagreement, or a legacy manifest predating the
    field).

    This exists so epoch-tune labels its reused arm from a fact rather than an
    assumption. Deriving it from --epochs instead meant `--epochs 10` would score
    the 5-epoch CV models as the 10-epoch arm AND submit five new 5-epoch
    fine-tunes -- a silently wrong result and ~$37."""
    m = _load_manifest(os.path.join(args.out_dir, "llm_cv"))
    eps = set()
    for k in range(5):
        rec = m["units"].get(_unit("restricted-group", k), {})
        if rec.get("fine_tuned_model"):
            e = rec.get("epochs")
            if e is None:
                return None
            eps.add(int(e))
    return next(iter(eps)) if len(eps) == 1 else None

def mode_llm_temp_tune(args):
    """Temperature sweep TEMP_GRID x the RG folds, scored BY ROW (all RG rows
    concatenated, one OME/log-R2 per temperature).

    Costs ZERO fine-tunes: temperature only affects inference, and the RG models
    trained by llm-cv are on the same folds at the same epochs, so this reuses
    them directly. Inference is cached per (row, model, temperature) in the
    llm_cv row store, so the temperature you already ran RG at costs nothing to
    re-score; only the other temperatures make calls."""
    rg = load_rg_dir(args.rg_dir)
    run_dir = os.path.join(args.out_dir, "llm_cv")   # reuse llm_cv's models + row store
    models = _cv_rg_models(args)
    missing = [k for k in range(len(rg)) if k not in models]
    if missing:
        sys.exit(f"temp-tune reuses the llm-cv restricted-group models, but folds "
                 f"{missing} have none in {run_dir}/manifest.json. Run llm-cv first.")
    print(f"temp-tune: reusing {len(models)} trained RG models from llm_cv "
          f"(0 fine-tunes). Sweeping {TEMP_GRID}.")
    n_rows = sum(len(te) for _, te in rg)
    print(f"  inference: {len(TEMP_GRID)} temps x {n_rows} rows x {INFER_REPEATS} repeats "
          f"= {len(TEMP_GRID)*n_rows*INFER_REPEATS} calls max (cached rows are free)")
    if args.prep_only:
        print("prep_only: nothing to prep -- temp-tune trains nothing."); return
    client = None
    rows = []
    tune_dir = os.path.join(args.out_dir, "llm_temp_tune")
    n_expected = sum(len(te) for _, te in rg)
    for temp in TEMP_GRID:
        yt_all, yp_all, n_bad = [], [], 0
        for k, (tr, te) in enumerate(rg):
            u = _unit("restricted-group", k)
            # Reuse llm_cv's row store (so the CV temperature is already cached
            # and free) but write this sweep's raw/predictions into a SEPARATE
            # per-temperature directory. The store is keyed by temperature; the
            # raw files are not, so writing them into llm_cv's unit dir would
            # overwrite the original CV provenance with the last temperature.
            t, mean, half, orient, flu = _cached_infer(
                client, run_dir, u, models[k], te, temp,
                artifact_dir=os.path.join(tune_dir, f"t{temp}", u))
            ok = np.isfinite(mean)
            n_bad += int((~ok).sum())
            yt_all.append(t[ok]); yp_all.append(mean[ok])
        yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
        if len(yt) != n_expected:
            # A row whose every repeat failed to parse (the prompt permits "null")
            # drops out. Scoring this arm on fewer rows than another arm makes the
            # comparison invalid -- a single hard row could decide the winner.
            print(f"  temp={temp}: INVALID -- {len(yt)}/{n_expected} rows have a finite "
                  f"prediction ({n_bad} row(s) unparseable). Arm NOT scored.")
            continue
        # BY ROW over every RG row, through the SAME metric_ci() the parity panels
        # and every other reported metric use -- so the tuning CSV carries the
        # bootstrap CIs Sec 1.2 requires of every reported metric, not a bare
        # log-R2 next to OME_std_rows (which is row-error dispersion, a
        # diagnostic, NOT an uncertainty of the estimate).
        r = metric_row(yt, yp, temperature=temp)
        rows.append(r)
        print(f"  temp={temp}: OME={r['OME']:.3f} [{r['OME_ci_lo']:.3f}, {r['OME_ci_hi']:.3f}] "
              f"logR2={r['logR2']:+.3f} [{r['logR2_ci_lo']:+.3f}, {r['logR2_ci_hi']:+.3f}] "
              f"n={r['n_rows']}")
    if len(rows) != len(TEMP_GRID):
        print(f"\ntemp-tune: only {len(rows)}/{len(TEMP_GRID)} temperatures scored on all "
              f"{n_expected} rows -- NOT writing llm_temp.json. Arms evaluated on "
              f"different row sets cannot be compared.")
        return
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "temp_tune_results.csv"), index=False)
    best = res.sort_values(["OME", "logR2"], ascending=[True, False]).iloc[0]
    json.dump({"best_temperature": float(best.temperature)},
              open(os.path.join(args.out_dir, "llm_temp.json"), "w"), indent=2)
    print(f"BEST temperature = {best.temperature}; wrote llm_temp.json")
    if float(best.temperature) != DEFAULT_TEMPERATURE:
        # random and variable are BOTH 10-fold over all rows, so each tests every
        # row exactly once: 201 + 201. (The old 181 here was a leftover from the
        # pre-KFold ShuffleSplit design.)
        n_rv = 201 + 201   # random + variable rows currently cached at DEFAULT
        print(f"\n  *** WARNING: the tuned temperature ({best.temperature}) differs from "
              f"DEFAULT_TEMPERATURE ({DEFAULT_TEMPERATURE}).\n"
              f"      _row_key includes the temperature and _best_temp now reads "
              f"llm_temp.json, so llm-cv / llm-report will resolve to {best.temperature}\n"
              f"      and MISS the cached random/variable predictions (which were run at "
              f"{DEFAULT_TEMPERATURE}): ~{n_rv} rows x {INFER_REPEATS} = {n_rv*INFER_REPEATS} "
              f"calls, and the numbers\n"
              f"      would not match the ones already reported. The restricted-group rows "
              f"ARE cached at {best.temperature} -- this sweep just wrote them.\n"
              f"      epoch-tune and llm-prod are MEANT to use the tuned temperature; that "
              f"is not a hazard. llm-ablation never uses the row cache and\n"
              f"      re-infers on every run regardless.\n"
              f"      To keep the existing CV results: delete runs/llm_temp.json, or pass "
              f"--temperature {DEFAULT_TEMPERATURE} to llm-cv / llm-report. ***")

def mode_llm_epoch_tune(args):
    """Epoch sweep EPOCH_GRID x the RG folds, scored BY ROW.

    The llm-cv restricted-group models were trained on these exact folds at
    DEFAULT_EPOCHS, so that arm of the grid is REUSED rather than retrained --
    only the other epoch settings cost fine-tunes. With EPOCH_GRID = [5, 10] and
    llm-cv trained at 5, that is 5 new models instead of 10."""
    rg = load_rg_dir(args.rg_dir)
    run_dir = os.path.join(args.out_dir, "llm_epoch_tune")
    cv_dir = os.path.join(args.out_dir, "llm_cv")
    reuse = _cv_rg_models(args)
    # The reused arm is labelled from llm_cv's manifest -- what those models were
    # really trained at -- never from --epochs.
    reuse_ep = _cv_rg_epochs(args)
    if reuse_ep is None and reuse:
        reuse_ep = DEFAULT_EPOCHS
        print(f"epoch-tune: llm_cv's manifest records no n_epochs for its RG units "
              f"(written before that field existed); assuming DEFAULT_EPOCHS="
              f"{DEFAULT_EPOCHS} for the reused arm. Verify with:\n"
              f"  python3 -c \"import json; m=json.load(open('{cv_dir}/manifest.json'))"
              f"['units']; print({{k: v.get('epochs') for k, v in m.items() "
              f"if k.startswith('restricted-group')}})\"")
    if args.epochs is not None and reuse and args.epochs != reuse_ep:
        sys.exit(f"epoch-tune: --epochs {args.epochs} contradicts the {reuse_ep} epochs "
                 f"llm_cv's restricted-group models were actually trained at. The reused "
                 f"arm is labelled from the manifest, so honouring --epochs here would "
                 f"score those {reuse_ep}-epoch models as the {args.epochs}-epoch arm and "
                 f"submit a fresh set for {reuse_ep}. Drop --epochs; the grid is "
                 f"EPOCH_GRID={EPOCH_GRID}.")
    can_reuse = len(reuse) == len(rg) and reuse_ep in EPOCH_GRID
    units_epochs = {}
    for ep in EPOCH_GRID:
        if can_reuse and ep == reuse_ep:
            continue                     # llm-cv already trained this arm
        for k, (tr, te) in enumerate(rg):
            u = f"ep{ep}__split_{k:02d}"
            write_jsonl(tr, os.path.join(run_dir, u, "train.jsonl"))
            write_jsonl(te, os.path.join(run_dir, u, "test.jsonl"))
            units_epochs[u] = ep
    if can_reuse:
        print(f"epoch-tune: reusing the {len(reuse)} llm-cv RG models as the "
              f"{reuse_ep}-epoch arm; {len(units_epochs)} new fine-tunes needed "
              f"(instead of {len(EPOCH_GRID)*len(rg)}).")
    else:
        print(f"epoch-tune: no complete llm-cv RG model set to reuse; "
              f"{len(units_epochs)} fine-tunes needed.")
    if args.prep_only:
        print(f"prepped {len(units_epochs)} units under {run_dir}"); return
    print("NOTE: epoch-tune submits real fine-tune jobs (costs money).")
    m = orchestrate(run_dir, units_epochs, wait=True) if units_epochs else {"units": {}}
    client = None
    temp = _best_temp(args)
    rows = []
    incomplete = []
    n_expected = sum(len(te) for _, te in rg)
    for ep in EPOCH_GRID:
        yt_all, yp_all, missing = [], [], []
        for k, (tr, te) in enumerate(rg):
            if can_reuse and ep == reuse_ep:
                # score the llm-cv model from its own row store (free), but write
                # artifacts into this run's dir -- never back into llm_cv's unit
                # dirs, whose raw files are the original CV provenance.
                t, mean, half, orient, flu = _cached_infer(
                    client, cv_dir, _unit("restricted-group", k), reuse[k], te, temp,
                    artifact_dir=os.path.join(run_dir, f"ep{ep}__split_{k:02d}"))
            else:
                u = f"ep{ep}__split_{k:02d}"
                rec = m["units"].get(u, {})
                model = rec.get("fine_tuned_model")
                if not model:
                    missing.append((u, rec.get("status", "not submitted"),
                                    rec.get("job_id")))
                    continue
                t, mean, half, orient, flu = _cached_infer(client, run_dir, u, model, te, temp)
            ok = np.isfinite(mean)
            yt_all.append(t[ok]); yp_all.append(mean[ok])
        if missing:
            # An arm scored on fewer folds covers fewer rows, so comparing it
            # against a complete arm is meaningless -- and would still write a
            # winner. Refuse to score it at all.
            incomplete.append((ep, missing))
            print(f"  epochs={ep}: INCOMPLETE -- {len(missing)}/{len(rg)} fold(s) have no "
                  f"model, so this arm is NOT scored (comparing arms of different "
                  f"sizes would be invalid):")
            for u, st, jid in missing:
                print(f"      {u}: status={st}" + (f" job={jid}" if jid else ""))
            continue
        yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
        if len(yt) != n_expected:
            # Same hazard one level down: a row whose every repeat failed to parse
            # (the prompt permits "null") silently drops out, so this arm would be
            # scored on fewer rows than the other one.
            incomplete.append((ep, []))
            print(f"  epochs={ep}: INVALID -- {len(yt)}/{n_expected} rows have a finite "
                  f"prediction. Arm NOT scored (arms must cover the same rows).")
            continue
        # BY ROW over every RG row, via the same metric_ci() as everything else
        r = metric_row(yt, yp, epochs=ep)
        rows.append(r)
        print(f"  epochs={ep}: OME={r['OME']:.3f} [{r['OME_ci_lo']:.3f}, {r['OME_ci_hi']:.3f}] "
              f"logR2={r['logR2']:+.3f} [{r['logR2_ci_lo']:+.3f}, {r['logR2_ci_hi']:+.3f}] "
              f"n={r['n_rows']}")
    if incomplete:
        # A job that ended 'failed'/'cancelled' keeps its job_id in the manifest,
        # so a plain rerun will NOT resubmit it -- say so rather than let a rerun
        # look like it did nothing.
        dead = [(u, st, jid) for _, ms in incomplete for u, st, jid in ms
                if st in ("failed", "cancelled")]
        if dead:
            print(f"\n  {len(dead)} job(s) ended terminally. A rerun will NOT resubmit "
                  f"them -- their job_id is still in the manifest. To retry, clear the "
                  f"unit(s) first:")
            names = ", ".join(repr(u) for u, _, _ in dead)
            print(f"    python3 -c \"import json; p='{run_dir}/manifest.json'; "
                  f"m=json.load(open(p)); [m['units'].pop(u,None) for u in [{names}]]; "
                  f"json.dump(m,open(p,'w'),indent=2)\"")
    if len(rows) < len(EPOCH_GRID):
        print(f"\nepoch-tune: only {len(rows)}/{len(EPOCH_GRID)} arms complete -- "
              f"NOT writing llm_best.json (a partial grid cannot select a winner). "
              f"Resolve the above and rerun.")
        return
    if not rows:
        print("no epoch arms complete; nothing written."); return
    res = pd.DataFrame(rows)
    best = res.sort_values(["OME", "logR2"], ascending=[True, False]).iloc[0]
    res.to_csv(os.path.join(args.out_dir, "epoch_tune_results.csv"), index=False)
    json.dump({"best_epochs": int(best.epochs), "temperature": temp},
              open(os.path.join(args.out_dir, "llm_best.json"), "w"), indent=2)
    print(f"BEST epochs = {int(best.epochs)} (temp {temp}); wrote llm_best.json")

def _best_epochs(args):
    p = os.path.join(args.out_dir, "llm_best.json")
    if os.path.exists(p):
        return json.load(open(p))["best_epochs"]
    if args.epochs:
        return args.epochs
    return DEFAULT_EPOCHS   # epoch-tune not run: fall back to the default

def mode_llm_cv(args):
    """Winning epochs on RG(5)+random(10)+variable(10) = 25 models; 10 repeats/test."""
    pool = load_master_csv(args.data_csv)
    epochs = _best_epochs(args)
    temp = _best_temp(args)
    run_dir = os.path.join(args.out_dir, "llm_cv")
    splits = {s: get_splits(s, pool, args.rg_dir, args.out_dir) for s in STRATEGIES}
    units_epochs = {}
    for s, sp in splits.items():
        for k, (tr, te) in enumerate(sp):
            u = _unit(s, k)
            write_jsonl(tr, os.path.join(run_dir, u, "train.jsonl"))
            write_jsonl(te, os.path.join(run_dir, u, "test.jsonl"))
            units_epochs[u] = epochs
    if args.prep_only:
        print(f"prepped {len(units_epochs)} CV units (epochs={epochs}) under {run_dir}"); return
    m = orchestrate(run_dir, units_epochs, wait=True)
    client = None
    rows, pooled = [], {s: ([], [], [], [], [], []) for s in STRATEGIES}
    for s, sp in splits.items():
        for k, (tr, te) in enumerate(sp):
            u = _unit(s, k)
            model = m["units"].get(u, {}).get("fine_tuned_model")
            cache = os.path.join(run_dir, u, "predictions.csv")
            if not os.path.exists(cache) and not model:
                print(f"  {u}: no model; skip"); continue
            t, mean, half, orient, flu = _cached_infer(client, run_dir, u, model, te, temp)
            ok = ~np.isnan(mean)
            ome, r2 = metrics(t[ok], mean[ok])
            rows.append(dict(strategy=s, split=k, OME=ome, logR2=r2,
                             n=int(ok.sum()), mean_sigma=float(np.nanmean(half))))
            pooled[s][0].append(t[ok]); pooled[s][1].append(mean[ok])
            pooled[s][2].append(half[ok]); pooled[s][3].append(te["smiles"].to_numpy()[ok])
            pooled[s][4].append(orient[ok]); pooled[s][5].append(flu[ok])
            print(f"  {u}: OME={ome:.3f} logR2={r2:+.3f} n={int(ok.sum())}")
    _summarize_and_plot(rows, pooled, run_dir, "llm")

def _split_hash(te):
    """Content hash of the test rows (identity + target + key features), so the
    cache is tied to the exact data it was computed on -- reproducibility."""
    cols = [c for c in ["smiles", "e_y (A3/atom)", "coating name", "orientation",
                        "mission_time (yr)", "solar (esh)", "fluence"] if c in te.columns]
    blob = te[cols].to_csv(index=False).encode()
    return hashlib.sha1(blob).hexdigest()[:12]

def _row_key(row, model, temp):
    """Stable identity for one polymer row + inference config, so a prediction
    can be reused across ANY split that contains this row."""
    ident = "|".join(str(row.get(c, "")) for c in
                     ["smiles", "e_y (A3/atom)", "coating name", "orientation",
                      "mission_time (yr)", "solar (esh)", "fluence"])
    h = hashlib.sha1(ident.encode()).hexdigest()[:16]
    return f"{model}__t{temp}__r{INFER_REPEATS}__{h}"

def _rowstore_path(run_dir):
    return os.path.join(run_dir, "row_predictions.csv")

def _load_rowstore(run_dir):
    p = _rowstore_path(run_dir)
    if os.path.exists(p):
        d = pd.read_csv(p)
        return {r["key"]: r for _, r in d.iterrows()}
    return {}

def _append_rowstore(run_dir, records):
    if not records:
        return
    p = _rowstore_path(run_dir)
    df_new = pd.DataFrame(records)
    if os.path.exists(p):
        df_new = pd.concat([pd.read_csv(p), df_new], ignore_index=True)
        df_new = df_new.drop_duplicates(subset="key", keep="last")
    df_new.to_csv(p, index=False)

def _cached_infer(client, run_dir, u, model, te, temp, variant="base", artifact_dir=None):
    """Return (truth, mean, half, orient, flu) for one split. Predictions are
    cached at the ROW level in run_dir/row_predictions.csv keyed by row identity
    + model + temperature + repeats, so a row inferred under ANY split (even a
    different fold scheme) is reused and never re-inferred. Only rows never seen
    before are sent to the API. Also keeps the per-split predictions.csv + raw
    per-repeat files for provenance.

    artifact_dir overrides where those per-split files are written. It defaults to
    run_dir/u. Callers that reuse another run's row store at a DIFFERENT setting
    (e.g. temp-tune sweeping temperature against the llm_cv models) must pass a
    separate directory: the row store is keyed by temperature, but raw/rep*.csv
    and predictions.csv are NOT, so writing them into the original unit dir would
    overwrite that unit's original provenance with the sweep's last setting."""
    udir = artifact_dir if artifact_dir else os.path.join(run_dir, u)
    os.makedirs(udir, exist_ok=True)
    store = _load_rowstore(run_dir)
    keys = [_row_key(r, model, temp) for _, r in te.iterrows()]
    have_mask = [k in store for k in keys]
    n_have, n_need = sum(have_mask), len(keys) - sum(have_mask)

    truth = target(te)
    orient = te["orientation"].astype(str).str.strip().str.lower().to_numpy()
    flu = pd.to_numeric(te["fluence"], errors="coerce").to_numpy()
    mean = np.full(len(te), np.nan); half = np.full(len(te), np.nan)

    # reuse rows already in the store
    for i, (k, hv) in enumerate(zip(keys, have_mask)):
        if hv:
            mean[i] = float(store[k]["pred"]); half[i] = float(store[k]["band"])

    # infer only the rows never seen before
    if n_need > 0:
        if model is None:
            raise RuntimeError(f"{u}: {n_need} rows have no cached prediction and no "
                               f"fine_tuned_model is available to infer them.")
        if client is None:
            client = _client()
        need_idx = [i for i, hv in enumerate(have_mask) if not hv]
        te_need = te.iloc[need_idx]
        t2, m2, h2, nval2, stats = infer_repeats(
            client, model, te_need, temp, variant, label=u,
            raw_dir=os.path.join(udir, "raw"))
        new_recs = []
        for j, i in enumerate(need_idx):
            mean[i] = m2[j]; half[i] = h2[j]
            new_recs.append({"key": keys[i], "model": model, "temperature": temp,
                             "repeats": INFER_REPEATS, "smiles": te.iloc[i]["smiles"],
                             "truth": truth[i], "pred": m2[j], "band": h2[j],
                             "orientation": orient[i], "fluence": flu[i]})
        _append_rowstore(run_dir, new_recs)
        print(f"  {u}: {n_have} reused, {n_need} newly inferred")
    else:
        print(f"  {u}: all {n_have} rows reused from row store (no inference)")

    # write the per-split predictions.csv for provenance / plotting
    pd.DataFrame(dict(smiles=te["smiles"].to_numpy(), truth=truth, pred=mean, band=half,
                      orientation=orient, fluence=flu)).to_csv(
        os.path.join(udir, "predictions.csv"), index=False)
    json.dump({"model": model, "temperature": temp, "repeats": INFER_REPEATS,
               "variant": variant, "n_rows": int(len(te)),
               "split_hash": _split_hash(te), "n_reused": int(n_have),
               "n_newly_inferred": int(n_need)},
              open(os.path.join(udir, "predictions_meta.json"), "w"), indent=2)
    return truth, mean, half, orient, flu

def mode_llm_report(args):
    """Read-only per-method reporting. Each split method (restricted-group,
    random, variable) is an INDEPENDENT experiment: as soon as all of a method's
    splits are finished it gets a FINAL plot/CSV (llm_<method>_...); a method
    still in progress gets a partial one (llm_partial_<method>_...). Uses cached
    predictions where present (no re-inference); only unfinished-but-trained
    splits are inferred once, then cached. Safe to run repeatedly alongside
    the running llm-cv job."""
    pool = load_master_csv(args.data_csv)
    temp = _best_temp(args)
    run_dir = os.path.join(args.out_dir, "llm_cv")
    m = _load_manifest(run_dir)
    want = [args.split] if getattr(args, "split", None) else STRATEGIES
    splits = {s: get_splits(s, pool, args.rg_dir, args.out_dir) for s in want}
    client = None
    for s in want:
        sp = splits[s]
        rows, pooled = [], {s: ([], [], [], [], [], [])}
        n_done, n_total = 0, len(sp)
        for k, (tr, te) in enumerate(sp):
            u = _unit(s, k)
            cache = os.path.join(run_dir, u, "predictions.csv")
            model = m["units"].get(u, {}).get("fine_tuned_model")
            if not os.path.exists(cache) and not model:
                continue
            if client is None and not os.path.exists(cache):
                client = _client()
            t, mean, half, orient, flu = _cached_infer(client, run_dir, u, model, te, temp)
            n_done += 1
            ok = ~np.isnan(mean)
            ome, r2 = metrics(t[ok], mean[ok])
            rows.append(dict(strategy=s, split=k, OME=ome, logR2=r2,
                             n=int(ok.sum()), mean_sigma=float(np.nanmean(half))))
            pooled[s][0].append(t[ok]); pooled[s][1].append(mean[ok])
            pooled[s][2].append(half[ok]); pooled[s][3].append(te["smiles"].to_numpy()[ok])
            pooled[s][4].append(orient[ok]); pooled[s][5].append(flu[ok])
            print(f"  {u}: OME={ome:.3f} logR2={r2:+.3f} n={int(ok.sum())}")
        if n_done == 0:
            print(f"{s}: no finished splits yet."); continue
        complete = (n_done == n_total)
        tag = f"llm_{s}" if complete else f"llm_partial_{s}"
        _summarize_and_plot(rows, {s: pooled[s]}, run_dir, tag, only=[s])
        state = "COMPLETE" if complete else f"partial ({n_done}/{n_total})"
        print(f"{s}: {state} -> wrote {tag}_* to {run_dir}/\n")

def mode_llm_prod(args):
    """1 model on all 201; 10 repeats per point; one parity."""
    pool = load_master_csv(args.data_csv)
    epochs = _best_epochs(args)
    temp = _best_temp(args)
    run_dir = os.path.join(args.out_dir, "llm_prod")
    write_jsonl(pool, os.path.join(run_dir, "prod", "train.jsonl"))
    if args.prep_only:
        print(f"prepped production train (epochs={epochs}) under {run_dir}"); return
    m = orchestrate(run_dir, {"prod": epochs}, wait=True)
    model = m["units"].get("prod", {}).get("fine_tuned_model")
    if not model:
        sys.exit("production model not ready")
    client = _client()
    t, mean, half, nval, prod_stats = infer_repeats(client, model, pool, temp, label="production",
                                                     raw_dir=os.path.join(run_dir, "raw"))
    ok = ~np.isnan(mean)
    # IN-SAMPLE. The production model trains on all 201 rows and is then asked
    # about those same 201 rows, so this is a fit, not a generalization estimate.
    # The CI is a CI on the fit. Held-out LLM performance is the validation
    # figure, on the original CSV. The box says so; do not quote these as
    # performance.
    ome, ome_ci, r2, r2_ci, sd = metric_ci(t[ok], mean[ok])
    # row_index = the canonical master-CSV position (load_master_csv reset the
    # index), so a prediction is auditable without leaning on PSMILES or target,
    # neither of which is unique across rows.
    pd.DataFrame(dict(row_index=pool.index.to_numpy(), smiles=pool["smiles"].to_numpy(),
                      truth=t, pred=mean, sigma=half, n=nval)).to_csv(
        os.path.join(run_dir, "production_predictions.csv"), index=False)
    pd.DataFrame([metric_row(t[ok], mean[ok], scope="in_sample", model=model,
                             temperature=temp, epochs=epochs,
                             repeats=INFER_REPEATS,
                             data_hash=_dataset_hash(args.data_csv))]).to_csv(
        os.path.join(run_dir, "production_metrics.csv"), index=False)
    _parity(t[ok], mean[ok], half[ok],
            PLOT_STYLE["prod_title_template"].format(model=PLOT_STYLE["model_display"]["llm"]),
            os.path.join(run_dir, "production_parity.svg"),
            metric_box(ome, ome_ci, r2, r2_ci, int(ok.sum()), prefix="In-sample "),
            orient=pool["orientation"].astype(str).str.strip().str.lower().to_numpy()[ok],
            flu=pd.to_numeric(pool["fluence"], errors="coerce").to_numpy()[ok],
            style=scaled_style(5.5))
    print(f"production (IN-SAMPLE): OME={ome:.3f} [{ome_ci[0]:.3f}, {ome_ci[1]:.3f}] "
          f"logR2={r2:+.3f} [{r2_ci[0]:+.3f}, {r2_ci[1]:+.3f}] n={int(ok.sum())}")

def mode_llm_ablation(args):
    """layers/thickness ablation (baseline/+layers/+thickness), matched-control
    rows (only rows with BOTH known). Built for completeness; you already ran it."""
    pool = load_master_csv(args.data_csv)
    epochs = _best_epochs(args)
    temp = _best_temp(args)
    both = (pd.to_numeric(pool["layers"], errors="coerce").notna()
            & pd.to_numeric(pool["thickness (mm)"], errors="coerce").notna())
    ctrl = pool[both].reset_index(drop=True)
    print(f"ablation matched-control rows (known layers+thickness): {len(ctrl)}/{len(pool)}")
    # RG test-chemistry membership from the provided split; rows taken from the
    # master pool (only source with layers/thickness) and restricted to matched-
    # control rows so baseline/+layers/+thickness compare on identical rows.
    rg_disk = load_rg_dir(args.rg_dir)
    rg_ctrl = []
    for _, te_disk in rg_disk:
        test_chems = set(te_disk["smiles"])
        te = ctrl[ctrl["smiles"].isin(test_chems)].reset_index(drop=True)
        tr = ctrl[~ctrl["smiles"].isin(test_chems)].reset_index(drop=True)
        rg_ctrl.append((tr, te))
    run_dir = os.path.join(args.out_dir, "llm_ablation")
    units_epochs, plan = {}, {"baseline": "base", "layers": "layers", "thickness": "thickness"}
    for cond, variant in plan.items():
        for k, (tr, te) in enumerate(rg_ctrl):
            u = f"{cond}__split_{k:02d}"
            write_jsonl(tr, os.path.join(run_dir, u, "train.jsonl"), variant)
            write_jsonl(te, os.path.join(run_dir, u, "test.jsonl"), variant)
            units_epochs[u] = epochs
    if args.prep_only:
        print(f"prepped {len(units_epochs)} ablation units under {run_dir}"); return
    print("NOTE: llm-ablation submits real fine-tune jobs (costs money).")
    m = orchestrate(run_dir, units_epochs, wait=True)
    client = _client()
    rows = []
    for cond, variant in plan.items():
        yt_all, yp_all = [], []
        n_missing = 0
        for k, (tr, te) in enumerate(rg_ctrl):
            u = f"{cond}__split_{k:02d}"
            model = m["units"].get(u, {}).get("fine_tuned_model")
            if not model:
                n_missing += 1
                continue
            t, mean, half, nval, _ = infer_repeats(client, model, te, temp, variant, label=f"{cond}/split_{k:02d}",
                                                    raw_dir=os.path.join(run_dir, u, "raw"))
            ok = ~np.isnan(mean); yt_all.append(t[ok]); yp_all.append(mean[ok])
        if not yt_all:
            print(f"  {cond}: no trained models yet ({n_missing}/{len(rg_ctrl)} folds "
                  f"missing); skipping -- rerun once they finish.")
            continue
        if n_missing:
            print(f"  {cond}: WARNING partial -- {n_missing}/{len(rg_ctrl)} folds "
                  f"have no model; metrics below are NOT the full ablation.")
        yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
        # same metric_ci the figure panels use, so table and figure agree by
        # construction rather than by coincidence
        ome, ome_ci, r2, r2_ci, sd = metric_ci(yt, yp)
        rows.append(dict(condition=cond,
                         OME=ome, OME_ci_lo=ome_ci[0], OME_ci_hi=ome_ci[1],
                         logR2=r2, logR2_ci_lo=r2_ci[0], logR2_ci_hi=r2_ci[1],
                         n_folds=len(yt_all), n=int(len(yt)), OME_std_rows=sd,
                         boot_B=METRIC_CI["B"], boot_seed=METRIC_CI["seed"],
                         ci_level=METRIC_CI["level"]))
        print(f"  {cond}: OME={ome:.3f} [{ome_ci[0]:.3f}, {ome_ci[1]:.3f}] "
              f"logR2={r2:+.3f} [{r2_ci[0]:+.3f}, {r2_ci[1]:+.3f}] "
              f"({len(yt_all)}/{len(rg_ctrl)} folds, n={len(yt)})")
    if not rows:
        print("no conditions complete yet; nothing written."); return
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "ablation_results.csv"), index=False)

# ===========================================================================
# ===  GPR MODES  ===========================================================
# ===========================================================================
def _fit_predict_gpr(tr, te, coat_lvls, bits, radius, kernel):
    ytr, yte = target(tr), target(te)
    ftr, ntr, ctr = build_blocks(tr, coat_lvls, bits, radius)
    fte, nte, cte = build_blocks(te, coat_lvls, bits, radius)
    ok_tr = ~np.isnan(ftr).any(1); ok_te = ~np.isnan(fte).any(1)
    ftr, ntr, ctr, ytr = ftr[ok_tr], ntr[ok_tr], ctr[ok_tr], ytr[ok_tr]
    fte, nte, cte, yte = fte[ok_te], nte[ok_te], cte[ok_te], yte[ok_te]
    n_drop = int((~ok_tr).sum() + (~ok_te).sum())
    sc = StandardScaler().fit(ntr)
    Xtr = np.hstack([ftr, sc.transform(ntr), ctr]); n_fp = ftr.shape[1]
    Xte = np.hstack([fte, sc.transform(nte), cte])
    g = make_gpr(n_fp, kernel=kernel).fit(Xtr, ytr)
    mu, sd = g.predict(Xte, return_std=True)
    chem = te["smiles"].to_numpy()[ok_te]
    orient = te["orientation"].astype(str).str.strip().str.lower().to_numpy()[ok_te]
    flu = pd.to_numeric(te["fluence"], errors="coerce").to_numpy()[ok_te]
    return yte, mu, sd, chem, n_drop, orient, flu

def mode_gpr_opt(args):
    """Sweep bits x radius x kernel; score by pooled OME/log-R2 on RG 5-fold,
    with the variable split reported alongside as a secondary check."""
    pool = load_master_csv(args.data_csv)
    coat = coating_levels(pool)
    rg = load_rg_dir(args.rg_dir)
    var = get_splits("variable", pool, args.rg_dir, args.out_dir)
    rows = []
    total_cfg = len(GPR_BITS_GRID) * len(GPR_RADIUS_GRID) * len(GPR_KERNEL_GRID)
    cfg_i = 0
    print(f"GPR sweep: {total_cfg} configs x (RG 5-fold + variable {len(var)}-fold)", flush=True)
    for bits in GPR_BITS_GRID:
        for radius in GPR_RADIUS_GRID:
            for kernel in GPR_KERNEL_GRID:
                cfg_i += 1
                print(f"  [{cfg_i}/{total_cfg}] fitting bits={bits} r={radius} "
                      f"{kernel} ...", flush=True)
                def score(splits):
                    yt, yp = [], []
                    for tr, te in splits:
                        yte, mu, sd, chem, _, _, _ = _fit_predict_gpr(tr, te, coat, bits, radius, kernel)
                        yt.append(yte); yp.append(mu)
                    yt = np.concatenate(yt); yp = np.concatenate(yp)
                    ome, r2 = metrics(yt, yp)
                    return ome, r2
                rg_ome, rg_r2 = score(rg)
                var_ome, var_r2 = score(var)
                rows.append(dict(bits=bits, radius=radius, kernel=kernel,
                                 rg_OME=rg_ome, rg_logR2=rg_r2,
                                 var_OME=var_ome, var_logR2=var_r2))
                print(f"  bits={bits} r={radius} {kernel:12s} | "
                      f"RG OME={rg_ome:.3f} logR2={rg_r2:+.3f} | "
                      f"VAR OME={var_ome:.3f} logR2={var_r2:+.3f}")
    res = pd.DataFrame(rows).sort_values(["rg_OME", "rg_logR2"],
                                         ascending=[True, False]).reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)
    res.to_csv(os.path.join(args.out_dir, "gpr_opt_results.csv"), index=False)
    best = res.iloc[0]
    var_best = res.sort_values(["var_OME", "var_logR2"], ascending=[True, False]).iloc[0]
    if (best.bits, best.radius, best.kernel) != (var_best.bits, var_best.radius, var_best.kernel):
        print(f"  NOTE: RG winner ({best.bits},{best.radius},{best.kernel}) != "
              f"VAR winner ({var_best.bits},{var_best.radius},{var_best.kernel}). Using RG.")
    json.dump({"bits": int(best.bits), "radius": int(best.radius), "kernel": best.kernel},
              open(os.path.join(args.out_dir, "gpr_best.json"), "w"), indent=2)
    print(f"BEST GPR (by RG): bits={int(best.bits)} radius={int(best.radius)} "
          f"kernel={best.kernel}; wrote gpr_best.json")

def _gpr_best(args):
    p = os.path.join(args.out_dir, "gpr_best.json")
    if os.path.exists(p):
        b = json.load(open(p)); return b["bits"], b["radius"], b["kernel"]
    sys.exit("no gpr_best.json found. Run 'gpr-opt' first to produce the "
             "optimized GPR config before gpr-cv / gpr-prod.")

def mode_gpr_cv(args):
    """Winning GPR config on RG(5)+random(10)+variable(10), each + sigma. Each
    split method is written as its own deliverable (gpr_<method>_*), in the
    order rg, random, variable. Use --split to run just one method."""
    pool = load_master_csv(args.data_csv)
    report_collisions(pool)
    coat = coating_levels(pool)
    bits, radius, kernel = _gpr_best(args)
    print(f"GPR config: bits={bits} radius={radius} kernel={kernel}")
    run_dir = os.path.join(args.out_dir, "gpr_cv"); os.makedirs(run_dir, exist_ok=True)
    want = [args.split] if getattr(args, "split", None) else STRATEGIES
    force = getattr(args, "force", False)
    for s in want:
        if not force and _gpr_cache_valid(run_dir, s, args.data_csv, args.out_dir,
                                          bits, radius, kernel, rg_dir=args.rg_dir):
            print(f"{s}: saved results validated (config + data + split hash + "
                  f"fold counts match) -- skipping.")
            continue
        if not force and os.path.exists(os.path.join(run_dir, f"gpr_{s}_predictions.csv")):
            print(f"{s}: existing results FAILED validation (stale/mismatched) "
                  f"-- recomputing.")
        sp = get_splits(s, pool, args.rg_dir, args.out_dir)
        print(f"{s}: fitting {len(sp)} GPR folds ...", flush=True)
        rows, pooled = [], {s: ([], [], [], [], [], [])}
        for k, (tr, te) in enumerate(sp):
            print(f"  {s} fold {k+1}/{len(sp)} ...", flush=True)
            yte, mu, sd, chem, ndrop, orient, flu = _fit_predict_gpr(tr, te, coat, bits, radius, kernel)
            ome, r2 = metrics(yte, mu)
            rows.append(dict(strategy=s, split=k, OME=ome, logR2=r2,
                             n=len(yte), mean_sigma=float(sd.mean()), n_dropped_fp=ndrop))
            pooled[s][0].append(yte); pooled[s][1].append(mu); pooled[s][2].append(sd)
            pooled[s][3].append(chem); pooled[s][4].append(orient); pooled[s][5].append(flu)
            print(f"  {s} split {k}: OME={ome:.3f} logR2={r2:+.3f} n={len(yte)} "
                  f"<sigma>={sd.mean():.3f}" + (f" [{ndrop} FP-dropped]" if ndrop else ""))
        _summarize_and_plot(rows, {s: pooled[s]}, run_dir, f"gpr_{s}", only=[s],
                            gpr_config=(bits, radius, kernel),
                            data_hash=_dataset_hash(args.data_csv),
                            membership_hash=_membership_hash(args.out_dir, s),
                            rg_split_hash=(_rg_split_hash(args.rg_dir) if s == "restricted-group" else None))
        print(f"{s}: wrote gpr_{s}_* to {run_dir}/\n")

def mode_gpr_prod(args):
    """1 GPR on all 201 + sigma; parity; 10-fold CV generalization.

    Two DIFFERENT quantities come out of this, and they must not be conflated:

      in_sample : the all-data model predicting the rows it was fit on. A fit,
                  not performance. Its CI is a CI on the fit.
      cv        : KFold(GPR_CV_FOLDS) refits, pooled OUT-OF-FOLD by row. This is
                  the held-out estimate, and it is the ONLY held-out GPR number
                  computed on the corrected production dataset -- gpr-cv's random
                  arm is the same split and config but runs on the ORIGINAL CSV
                  (Sec 1.1), and the corrected SMILES change the fingerprints and
                  therefore every prediction. Not a duplicate; do not delete it.

    Both are scored BY ROW via metric_ci(), never by averaging per-fold metrics.
    KFold tests every row exactly once, so the fold boundaries are an
    implementation detail and averaging over them averages over nothing (Sec 1.2);
    the previous mean-of-per-fold-log-R2 here was the same defect as Sec 3.7,
    which produced a fold-averaged -0.44 where the by-row value was +0.55."""
    pool = load_master_csv(args.data_csv)
    coat = coating_levels(pool)
    bits, radius, kernel = _gpr_best(args)
    run_dir = os.path.join(args.out_dir, "gpr_prod"); os.makedirs(run_dir, exist_ok=True)
    y = target(pool)
    f, n, c = build_blocks(pool, coat, bits, radius)
    ok = ~np.isnan(f).any(1); f, n, c, y = f[ok], n[ok], c[ok], y[ok]
    ch = pool["smiles"].to_numpy()[ok]
    # canonical master-CSV position, carried through the fingerprint-drop mask so
    # every prediction stays traceable to its dataset row
    row_index = pool.index.to_numpy()[ok]
    n_drop = int((~ok).sum())
    if n_drop:
        print(f"  {n_drop} row(s) dropped: invalid fingerprint. n = {len(y)}")
    p_orient = pool["orientation"].astype(str).str.strip().str.lower().to_numpy()[ok]
    p_flu = pd.to_numeric(pool["fluence"], errors="coerce").to_numpy()[ok]
    sc = StandardScaler().fit(n)
    X = np.hstack([f, sc.transform(n), c]); n_fp = f.shape[1]
    print(f"GPR production: fitting on all {len(y)} rows ...", flush=True)
    g = make_gpr(n_fp, kernel=kernel).fit(X, y)
    mu, sd = g.predict(X, return_std=True)
    tr_ome, tr_ome_ci, tr_r2, tr_r2_ci, _ = metric_ci(y, mu)

    kf = KFold(n_splits=GPR_CV_FOLDS, shuffle=True, random_state=SEED)
    cv_pred = np.full(len(y), np.nan); cv_fold = np.full(len(y), -1, int)
    print(f"  {GPR_CV_FOLDS}-fold generalization CV ...", flush=True)
    for fold_i, (tri, tei) in enumerate(kf.split(X)):
        print(f"    CV fold {fold_i+1}/{GPR_CV_FOLDS} ...", flush=True)
        s2 = StandardScaler().fit(n[tri])
        Xtr = np.hstack([f[tri], s2.transform(n[tri]), c[tri]])
        Xte = np.hstack([f[tei], s2.transform(n[tei]), c[tei]])
        gg = make_gpr(n_fp, kernel=kernel).fit(Xtr, y[tri])
        cv_pred[tei] = gg.predict(Xte); cv_fold[tei] = fold_i
    # KFold tests every row exactly once, so full coverage is guaranteed unless
    # something upstream is broken. Warn rather than block, and score whatever
    # was covered -- with n reported, so a short arm cannot pass unnoticed.
    covered = np.isfinite(cv_pred)
    if not covered.all():
        print(f"  WARNING: {int((~covered).sum())} row(s) never landed in a test fold; "
              f"the CV metrics below cover {int(covered.sum())}/{len(y)} rows.")

    pd.DataFrame([metric_row(y, mu, scope="in_sample"),
                  metric_row(y[covered], cv_pred[covered], scope="cv",
                             n_folds=GPR_CV_FOLDS)]).assign(
        fp_bits=bits, fp_radius=radius, kernel=kernel,
        data_hash=_dataset_hash(args.data_csv)).to_csv(
        os.path.join(run_dir, "production_cv.csv"), index=False)
    # the pooled out-of-fold predictions themselves, so the cv row above is
    # auditable without refitting ten GPRs
    pd.DataFrame(dict(row_index=row_index[covered], smiles=ch[covered],
                      fold=cv_fold[covered], truth=y[covered],
                      pred=cv_pred[covered])).to_csv(
        os.path.join(run_dir, "production_cv_predictions.csv"), index=False)
    pd.DataFrame(dict(row_index=row_index, smiles=ch, truth=y, pred=mu,
                      sigma=sd)).to_csv(
        os.path.join(run_dir, "production_predictions.csv"), index=False)

    cv_ome, cv_ome_ci, cv_r2, cv_r2_ci, _ = metric_ci(y[covered], cv_pred[covered])
    _parity(y, mu, sd,
            PLOT_STYLE["prod_title_template"].format(model=PLOT_STYLE["model_display"]["gpr"]),
            os.path.join(run_dir, "production_parity.svg"),
            metric_box(tr_ome, tr_ome_ci, tr_r2, tr_r2_ci, len(y), prefix="In-sample "),
            orient=p_orient, flu=p_flu, style=scaled_style(5.5))
    print(f"GPR production IN-SAMPLE: OME={tr_ome:.3f} [{tr_ome_ci[0]:.3f}, {tr_ome_ci[1]:.3f}] "
          f"logR2={tr_r2:+.3f} [{tr_r2_ci[0]:+.3f}, {tr_r2_ci[1]:+.3f}] n={len(y)}")
    print(f"GPR production CV (by row, pooled out-of-fold): "
          f"OME={cv_ome:.3f} [{cv_ome_ci[0]:.3f}, {cv_ome_ci[1]:.3f}] "
          f"logR2={cv_r2:+.3f} [{cv_r2_ci[0]:+.3f}, {cv_r2_ci[1]:+.3f}] "
          f"n={int(covered.sum())}")

def mode_gpr_ablation(args):
    """GPR macroscopic-descriptor ablation on matched-control rows (known
    layers AND thickness): baseline vs +layers vs +thickness. Runs per split
    method (restricted-group, random, variable); use --split for just one,
    else all three. Folds come from get_splits; test/train rows are selected by
    exact master-row index (random/variable) or by test chemistry set (RG),
    avoiding the smiles+target identity ambiguity. Reports the BY-ROW OME and
    log-R2 over all matched-control rows, each with a row-bootstrap 95% CI (the
    same metric_ci the figures use, so table and figure agree when built from the
    same run). The per-fold mean/std survive as perfold_* diagnostics only --
    nothing reports them, because the RG matched-control folds are 6/6/5/5/4 over
    26 rows and a per-fold R2 on 4 points is noise.
    (The LLM ablation remains RG-only by design.)"""
    pool = load_master_csv(args.data_csv)
    coat = coating_levels(pool)
    bits, radius, kernel = _gpr_best(args)
    both = (pd.to_numeric(pool["layers"], errors="coerce").notna()
            & pd.to_numeric(pool["thickness (mm)"], errors="coerce").notna())
    ctrl = pool[both].copy()          # keep canonical master indices (no reset)
    print(f"GPR ablation matched-control rows (known layers+thickness): {len(ctrl)}/{len(pool)}")
    run_dir = os.path.join(args.out_dir, "gpr_ablation"); os.makedirs(run_dir, exist_ok=True)
    want = [args.split] if getattr(args, "split", None) else STRATEGIES

    def extra_block(df, cond):
        if cond == "baseline":
            return np.zeros((len(df), 0))
        col = "layers" if cond == "layers" else "thickness (mm)"
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(float).reshape(-1, 1)
        return v

    def restrict_fold(tr_full, te_full, strategy):
        # random/variable: rows carry master indices -> intersect exactly.
        # restricted-group: RG files have their own indices -> select by the
        # test chemistry (smiles) set.
        if strategy in ("random", "variable"):
            te = ctrl.loc[ctrl.index.intersection(te_full.index)]
            tr = ctrl.loc[ctrl.index.intersection(tr_full.index)]
        else:
            test_smiles = set(te_full["smiles"].astype(str))
            te = ctrl[ctrl["smiles"].astype(str).isin(test_smiles)]
            tr = ctrl[~ctrl["smiles"].astype(str).isin(test_smiles)]
        return tr, te

    for s in want:
        sp = get_splits(s, pool, args.rg_dir, args.out_dir)
        rows, all_pts = [], []
        for cond in ["baseline", "layers", "thickness"]:
            fold_ome, fold_r2 = [], []
            yt_pool, yp_pool, pts = [], [], []
            for k_fold, (tr_full, te_full) in enumerate(sp):
                tr, te = restrict_fold(tr_full, te_full, s)
                if len(te) == 0 or len(tr) == 0:
                    continue
                ytr, yte = target(tr), target(te)
                ftr, ntr, ctr = build_blocks(tr, coat, bits, radius)
                fte, nte, cte = build_blocks(te, coat, bits, radius)
                etr, ete = extra_block(tr, cond), extra_block(te, cond)
                ntr2 = np.hstack([ntr, etr]); nte2 = np.hstack([nte, ete])
                ok_tr = ~np.isnan(ftr).any(1) & ~np.isnan(ntr2).any(1)
                ok_te = ~np.isnan(fte).any(1) & ~np.isnan(nte2).any(1)
                if ok_tr.sum() == 0 or ok_te.sum() == 0:
                    continue
                sc = StandardScaler().fit(ntr2[ok_tr])
                Xtr = np.hstack([ftr[ok_tr], sc.transform(ntr2[ok_tr]), ctr[ok_tr]])
                Xte = np.hstack([fte[ok_te], sc.transform(nte2[ok_te]), cte[ok_te]])
                g = make_gpr(ftr.shape[1], kernel=kernel).fit(Xtr, ytr[ok_tr])
                mu, sd = g.predict(Xte, return_std=True)
                o, r = metrics(yte[ok_te], mu)
                fold_ome.append(o); fold_r2.append(r)
                yt_pool.append(yte[ok_te]); yp_pool.append(mu)
                # keep the per-point rows so gpr-ablation-fig can draw parity
                # panels without refitting. band = the GP predictive sd, the same
                # 1-sigma convention the LLM figure uses.
                te_ok = te[ok_te]
                for i in range(len(mu)):
                    pts.append(dict(strategy=s, split=k_fold, truth=float(yte[ok_te][i]),
                                    pred=float(mu[i]), band=float(sd[i]),
                                    orientation=str(te_ok["orientation"].iloc[i]).strip().lower(),
                                    fluence=float(pd.to_numeric(te_ok["fluence"],
                                                                errors="coerce").iloc[i])))
            if not fold_ome:
                print(f"  {s}/{cond}: no usable rows; skipped"); continue
            yt = np.concatenate(yt_pool); yp = np.concatenate(yp_pool)
            # same metric_ci the figure panels use, so table and figure agree by
            # construction. perfold_* kept for reference only; nothing reports them.
            p_ome, ome_ci, p_r2, r2_ci, sd = metric_ci(yt, yp)
            om = float(np.mean(fold_ome)); os_ = float(np.std(fold_ome, ddof=1) if len(fold_ome) > 1 else 0.0)
            rm = float(np.mean(fold_r2)); rs = float(np.std(fold_r2, ddof=1) if len(fold_r2) > 1 else 0.0)
            rows.append(dict(strategy=s, condition=cond,
                             OME=p_ome, OME_ci_lo=ome_ci[0], OME_ci_hi=ome_ci[1],
                             logR2=p_r2, logR2_ci_lo=r2_ci[0], logR2_ci_hi=r2_ci[1],
                             n_folds=len(fold_ome), n=int(len(yt)), OME_std_rows=sd,
                             boot_B=METRIC_CI["B"], boot_seed=METRIC_CI["seed"],
                             ci_level=METRIC_CI["level"],
                             perfold_OME_mean=om, perfold_OME_std=os_,
                             perfold_logR2_mean=rm, perfold_logR2_std=rs))
            all_pts += [dict(condition=cond, **q) for q in pts]
            print(f"  {s} {cond:9s}: OME={p_ome:.3f} [{ome_ci[0]:.3f}, {ome_ci[1]:.3f}] "
                  f"logR2={p_r2:+.3f} [{r2_ci[0]:+.3f}, {r2_ci[1]:+.3f}] n={len(yt)}")
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(run_dir, f"gpr_ablation_{s}_results.csv"), index=False)
            # per-point predictions, so gpr-ablation-fig can draw parity panels
            # without refitting -- the LLM side gets these from its saved raws.
            pd.DataFrame(all_pts).to_csv(
                os.path.join(run_dir, f"gpr_ablation_{s}_predictions.csv"), index=False)
            print(f"{s}: wrote gpr_ablation_{s}_results.csv + "
                  f"gpr_ablation_{s}_predictions.csv to {run_dir}/\n")

# ===========================================================================
# ===  shared reporting  ====================================================
# ===========================================================================
def _summarize_and_plot(rows, pooled, run_dir, tag, only=None, gpr_config=None,
                        data_hash=None, membership_hash=None, rg_split_hash=None):
    strategies = only if only is not None else STRATEGIES
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(run_dir, f"{tag}_cv_per_split.csv"), index=False)
    # dump every prediction point so plots can be redrawn later with NO inference
    pts = []
    for s in strategies:
        if s not in pooled or not pooled[s][0]:
            continue
        for k in range(len(pooled[s][0])):
            yt_k = pooled[s][0][k]; yp_k = pooled[s][1][k]; bd_k = pooled[s][2][k]
            or_k = pooled[s][4][k] if pooled[s][4] else [""] * len(yt_k)
            fl_k = pooled[s][5][k] if pooled[s][5] else [np.nan] * len(yt_k)
            for i in range(len(yt_k)):
                pts.append(dict(strategy=s, split=k, truth=yt_k[i], pred=yp_k[i],
                                band=bd_k[i], orientation=or_k[i], fluence=fl_k[i]))
    pred_csv = os.path.join(run_dir, f"{tag}_predictions.csv")
    pd.DataFrame(pts).to_csv(pred_csv, index=False)
    summ = []
    for s in strategies:
        g = res[res.strategy == s]
        if g.empty:
            continue
        yt = np.concatenate(pooled[s][0]); yp = np.concatenate(pooled[s][1])
        # BY ROW, matching the panels exactly: same metric_ci call, so the CSV and
        # the figure and this table agree when built from the same completed run.
        # OME_ci / logR2_ci are row-bootstrap percentile intervals -- the same
        # quantity for both metrics, computed identically for both models but NOT
        # paired across them (row identity is unverified). OME_std_rows
        # is the dispersion of the individual row errors: a useful diagnostic, but
        # NOT an uncertainty of the estimate, so it is kept out of the figure box.
        # perfold_* are retained for reference only; nothing reports them.
        p_ome, ome_ci, p_r2, r2_ci, sd = metric_ci(yt, yp)
        summ.append(dict(strategy=s,
                         OME=p_ome, OME_ci_lo=ome_ci[0], OME_ci_hi=ome_ci[1],
                         logR2=p_r2, logR2_ci_lo=r2_ci[0], logR2_ci_hi=r2_ci[1],
                         n_rows=int(len(yt)), boot_B=METRIC_CI["B"],
                         boot_seed=METRIC_CI["seed"], ci_level=METRIC_CI["level"],
                         OME_std_rows=sd,
                         perfold_OME_mean=g.OME.mean(), perfold_OME_std=g.OME.std(),
                         perfold_logR2_mean=g.logR2.mean(), perfold_logR2_std=g.logR2.std()))
    pd.DataFrame(summ).to_csv(os.path.join(run_dir, f"{tag}_cv_summary.csv"), index=False)
    # run provenance: exactly what config produced these results. For GPR runs
    # gpr_config carries the ACTUAL optimized (bits, radius, kernel) used, not
    # the global FP_* defaults. Row counts, dataset hash and split-membership
    # hash are recorded so a later run can validate cached results.
    if gpr_config is not None:
        prov_bits, prov_radius, prov_kernel = gpr_config
    else:
        prov_bits, prov_radius, prov_kernel = FP_BITS, FP_RADIUS, KERNEL
    per_fold_counts = {}
    for s in strategies:
        sub = pd.DataFrame(rows)
        if not sub.empty and "strategy" in sub and "n" in sub:
            per_fold_counts[s] = [int(x) for x in sub[sub.strategy == s]["n"].tolist()]
    prov = {"tag": tag, "seed": SEED, "fp_bits": prov_bits, "fp_radius": prov_radius,
            "kernel": prov_kernel, "infer_repeats": INFER_REPEATS, "var_bins": VAR_BINS,
            "n_random": N_RANDOM, "n_variable": N_VARIABLE,
            "data_hash": data_hash, "membership_hash": membership_hash,
            "rg_split_hash": rg_split_hash,
            "per_fold_counts": per_fold_counts,
            "n_points_by_strategy": {s: int((pd.DataFrame(pts).strategy == s).sum())
                                     for s in strategies if pts}}
    json.dump(prov, open(os.path.join(run_dir, f"{tag}_provenance.json"), "w"), indent=2)
    plot_from_predictions(pd.DataFrame(pts), run_dir, tag, only=strategies)
    print(f"wrote {tag} CV summary + predictions + separate parity plots to {run_dir}/")

def _savefig_multi(fig, svg_path):
    """Save a figure as svg + pdf + eps (publication formats).

    Exports at the figure's NOMINAL width, not a tight bounding box. This matters
    because scaled_style() derives font sizes from the nominal width while LaTeX
    scales by the EXPORTED width: if the two disagree, the width-invariance is
    wrong by exactly that much. Measured on this pipeline's own geometries,
    bbox_inches="tight" moved the exported width by -2.6% (single parity, with an
    outside legend and a colourbar) to +0.2% (the 3-panel ablation row) -- small
    beside the 3.5x spread the scaling removes, but free to eliminate.

    The cost of dropping "tight" is that content outside the canvas is CLIPPED
    rather than cropped into view. Every figure here calls tight_layout() first,
    which keeps the outside legends inside the canvas, but that is checked rather
    than assumed: the overflow is measured and reported. It does not block -- a
    warned figure is still written, and can be looked at."""
    base = os.path.splitext(svg_path)[0]
    # Place any deferred parity legends/metrics boxes now -- AFTER the caller's
    # tight_layout() -- so positions hold against the FINAL axes geometry. No-op
    # on figures without parity panels.
    finalize_parity_panels(fig)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    over = {"left": max(0.0, -bb.x0), "right": max(0.0, bb.x1 - w),
            "bottom": max(0.0, -bb.y0), "top": max(0.0, bb.y1 - h)}
    spill = {k: v for k, v in over.items() if v > 0.01}
    if spill:
        print(f"  WARNING {os.path.basename(base)}: content extends past the "
              f"{w:.2f}x{h:.2f}in canvas and WILL BE CLIPPED -- "
              + ", ".join(f"{k} +{v:.2f}in" for k, v in spill.items())
              + ". Widen figsize or pull the legend/colourbar in; the export is "
                "deliberately not tight-cropped so the width stays predictable.")
    for fmt in ("svg", "pdf", "eps"):
        fig.savefig(f"{base}.{fmt}", format=fmt)

# ------------------------------------------------------- tuning figure --------
# tuning-fig only. Curve colours/markers are specific to the sweep panels and do
# NOT belong in PLOT_STYLE, which configures the parity panels; the sweep axes
# still inherit their fonts and grid from PLOT_STYLE so the figure stays
# stylistically identical to the rest of the paper.
TUNING_FIG = {
    "logr2_color":   "#1f77b4",
    "ome_color":     "#d62728",
    "logr2_marker":  "o",
    "ome_marker":    "s",
    "line_width":    1.5,
    "marker_size":   6,
    "figsize":       (13.0, 10.0),
    "height_ratios": [1.2, 0.8],   # parity row taller than the sweep row
}

ORIENT_MARKERS = {"ram": "o", "zenith": "^", "nadir": "s", "wake": "D", "unknown": "X"}
SPLIT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

# ------------------------------------------------------- dataset figure -------
# Standalone dataset-description figure (data-fig mode). Its own style block:
# these are NOT the parity-plot fonts, so tuning one never disturbs the other.
DATA_FIG = {
    # orientation colours sampled from the schematic. Wake/zenith/nadir are the
    # modal pixel of each arrow body; ram uses the darker Ram LABEL colour --
    # the arrow's #1850FC reads as an over-bright electric blue/purple in bars.
    "orient_colors": {
        "ram":    "#184EA2",
        "wake":   "#EC0E13",
        "zenith": "#FEB901",
        "nadir":  "#4CC827",
    },
    "orient_order":      ["ram", "wake", "zenith", "nadir"],
    "fluence_bins":      20,
    "ey_bins":           40,
    "axis_label_size":   16,
    "tick_label_size":   14,
    "legend_size":       14,
    "bar_label_size":    9,     # rotated PSMILES-count labels in panel (a)
    "annot_size":        13,    # the "N chemistries with a single sample" note
    "singleton_color":   "#333333",
    "figsize":           (12, 7),
    # True  -> label every chemistry in panel (a)
    # False -> label only those occurring more than once, and annotate the
    #          singleton tail with a bracket instead of labelling it
    "label_all":         True,
    # panel (a) must be at least as tall as its rotated y-label, or the label
    # overflows the axes and the top of the word gets clipped on save.
    "height_ratios":     [1.0, 1.5],
}

# Short display names keyed by PSMILES. Abbreviation is preferred over trade
# name (PVF not Tedlar, PET not Mylar, ETFE not Tefzel, ...). The exceptions are
# Kapton / Upilex-S / Kevlar / CP1, where the trade name IS the common name and
# the monomer code (PMDA-ODA / BPDA-PDA / PPTA) would be less recognizable.
# Every chemistry occurring more than once must appear here; data-fig fails
# loudly if one is missing rather than falling back to an unreadable PSMILES.
PSMILES_NAMES = {
    # Corrected identities for three rows whose legacy PSMILES were erroneous:
    # the structures were intended as Kevlar, Nomex and PFA (documented in the
    # handoff). Labelled by their true material name here.
    "[*]CONc1ccc(NOCc2ccc([*])cc2)cc1": "Kevlar",
    "[*]OCCOC(=O)c1cccc(C([*])=O)c1": "Nomex",
    "[*]C(OC(F)(F)C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)C([*])(F)F": "PFA",
    "[*]C(=O)c1ccc(Oc2ccc(Oc3ccc([*])cc3)cc2)cc1": "PEEK",
    "[*]C(F)(F)C(F)(F)C(F)(F)C([*])(F)C(F)(F)F": "FEP",
    "[*]C(F)(F)C([*])(F)Cl": "PCTFE",
    "[*]C(F)(F)C([*])(F)F": "PTFE",
    "[*]CC([*])(C)C(=O)OC": "PMMA",
    "[*]CC([*])(F)F": "PVDF",
    "[*]CC([*])C": "PP",
    "[*]CC([*])CC(C)C": "PMP",
    "[*]CC([*])Cl": "PVC",
    "[*]CC([*])F": "PVF",
    "[*]CC([*])O": "PVA",
    "[*]CC([*])c1ccccc1": "PS",
    "[*]CCC(F)(Cl)C([*])(F)F": "ECTFE",
    "[*]CCC(F)(F)C([*])(F)F": "ETFE",
    "[*]COC[*]": "PEO",
    "[*]CO[*]": "POM",
    "[*]C[*]": "PE",
    "[*]OCCOC(=O)c1ccc(C([*])=O)cc1": "PET",
    "[*]OCCOC(=O)c1ccc2cc(C([*])=O)ccc2c1": "PEN",
    "[*]c1ccc(Oc2ccc(-c3ccc(Oc4ccc(S([*])(=O)=O)cc4)cc3)cc2)cc1": "PPSU",
    "[*]c1ccc(Oc2ccc(-n3c(=O)c4cc5c(=O)n([*])c(=O)c5cc4c3=O)cc2)cc1": "Kapton",
    "[*]c1ccc(Oc2ccc(S(=O)(=O)c3ccc(Oc4ccc(C([*])(C)C)cc4)cc3)cc2)cc1": "PSU",
    "[*]c1ccc2c(c1)C(=O)N(c1ccc(N3C(=O)c4ccc([*])cc4C3=O)cc1)C2=O": "Upilex-S",
    "[*]c1ccc2c(c1)C(=O)N(c1ccc(Oc3ccc(C(c4ccc(Oc5ccc(N6C(=O)c7ccc(C([*])(C(F)(F)F)C(F)(F)F)cc7C6=O)cc5)cc4)(C(F)(F)F)C(F)(F)F)cc3)cc1)C2=O": "CP1",
    "[*]c1cccc(-c2nc3ccc(-c4ccc5nc([*])[nH]c5c4)cc3[nH]2)c1": "PBI",
    # --- singletons (n=1): abbreviation where one exists, else trade name ---
    "[*]CC([*])c1cccc(C)c1": "PVT",
    "[*]c1ccc(Oc2ccc(N3Cc4cc5c(cc4C3)CN(c3ccc(Oc4ccc(C([*])(C(F)(F)F)C(F)(F)F)cc4)cc3)C5)cc2)cc1": "Eymyd-F",
    "[*]Oc1ccc(-c2ccc(OC(=O)c3ccc(C(=O)Oc4ccc(C([*])=O)cc4)cc3)cc2)cc1": "Xydar",
    "[*]C(F)(F)C(F)(F)C(F)(F)C(F)(F)C1(F)OC(C(F)(F)F)(C(F)(F)F)OC1(F)C1(F)OC(C(F)(F)F)(C(F)(F)F)OC1(F)C1(F)OC(C(F)(F)F)(C(F)(F)F)OC1([*])F": "Teflon AF",
    "[*]C=C(/C=C(\\CC(C#N)C(=C)C([*])=C)c1ccccc1)c1ccccc1": "ABS",
    "[*]c1ccc(-c2nc3cc4nc([*])oc4cc3o2)cc1": "PBO",
    "[*]c1ccc(OC(=O)Oc2ccc(C([*])(C)C)cc2)cc1": "PC",
    "[*]CC([*])C#N": "PAN",
    "[*]CCCCC(=O)NCCCCCCNC([*])=O": "PA66",
    "[*]CCCCCNC([*])=O": "PA6",
    "[*]Nc1ccc(Cc2ccc(NC(=O)OCCOC([*])=O)cc2)cc1": "PU",
    "[*]OC1C(COC(C)=O)OC([*])C(OC(C)=O)C1OC(C)=O": "CA",
    "[*]OCCCCOC(=O)c1ccc(C([*])=O)cc1": "PBT",
    "[*]c1ccc(Oc2ccc3c(c2)C(=O)N(c2cccc(N4C(=O)c5ccc(Oc6ccc(C([*])(C)C)cc6)cc5C4=O)c2)C3=O)cc1": "PEI",
    "[*]C(F)(F)C(F)(F)C(F)(F)C(F)(OC(F)(F)C(F)(F)C(F)(F)F)C(F)(F)C([*])(F)OC(F)(F)F": "MFA",
    "[*]c1ccc(Oc2ccc(S([*])(=O)=O)cc2)cc1": "PES",
    "[*]Nc1ccc(Cc2ccc(N3C(=O)c4ccc(C([*])=O)cc4C3=O)cc2)cc1": "PAI",
    "[*]OC1C(O[N+](=O)[O-])OC(OC2C(O[N+](=O)[O-])OC([*])C(O[N+](=O)[O-])C2O[N+](=O)[O-])C(O[N+](=O)[O-])C1O[N+](=O)[O-]": "CN",
    "[*]C(=O)c1ccc(-n2c(=O)c3cc4c(=O)n(-c5ccc([*])cc5)c(=O)c4cc3c2=O)cc1": "PMDA-pp\'-DABP",
    "[*]C(=O)c1ccc(N2C(=O)c3ccc(C(=O)c4ccc5c(c4)C(=O)N(c4ccc([*])cc4)C5=O)cc3C2=O)cc1": "BDTA-pp\'-DABP",
    "[*]C(=O)c1ccc2c(c1)C(=O)N(c1ccc(-c3ccc(N4C(=O)c5ccc([*])cc5C4=O)cc3)cc1)C2=O": "BTDA-benzidine",
    "[*]C(=O)c1ccc2c(c1)C(=O)N(c1ccc(Oc3ccc(N4C(=O)c5ccc([*])cc5C4=O)cc3)cc1)C2=O": "BTDA-pp\'-ODA",
    "[*]C(=O)c1ccc2c(c1)C(=O)N(c1ccc3c(c1)Cc1cc(N4C(=O)c5ccc([*])cc5C4=O)ccc1-3)C2=O": "BTDA-DAF",
    "[*]C(=O)c1ccc2c(c1)C(=O)N(c1cccc(Cc3cccc(N4C(=O)c5ccc([*])cc5C4=O)c3)c1)C2=O": "BTDA-mm\'-MDA",
    "[*]C(=O)c1ccc2c(c1)C(=O)N(c1cccc(S(=O)(=O)c3cccc(N4C(=O)c5ccc([*])cc5C4=O)c3)c1)C2=O": "BTDA-mm\'-DDSO2",
    "[*]C(=O)c1cccc(-n2c(=O)c3cc4c(=O)n(-c5cccc([*])c5)c(=O)c4cc3c2=O)c1": "PMDA-DAB",
    "[*]c1ccc(Cc2ccc(-n3c(=O)c4cc5c(=O)n([*])c(=O)c5cc4c3=O)cc2)cc1": "PMDA-pp\'-MDA",
    "[*]Nc1ccc(NC(=O)c2ccc(C([*])=O)cc2)cc1": "Kevlar",
    "[*]Nc1cccc(NC(=O)c2cccc(C([*])=O)c2)c1": "Nomex",
    "[*]C(F)(F)C(F)(F)C(F)(F)C([*])(F)OC(F)(F)C(F)(F)C(F)(F)F": "PFA",
}

# ---------------------------------------------------------------------------
# In-panel legend placement. The legend ALWAYS sits INSIDE the axes -- there is
# no outside fallback (the outside fallback is what produced clipped/floating
# legends between panels and under colourbars).
#
# A parity panel's data hugs the corner-to-corner 1:1 line, so the two
# anti-diagonal corners (upper-left, lower-right) are the only reliably clear
# regions -- and the metrics box already holds one of them. So the legend and the
# box compete for those two corners, and the ladder is:
#
#   1. legend upper-left,  metrics bottom-right   (the default; both data-clear)
#   2. legend bottom-right, metrics upper-left    (swap the two corners)
#   3. any other inside slot with ZERO overlap
#   4. no data-clear slot exists: legend goes upper-left INSIDE ANYWAY, over the
#      data, with a MORE TRANSPARENT frame so the covered points and error bars
#      show through; the metrics box takes a corner that avoids the legend
#      (data-clear if one exists, else semi-transparent over data).
#
# The axes are never expanded (that would change the visual scale and fight
# tuning-fig's shared lim_override), and the colourbar is never moved (it is a
# quantitative scale, not a legend). The 1:1 line and grid MAY be covered -- only
# datapoints and error bars are protected -- but the diagonal is used as a tie-
# break preference so a legend does not sit on it when a clear slot exists.
#
# The overlap test is CONSERVATIVE, not "pixel-exact": each datapoint's footprint
# is the marker's actual path extent (not an assumed ms/2) inflated to cover the
# error-bar caps, and the metrics Text is included via its rendered extent. There
# is no connecting line to worry about -- every series is drawn marker-only or
# with fmt="none" -- and 10**(yp +/- band) is always > 0, so the log transform of
# the bar extremes is always finite.
# ---------------------------------------------------------------------------
def _marker_radius_px(marker, ms, fig):
    """Conservative pixel radius of one marker of size ms (points), from the
    marker path's true extent rather than an assumed half-size."""
    from matplotlib.markers import MarkerStyle
    try:
        m = MarkerStyle(marker)
        ext = m.get_path().transformed(m.get_transform()).get_extents()
        half = max(abs(ext.x0), abs(ext.x1), abs(ext.y0), abs(ext.y1))
    except Exception:
        half = 0.7
    return half * ms * fig.dpi / 72.0

def _occupied_boxes(ax, fig, xt, xp, yerr, markers_ms, capsize_pt, extra_texts):
    """Pixel bboxes of everything the legend must not cover: every point's marker
    (inflated by its own radius) unioned with its vertical error bar and caps,
    plus each protected Text artist."""
    from matplotlib.transforms import Bbox
    r = fig.canvas.get_renderer()
    P  = ax.transData.transform(np.column_stack([xt, xp]))
    LO = ax.transData.transform(np.column_stack([xt, xp - yerr[0]]))
    HI = ax.transData.transform(np.column_stack([xt, xp + yerr[1]]))
    cap_px = capsize_pt * fig.dpi / 72.0
    boxes = []
    for (px, _), (_, ly), (_, hy), (mk, ms) in zip(P, LO, HI, markers_ms):
        pad = max(_marker_radius_px(mk, ms, fig), cap_px)
        boxes.append(Bbox([[px - pad, min(ly, hy) - pad],
                           [px + pad, max(ly, hy) + pad]]))
    for t in extra_texts:
        try:
            boxes.append(t.get_window_extent(r))
        except Exception:
            pass
    return boxes

# (loc string, axes-fraction anchor) for each inside slot, corners inset slightly.
# Order encodes the ruled preference: the two anti-diagonal corners first.
_LEGEND_SLOTS = [
    ("upper left",  (0.02, 0.98)),
    ("lower right", (0.98, 0.02)),
    ("upper right", (0.98, 0.98)),
    ("lower left",  (0.02, 0.02)),
    ("center left", (0.02, 0.50)),
]
def _overlaps_any(bbox, boxes):
    return any(bbox.overlaps(b) for b in boxes)

def _overlap_area(bbox, boxes):
    """Total intersection area (px^2) of bbox with the obstacle boxes -- the
    quantity minimized when NO overlap-free slot exists."""
    from matplotlib.transforms import Bbox
    a = 0.0
    for b in boxes:
        ib = Bbox.intersection(bbox, b)
        if ib is not None:
            a += max(ib.width, 0.0) * max(ib.height, 0.0)
    return a

def _corner_coords(corner):
    return {
        "upper left":  (0.03, 0.97, "top",    "left"),
        "upper right": (0.97, 0.97, "top",    "right"),
        "lower left":  (0.03, 0.03, "bottom", "left"),
        "lower right": (0.97, 0.03, "bottom", "right"),
    }[corner]

def _equalize_parity_axes(fig, axes, suptitle_in, extra_bot_in=0.0):
    """Final layout pass for the ablation figures: at the FINAL font sizes,
    measure each cell's decoration extents (ylabel/ticks left, title top,
    xlabel/ticks bottom, and -- for panels with a colourbar -- the colourbar
    strip plus its labels on the right), then place every axes MANUALLY as an
    identical square of the largest side the fixed figure width allows.
    tight_layout is bypassed: it re-derives slot sizes and, combined with
    set_box_aspect and the colourbar divider, yields unequal panels and dead
    inter-panel space no matter what ratios it is given. A colourbar panel's
    position box is widened by exactly the divider's take (4% + 0.08in), so its
    PARENT axes still comes out at `side` like every other panel. Figure WIDTH
    never changes, so scaled_style's font contract holds. Returns fig height."""
    from matplotlib.transforms import Bbox
    fig.tight_layout()          # realize one geometry to measure against
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    nrows, ncols = axes.shape
    DL = [0.0] * ncols
    DRn = [0.0] * ncols          # right decorations, non-colourbar cells
    CBR = [0.0] * ncols          # labels right of the colourbar, cb cells
    has_cb = [False] * ncols
    DT = [0.0] * nrows; DB = [0.0] * nrows
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]
            if not ax.get_visible():
                continue
            ab = ax.get_window_extent(r)
            fb = ax.get_tightbbox(r)
            cax = getattr(ax, "_parity_colorbar_ax", None)
            if cax is not None:
                cab = cax.get_window_extent(r)
                cfb = cax.get_tightbbox(r)
                fb = Bbox.union([fb, cfb])
                has_cb[j] = True
                CBR[j] = max(CBR[j], (fb.x1 - cab.x1) / fig.dpi)
            else:
                DRn[j] = max(DRn[j], (fb.x1 - ab.x1) / fig.dpi)
            DL[j] = max(DL[j], (ab.x0 - fb.x0) / fig.dpi)
            DT[i] = max(DT[i], (fb.y1 - ab.y1) / fig.dpi)
            DB[i] = max(DB[i], (ab.y0 - fb.y0) / fig.dpi)
    pad, edge = 0.12, 0.15
    fig_w = fig.get_size_inches()[0]
    # side depends on the colourbar strip (4% of side + 0.08in), so iterate --
    # converges in 2-3 rounds
    side = 4.5
    for _ in range(4):
        DR = [max(DRn[j], (0.04 * side + 0.08 + CBR[j]) if has_cb[j] else 0.0)
              for j in range(ncols)]
        side = (fig_w - 2 * edge - sum(DL) - sum(DR) - pad * (ncols - 1)) / ncols
    fig_h = (2 * edge + suptitle_in + extra_bot_in + pad * (nrows - 1)
             + sum(DT) + sum(DB) + side * nrows)
    fig.set_size_inches(fig_w, fig_h)
    xs = []
    x = edge
    for j in range(ncols):
        x += DL[j]; xs.append(x); x += side + DR[j] + pad
    y = fig_h - edge - suptitle_in
    for i in range(nrows):
        y -= DT[i]
        y_bot = y - side
        for j in range(ncols):
            ax = axes[i][j]
            # a colourbar cell's box is widened by the divider's exact take, so
            # the parent axes still lands at `side`
            bw = (side * 1.04 + 0.08) if getattr(ax, "_parity_colorbar_ax", None) \
                 is not None else side
            ax.set_position([xs[j] / fig_w, y_bot / fig_h,
                             bw / fig_w, side / fig_h])
        y = y_bot - DB[i] - pad
    return fig_h

def finalize_parity_panels(fig):
    """Place the legend and metrics box on every parity axes that stashed a
    request, AFTER the figure's final tight_layout(). Guarantees neither artist
    covers a datapoint or an error bar, and that they do not cover each other.
    Called from _savefig_multi so no builder can forget it and it always runs
    post-layout.

    Ladder: 1) legend upper-left + box lower-right; 2) swap; 3) any other inside
    legend slot with the box in a tested-clear corner; 4) legend upper-left
    INSIDE over the data with a semi-transparent frame (never outside), box in a
    legend-clear corner (data-clear preferred, else semi-transparent). There is
    no silent overlap and no duplicate box (every trial artist is removed
    first)."""
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        req = getattr(ax, "_parity_placement", None)
        if not req:
            continue
        data_occ = _occupied_boxes(ax, fig, req["xt"], req["xp"], req["yerr"],
                                   req["markers_ms"], req["capsize_pt"], [])
        # (No colourbar handling needed here any more: the legend never leaves
        # the axes, and every inside slot is strictly within the panel, so it
        # cannot collide with a colourbar that lives outside-right.)

        def box_at(corner):
            x, y, va, ha = _corner_coords(corner)
            t = ax.text(x, y, req["metrics_text"], transform=ax.transAxes, va=va,
                        ha=ha, fontsize=req["metrics_fontsize"],
                        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=.75))
            fig.canvas.draw()
            return t, t.get_window_extent(r)

        def clear_corner():
            """(corner, is_clear): a box corner clearing the data, preferring the
            anti-diagonal corners; is_clear=False -> none is clear."""
            for c in ("lower right", "upper left", "upper right", "lower left"):
                t, bb = box_at(c); hit = _overlaps_any(bb, data_occ); t.remove()
                if not hit:
                    return c, True
            return "lower right", False

        def draw_box(corner, box_alpha=0.75):
            # box_alpha < 0.95 is used ONLY when the box has no clear corner and
            # must sit over data: a semi-transparent face lets the covered points
            # and error bars show through, so an unavoidable overlap stops HIDING
            # data. A clear corner keeps the crisp opaque box.
            x, y, va, ha = _corner_coords(corner)
            ax.text(x, y, req["metrics_text"], transform=ax.transAxes, va=va,
                    ha=ha, fontsize=req["metrics_fontsize"],
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=box_alpha))

        placed = False
        if req["show_legend"]:
            def legend_bbox(leg, loc, anchor):
                leg.set_loc(loc); leg.set_bbox_to_anchor(anchor, transform=ax.transAxes)
                fig.canvas.draw()
                return leg.get_window_extent(r)

            def try_inside(lloc, lanchor, bcorner):
                t, bb = box_at(bcorner)
                if _overlaps_any(bb, data_occ):
                    t.remove(); return False
                leg = ax.legend(**req["legend_kw"])
                lbb = legend_bbox(leg, lloc, lanchor)
                if _overlaps_any(lbb, data_occ) or lbb.overlaps(bb):
                    leg.remove(); t.remove(); return False
                return True

            for lloc, lanchor, bcorner in (
                    ("upper left",  (0.02, 0.98), "lower right"),
                    ("lower right", (0.98, 0.02), "upper left")):
                if try_inside(lloc, lanchor, bcorner):
                    placed = True; break
            if not placed:
                for lloc, lanchor in _LEGEND_SLOTS:
                    for bcorner in ("upper left", "upper right", "lower left", "lower right"):
                        if try_inside(lloc, lanchor, bcorner):
                            placed = True; break
                    if placed:
                        break
            if not placed:
                # FORCED INSIDE (rung 4): no data-clear slot exists, so the
                # legend stays inside and takes the slot with the MINIMUM total
                # overlap area against the data, with a more transparent frame
                # so anything it covers stays visible. Never outside the axes.
                ok = dict(req["legend_kw"]); ok.update(framealpha=0.65)
                leg = ax.legend(**ok)
                best = None
                for lloc, lanchor in _LEGEND_SLOTS:
                    lbb = legend_bbox(leg, lloc, lanchor)
                    area = _overlap_area(lbb, data_occ)
                    if best is None or area < best[0]:
                        best = (area, lloc, lanchor)
                _, lloc, lanchor = best
                lbb = legend_bbox(leg, lloc, lanchor)
                # metrics box: among corners clear of the LEGEND, take the one
                # with the least data overlap (opaque if zero, else semi-
                # transparent).
                bcorner, barea = None, None
                for c in ("lower right", "upper left", "upper right", "lower left"):
                    t, bb = box_at(c)
                    hit_leg = bb.overlaps(lbb)
                    area = _overlap_area(bb, data_occ)
                    t.remove()
                    if hit_leg:
                        continue
                    if bcorner is None or area < barea:
                        bcorner, barea = c, area
                if bcorner is None:
                    bcorner, barea = "lower right", 1.0
                draw_box(bcorner, box_alpha=0.75 if barea == 0 else 0.65)
                ttl = ax.get_title() or "panel"
                print(f"  NOTE [{ttl}]: no overlap-free slot; legend placed "
                      f"inside {lloc} (minimum-overlap slot) with a semi-"
                      f"transparent frame; metrics box {bcorner}.")
        else:
            # No legend here (a sibling panel carries the shared one). The box is
            # STILL tested against the data -- the old show_legend=False path
            # dropped it lower-right blindly.
            bcorner, is_clear = clear_corner()
            draw_box(bcorner, box_alpha=0.75)
            if not is_clear:
                ttl = ax.get_title() or "panel"
                print(f"  WARNING [{ttl}]: no corner clears the data; metrics box "
                      f"placed {bcorner} SEMI-TRANSPARENT so points show through.")
        try:
            del ax._parity_placement
        except Exception:
            pass
def _draw_parity_panel(ax, fig, sub, s, title, show_legend=True, show_colorbar=True,
                       show_ylabel=True, show_xlabel=True, pooled=True,
                       lim_override=None, style=None):
    """Draw one parity panel onto a given axis. RG: one colour AND one marker per
    split. random/variable: orientation -> marker shape, AO fluence -> viridis
    colour. Returns the metrics (ome, r2) for the panel.

    `style` is a scaled_style() dict; it defaults to the canonical PLOT_STYLE.
    Passing it explicitly (rather than mutating the global) is what lets a wide
    multi-panel figure render its text at the same size on the page as a narrow
    one without any figure being able to disturb another's fonts.

    Metrics are BY ROW and both carry the SAME kind of uncertainty: a
    row-bootstrap 95% percentile CI from metric_ci(), shown in brackets (not
    +/-, since the intervals are asymmetric). OME's point estimate is the mean of
    the per-row absolute errors; log-R2 is computed once over every row. The
    dispersion of the row errors is a diagnostic and lives in the summary CSVs as
    OME_std_rows -- it is NOT an uncertainty of the estimate and does not belong
    beside one. Nothing is averaged across folds: the folds are unequal (RG
    6/6/6/6/5 over 29 rows; the matched-control ablation 6/6/5/5/4 over 26), so a
    fold-averaged statistic weights a 4-row fold like a 6-row one, and a per-fold
    R2 on a handful of points is noise."""
    st = PLOT_STYLE if style is None else style
    yt = sub["truth"].to_numpy(float); yp = sub["pred"].to_numpy(float)
    band = sub["band"].to_numpy(float)
    ome_mean, ome_ci, r2_mean, r2_ci, ome_std = metric_ci(yt, yp)
    xt = 10.0 ** yt; xp = 10.0 ** yp
    lo = 10.0 ** (yp - band); hi = 10.0 ** (yp + band)
    yerr = np.vstack([xp - lo, hi - xp])
    allv = np.concatenate([xt, xp])
    # lim_override lets a caller force the SAME axis range across sibling panels.
    # Without it each panel derives its own range, so two arms being compared can
    # look more or less alike purely because their limits differ. Default None
    # keeps every existing figure byte-identical.
    lim = (list(lim_override) if lim_override is not None else
           [10 ** np.floor(np.log10(allv.min())), 10 ** np.ceil(np.log10(allv.max()))])

    # markers_ms[i] = (marker, size_pt) for point i, IN ROW ORDER of sub, so the
    # overlap test can bound each point's footprint by its own marker.
    markers_ms = [None] * len(sub)
    legend_kw = dict(fontsize=st["legend_size"], framealpha=.75, edgecolor="0.6",
                     borderpad=.45, labelspacing=.4, handletextpad=.5,
                     borderaxespad=.4)
    if s == "restricted-group":
        pos = {ix: i for i, ix in enumerate(sub.index.to_numpy())}
        for kk, g in sub.groupby("split"):
            gi = [pos[x] for x in g.index.to_numpy()]
            # circles for EVERY split: shape carries orientation meaning in the
            # random/variable panels, so any non-circle here would read as an
            # orientation claim. Splits are distinguished by colour only.
            mk = "o"
            ax.errorbar(xt[gi], xp[gi], yerr=yerr[:, gi], fmt=mk, ms=6, alpha=.9,
                        lw=.7, capsize=2, color=SPLIT_COLORS[int(kk) % len(SPLIT_COLORS)],
                        label=f"Split {int(kk)+1}", zorder=2)
            for i in gi:
                markers_ms[i] = (mk, 6)
    else:
        orient = sub["orientation"].astype(str).str.strip().str.lower().to_numpy()
        flu = sub["fluence"].to_numpy(float)
        lflu = np.log10(np.where(flu > 0, flu, np.nan))
        vmin, vmax = np.nanmin(lflu), np.nanmax(lflu)
        sc_obj = None
        for o, mk in ORIENT_MARKERS.items():
            sel = orient == o
            if not sel.any():
                continue
            ax.errorbar(xt[sel], xp[sel], yerr=yerr[:, sel], fmt="none", ecolor="grey",
                        alpha=.35, lw=.5, capsize=2, zorder=1)
            sc_obj = ax.scatter(xt[sel], xp[sel], c=lflu[sel], cmap="viridis",
                                vmin=vmin, vmax=vmax, marker=mk, s=46,
                                edgecolors="k", linewidths=.3, zorder=2, label=o)
        # scatter s is area in points^2; sqrt gives the side/diameter in points.
        for i, o in enumerate(orient):
            markers_ms[i] = (ORIENT_MARKERS.get(o, "X"), float(np.sqrt(46)))
        # Legend handles built by hand in ONE neutral colour: point colour
        # encodes fluence (the colourbar), so coloured legend markers would
        # falsely suggest colour maps to orientation. Shape alone differs.
        from matplotlib.lines import Line2D
        _hd = [Line2D([], [], linestyle="", marker=mk, markersize=7,
                      markerfacecolor="0.7", markeredgecolor="k",
                      markeredgewidth=.4, label=o)
               for o, mk in ORIENT_MARKERS.items()
               if (orient == o).any()]
        if sc_obj is not None and show_colorbar:
            # Explicit cax via append_axes: a colourbar made with colorbar(ax=ax)
            # STEALS width from the parity axes (so panels with a colourbar end up
            # narrower than those without) AND lands wherever matplotlib decides,
            # which is exactly where the outside-legend fallback wants to go --
            # that was the legend-under-the-fluence-scale collision. A dedicated
            # cax keeps the panel full-width and gives the colourbar a KNOWN axes
            # the finalizer can treat as an obstacle and offset the legend past.
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="4%", pad=0.08)
            cb = fig.colorbar(sc_obj, cax=cax)
            cb.set_label("log10 AO fluence (atoms/cm$^2$)",
                         fontsize=st["colorbar_label_size"])
            cb.ax.tick_params(labelsize=st["colorbar_tick_size"])
            ax._parity_colorbar_ax = cax   # obstacle + fallback anchor for finalizer
        legend_kw.update(title="orientation", title_fontsize=st["legend_title_size"],
                         handles=_hd)

    # Axes and limits BEFORE placement: the overlap test transforms data through
    # ax.transData, which is only final once the scale and limits are set.
    ax.plot(lim, lim, "k--", lw=1.2, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    # Identical square proportions for EVERY parity panel, in every figure,
    # regardless of the figure's overall width or panel count. This is what makes
    # the panels look the same shape across images (the aspect half of the ask);
    # set_box_aspect fixes the axes BOX to 1:1 without touching the data limits.
    ax.set_box_aspect(1)
    ax.grid(True, which="both", ls=":", lw=.5, alpha=.5)
    ax.tick_params(labelsize=st["tick_label_size"])
    if show_xlabel:
        ax.set_xlabel(st["xlabel"], fontsize=st["axis_label_size"])
    if show_ylabel:
        ax.set_ylabel(st["ylabel"], fontsize=st["axis_label_size"])
    if title:
        ax.set_title(title, fontsize=st["title_size"], pad=st["title_pad"])

    n_folds = sub["split"].nunique()
    # Brackets, not +/-: percentile CIs, asymmetric (R2 <= 1, OME >= 0). The
    # reference figure's box shows a bare OME/LogR2 with no interval; that is NOT
    # copied -- Sec 1.2 requires the CI and predates that figure.
    metrics_text = (f"OME = {ome_mean:.3f} [{ome_ci[0]:.3f}, {ome_ci[1]:.3f}]\n"
                    f"LogR$^2$ = {r2_mean:.3f} [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]\n"
                    f"{n_folds} folds, n = {len(yt)} rows")
    # DEFER placement to after the caller's final tight_layout(): tight_layout
    # rescales the axes, so any position computed here is measured against a
    # transform that no longer holds at render time. We
    # stash what placement needs; finalize_parity_panels(fig), invoked from
    # _savefig_multi after layout, does the work -- always testing BOTH the legend
    # and the metrics box against the data, whether or not this panel shows a
    # legend (closing the show_legend=False box-untested hole too).
    ax._parity_placement = dict(
        show_legend=bool(show_legend), legend_kw=legend_kw,
        metrics_text=metrics_text, metrics_fontsize=st["metrics_size"],
        xt=xt, xp=xp, yerr=yerr, markers_ms=markers_ms, capsize_pt=2)
    return ome_mean, r2_mean

def plot_from_predictions(df, run_dir, tag, only=None):
    """Draw the separate parity plots (one file per strategy) from a predictions
    dataframe. No inference -- pure re-plot, free to call any number of times."""
    for s in (only if only is not None else STRATEGIES):
        sub = df[df.strategy == s]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 5))
        _draw_parity_panel(ax, fig, sub, s, _pretty_title(tag, s),
                           style=scaled_style(5.5))
        fig.tight_layout()
        fname = f"{tag}_parity.svg" if tag.endswith(s) else f"{tag}_parity_{s}.svg"
        _savefig_multi(fig, os.path.join(run_dir, fname))
        plt.close(fig)

def _parity(y, mu, band, title, path, note, orient=None, flu=None, style=None):
    """Single-panel parity for the production models. Kept separate from
    _draw_parity_panel because it takes arrays + a caller-built note rather than
    a dataframe it scores itself, but it consumes the SAME scaled_style() dict --
    otherwise the production figures would be the one pair in the paper whose
    typography drifted."""
    st = PLOT_STYLE if style is None else style
    fig, ax = plt.subplots(figsize=(5.5, 5))
    xt = 10.0 ** np.asarray(y, float); xp = 10.0 ** np.asarray(mu, float)
    band = np.asarray(band, float)
    lo = 10.0 ** (np.asarray(mu, float) - band); hi = 10.0 ** (np.asarray(mu, float) + band)
    yerr = np.vstack([xp - lo, hi - xp])
    allv = np.concatenate([xt, xp])
    lim = [10 ** np.floor(np.log10(allv.min())), 10 ** np.ceil(np.log10(allv.max()))]
    ORIENT_MARKERS = {"ram": "o", "zenith": "^", "nadir": "s", "wake": "D", "unknown": "X"}
    markers_ms = None
    has_legend = orient is not None and flu is not None
    if has_legend:
        orient = np.asarray(orient); flu = np.asarray(flu, float)
        lflu = np.log10(np.where(flu > 0, flu, np.nan))
        vmin, vmax = np.nanmin(lflu), np.nanmax(lflu)
        sc_obj = None
        for o, mk in ORIENT_MARKERS.items():
            sel = orient == o
            if not sel.any():
                continue
            ax.errorbar(xt[sel], xp[sel], yerr=yerr[:, sel], fmt="none", ecolor="grey",
                        alpha=.35, lw=.5, capsize=2, zorder=1)
            sc_obj = ax.scatter(xt[sel], xp[sel], c=lflu[sel], cmap="viridis",
                                vmin=vmin, vmax=vmax, marker=mk, s=46,
                                edgecolors="k", linewidths=.3, zorder=2, label=o)
        if sc_obj is not None:
            # explicit cax (see _draw_parity_panel): keeps the panel full-width and
            # gives the finalizer a known colourbar bbox to avoid.
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
            cb = fig.colorbar(sc_obj, cax=cax); cb.set_label("log10 AO fluence (atoms/cm$^2$)",
                              fontsize=st["colorbar_label_size"])
            cb.ax.tick_params(labelsize=st["colorbar_tick_size"])
            ax._parity_colorbar_ax = cax
        markers_ms = [(ORIENT_MARKERS.get(o, "X"), float(np.sqrt(46))) for o in orient]
        # Legend handles built by hand in ONE neutral colour: point colour
        # encodes fluence (the colourbar), so coloured legend markers would
        # falsely suggest colour maps to orientation. Shape alone differs.
        from matplotlib.lines import Line2D
        _hd = [Line2D([], [], linestyle="", marker=mk, markersize=7,
                      markerfacecolor="0.7", markeredgecolor="k",
                      markeredgewidth=.4, label=o)
               for o, mk in ORIENT_MARKERS.items()
               if (orient == o).any()]
    else:
        ax.errorbar(xt, xp, yerr=yerr, fmt="o", ms=5, alpha=.7, lw=.6, capsize=2, zorder=2)
    ax.plot(lim, lim, "k--", lw=1.2, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_box_aspect(1)   # identical square proportions, like every parity panel
    ax.grid(True, which="both", ls=":", lw=.5, alpha=.5)
    ax.tick_params(labelsize=st["tick_label_size"])
    ax.set_xlabel(st["xlabel"], fontsize=st["axis_label_size"])
    ax.set_ylabel(st["ylabel"], fontsize=st["axis_label_size"])
    ax.set_title(title, fontsize=st["title_size"], pad=st["title_pad"])
    # Defer placement to after tight_layout (finalize_parity_panels, via
    # _savefig_multi). Both the note box and, if present, the orientation legend
    # are placed and tested there. markers_ms is None for the single-series
    # production plot -> box placed (tested vs data), no legend.
    ax._parity_placement = dict(
        show_legend=bool(has_legend),
        legend_kw=dict(fontsize=st["legend_size"], title="orientation",
                       title_fontsize=st["legend_title_size"], framealpha=.75,
                       edgecolor="0.6", borderpad=.8, labelspacing=.6,
                       **({"handles": _hd} if has_legend else {})),
        metrics_text=note, metrics_fontsize=st["metrics_size"],
        xt=xt, xp=xp, yerr=yerr,
        markers_ms=(markers_ms if markers_ms is not None else [("o", 5.0)] * len(xt)),
        capsize_pt=2)
    fig.tight_layout(); _savefig_multi(fig, path); plt.close(fig)

# ===========================================================================
EXPECTED_SPLITS = {"restricted-group": 5, "random": N_RANDOM, "variable": N_VARIABLE}

def _load_cv_predictions(model, s, out_dir, rg_dir=None):
    """Load a model's CV predictions for split s only if COMPLETE and VALID:
    all expected folds present AND per-fold row counts match the frozen split
    membership (random/variable) or the RG split files (restricted-group). This
    rejects legacy/stale outputs (e.g. a 210-row 10x21 ShuffleSplit file) that
    merely have the right number of fold labels. Returns the dataframe or None."""
    run_dir = os.path.join(out_dir, f"{model}_cv")
    path = os.path.join(run_dir, f"{model}_{s}_predictions.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "split" not in df.columns:
        return None
    n_folds = df["split"].nunique()
    if n_folds < EXPECTED_SPLITS[s]:
        print(f"  {model} {s}: only {n_folds}/{EXPECTED_SPLITS[s]} folds -> leaving blank")
        return None
    got_counts = {int(k): int(v) for k, v in df.groupby("split").size().to_dict().items()}
    exp_counts = None
    if s in ("random", "variable"):
        mem_path = os.path.join(out_dir, f"{s}_split_membership.csv")
        if os.path.exists(mem_path):
            mem = pd.read_csv(mem_path)
            exp_counts = {int(k): int(v) for k, v in mem.groupby("fold").size().to_dict().items()}
    elif s == "restricted-group" and rg_dir:
        try:
            rg = load_rg_dir(rg_dir)
            exp_counts = {k: len(te) for k, (_, te) in enumerate(rg)}
        except Exception:
            exp_counts = None
    if exp_counts is not None and exp_counts != got_counts:
        print(f"  {model} {s}: per-fold counts {got_counts} != expected {exp_counts} "
              f"(stale/legacy file) -> leaving blank")
        return None
    return df

def _fig_cols(args):
    """Which split columns to draw. --fig_splits takes a comma-separated list
    (e.g. 'restricted-group,random' to drop variable); default = all three, in
    the canonical rg, random, variable order."""
    raw = getattr(args, "fig_splits", None)
    if not raw:
        return list(STRATEGIES)
    want = [s.strip() for s in raw.split(",") if s.strip()]
    bad = [s for s in want if s not in STRATEGIES]
    if bad:
        sys.exit(f"--fig_splits: unknown split(s) {bad}; choose from {list(STRATEGIES)}")
    return [s for s in STRATEGIES if s in want]   # canonical order preserved

def mode_validation_fig(args):
    """Combined CROSS-VALIDATION figure: 2 rows x N cols. Top row = LLM, bottom
    row = GPR, columns = the selected splits (default rg, random, variable; use
    --fig_splits to include/exclude, e.g. --fig_splits restricted-group,random).

    This is the validation figure, NOT production: it shows held-out CV
    performance. Production is llm-prod / gpr-prod -- a single model fit on all
    201 rows, with its own parity plot.

    A cell is drawn only if that model+split is COMPLETE; otherwise it's left
    blank and the grid spacing is preserved (no dynamic refitting). Pure
    replot -- no inference/fitting. Clean (no a/b/c letters)."""
    out_dir = args.out_dir
    rows = [("llm", "Fine-Tuned GPT-4o"), ("gpr", "Gaussian Process Regression")]
    cols = _fig_cols(args)
    print(f"validation figure columns: {cols}")
    # SHARED PARITY CELL (see PARITY_PANEL_W_IN): same per-panel inches as
    # ablation-fig / gpr-ablation-fig, fonts width-compensated by
    # scaled_style(fig_w) so all figures match on the page at width=\textwidth.
    fig_w = PARITY_PANEL_W_IN * len(cols)
    st = scaled_style(fig_w)
    fig, axes = plt.subplots(2, len(cols), figsize=(fig_w, PARITY_PANEL_H_IN * 2),
                             squeeze=False)
    any_drawn = False
    # ONE legend of each KIND per figure, IN-PANEL, on the TOP row only: the
    # split legend on the RG column, the orientation legend on the FIRST
    # orientation-encoded column (random/variable share the encoding, so a
    # second orientation legend would just repeat it). The bottom (GPR) row
    # repeats both encodings and carries no legend.
    seen_orient = False
    for ri, (model, model_name) in enumerate(rows):
        for ci, s in enumerate(cols):
            ax = axes[ri][ci]
            df = _load_cv_predictions(model, s, out_dir, rg_dir=getattr(args, "rg_dir", None))
            if df is None:
                ax.axis("off")     # blank cell, spacing preserved
                continue
            title = f"{model_name} | {PLOT_STYLE['split_display'].get(s, s)}"
            show_leg = (ri == 0) and (s == "restricted-group" or not seen_orient)
            _draw_parity_panel(ax, fig, df, s, title, style=st, show_legend=show_leg,
                               show_ylabel=(ci == 0), show_xlabel=(ri == 1))
            if show_leg and s != "restricted-group":
                seen_orient = True
            any_drawn = True
    if not any_drawn:
        sys.exit("no complete CV predictions found for any model/split.")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    # distinct filename when a subset of splits is drawn, so the full 3-column
    # figure is never silently overwritten by a 2-column one
    stem = "validation_figure" if cols == list(STRATEGIES) else \
           "validation_figure_" + "_".join(s.replace("restricted-group", "rg") for s in cols)
    out = os.path.join(out_dir, f"{stem}.svg")
    _savefig_multi(fig, out)
    plt.close(fig)
    print(f"wrote {stem}.svg/.pdf/.eps to {out_dir}/ "
          f"(top row LLM, bottom row GPR; columns {cols}; blank cells = incomplete splits)")

def mode_plot_only(args):
    """Redraw parity plots from an already-saved predictions CSV -- no fitting,
    no inference, no API. Works for any *_predictions.csv written by gpr-cv,
    llm-cv, llm-report, gpr-cv, or ablation-fig -- any file with strategy/split/
    truth/pred/band columns. NOT the llm-prod / gpr-prod outputs: those carry a
    'sigma' column and no 'strategy'. Point --pred_csv at the file; the plot is
    regenerated in place next to it."""
    pred = getattr(args, "pred_csv", None)
    if not pred or not os.path.exists(pred):
        sys.exit("plot-only requires --pred_csv PATH to a saved *_predictions.csv")
    df = pd.read_csv(pred)
    run_dir = os.path.dirname(os.path.abspath(pred))
    base = os.path.basename(pred)
    tag = base[:-len("_predictions.csv")] if base.endswith("_predictions.csv") else base.split(".")[0]
    strat_present = [s for s in STRATEGIES if s in set(df["strategy"])]
    plot_from_predictions(df, run_dir, tag, only=strat_present)
    print(f"redrew {tag} parity plot(s) for {strat_present} in {run_dir}/ (no recompute)")

def mode_data_fig(args):
    """Dataset-description figure: (a) per-chemistry sample count across the
    full chemistry list -- only chemistries occurring more than once are
    labelled, the singleton tail is annotated rather than labelled; (b) AO
    fluence and (c) erosion yield, both stacked by orientation. Pure
    description of --data_csv: no models, no inference, no fitting."""
    from matplotlib.gridspec import GridSpec
    # Match the parity figures' rendered font EXACTLY: overlay the width-scaled
    # canonical sizes onto data-fig's own dict, so its text lands at the same
    # on-page pt as every other figure (scaled_style cancels the figure width,
    # so 12in here renders the same as the parity figures at their widths). The
    # Computer Modern family is already global via plt.rcParams, so data-fig
    # inherits the correct FONT too -- this only aligns the SIZES. Non-font
    # DATA_FIG keys (colours, bins, figsize) are untouched.
    _dfw = DATA_FIG["figsize"][0]
    _sc = scaled_style(_dfw)
    D = dict(DATA_FIG)
    D["axis_label_size"] = _sc["axis_label_size"]
    D["tick_label_size"] = _sc["tick_label_size"]
    D["legend_size"]     = _sc["legend_size"]
    D["annot_size"]      = _sc["metrics_size"]
    # bar_label_size (rotated PSMILES tick labels) scales too, kept proportionally
    # smaller than the axis labels as before
    D["bar_label_size"]  = _sc["tick_label_size"] * (DATA_FIG["bar_label_size"] /
                                                     DATA_FIG["tick_label_size"])
    pool = load_master_csv(args.data_csv)
    orient = pool["orientation"].astype(str).str.strip().str.lower()
    ey = pd.to_numeric(pool["e_y (A3/atom)"], errors="coerce")
    flu = pd.to_numeric(pool["fluence"], errors="coerce")
    print(f"data-fig: {len(pool)} rows from {args.data_csv}")

    # ---- (a) per-chemistry counts -------------------------------------------
    counts = pool["smiles"].dropna().value_counts().sort_values(ascending=False)
    multi = counts[counts > 1]
    need = counts.index if D["label_all"] else multi.index
    # PSMILES_NAMES is a nicety, not a requirement: the PSMILES string is itself
    # the unique chemistry key, so any chemistry without a dict entry is labelled
    # by its PSMILES. No reason to halt the whole figure over a missing label.
    unnamed = [ps for ps in need if ps not in PSMILES_NAMES]
    if unnamed:
        print(f"data-fig: {len(unnamed)} chemistr"
              f"{'y' if len(unnamed)==1 else 'ies'} without a PSMILES_NAMES entry; "
              f"labelling by PSMILES.")
    n_single = int((counts == 1).sum())
    print(f"data-fig: {len(counts)} chemistries, "
          f"{len(need)} labelled" + ("" if D["label_all"] else
          f" (n>1), {n_single} singletons annotated"))

    fig = plt.figure(figsize=D["figsize"])
    gs = GridSpec(2, 2, figure=fig, height_ratios=D["height_ratios"],
                  hspace=0.5, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    x = np.arange(len(counts))
    ax_a.bar(x, counts.values, color=D["singleton_color"],
             edgecolor="white", linewidth=0.5)
    n_lab = len(counts) if D["label_all"] else len(multi)
    ax_a.set_xticks(x[:n_lab])
    ax_a.set_xticklabels([PSMILES_NAMES.get(ps, ps) for ps in counts.index[:n_lab]],
                         rotation=45, ha="right", fontsize=D["bar_label_size"])
    ax_a.set_xlim(-0.8, len(counts) - 0.2)
    ax_a.set_ylabel("PSMILES Count", fontsize=D["axis_label_size"])
    ax_a.tick_params(axis="y", labelsize=D["tick_label_size"])
    if n_single and not D["label_all"]:
        x0, x1 = len(multi) - 0.5, len(counts) - 0.5
        y = max(counts.values) * 0.16
        ax_a.plot([x0, x1], [y, y], color="0.35", lw=1.0, clip_on=False)
        for xe in (x0, x1):
            ax_a.plot([xe, xe], [y, y - max(counts.values) * 0.03],
                      color="0.35", lw=1.0, clip_on=False)
        ax_a.text((x0 + x1) / 2, y * 1.12,
                  f"{n_single} chemistries with a single sample",
                  ha="center", va="bottom", fontsize=D["annot_size"], color="0.2")

    # ---- (b) AO fluence, stacked by orientation ------------------------------
    fb = np.logspace(np.floor(np.log10(flu.min())), np.ceil(np.log10(flu.max())),
                     D["fluence_bins"] + 1)
    # single hist() call with a list -> genuinely stacked. Separate per-series
    # calls do NOT stack (each draws from baseline and overpaints the last).
    ax_b.hist([flu[orient == o].dropna().values for o in D["orient_order"]],
              bins=fb, color=[D["orient_colors"][o] for o in D["orient_order"]],
              label=[o.capitalize() for o in D["orient_order"]],
              edgecolor="white", linewidth=0.5, stacked=True)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("AO Fluence (atoms/cm$^2$)", fontsize=D["axis_label_size"])
    ax_b.set_ylabel("Count", fontsize=D["axis_label_size"])
    ax_b.tick_params(axis="both", labelsize=D["tick_label_size"])
    ax_b.legend(title="Orientation", fontsize=D["legend_size"],
                title_fontsize=D["legend_size"])

    # ---- (c) erosion yield, stacked by orientation ---------------------------
    eb = np.logspace(np.floor(np.log10(ey.min())), np.ceil(np.log10(ey.max())),
                     D["ey_bins"] + 1)
    ax_c.hist([ey[orient == o].dropna().values for o in D["orient_order"]],
              bins=eb, color=[D["orient_colors"][o] for o in D["orient_order"]],
              label=[o.capitalize() for o in D["orient_order"]],
              edgecolor="white", linewidth=0.5, stacked=True)
    ax_c.set_xscale("log")
    # mathtext \AA, never a literal U+00C5: cmr10 has no glyph for it and would
    # silently substitute a tofu box.
    ax_c.set_xlabel(r"Erosion Yield ($\mathrm{\AA^3/atom}$)",
                    fontsize=D["axis_label_size"])
    ax_c.set_ylabel("Count", fontsize=D["axis_label_size"])
    ax_c.tick_params(axis="both", labelsize=D["tick_label_size"])
    ax_c.legend(title="Orientation", fontsize=D["legend_size"],
                title_fontsize=D["legend_size"])

    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "dataset_figure.svg")
    _savefig_multi(fig, out)
    plt.close(fig)
    print(f"wrote dataset_figure.svg/.pdf/.eps to {args.out_dir}/")


def mode_rebuild_bands(args):
    """Recompute the cached per-row error band as one sigma (sd, ddof=1) from the
    saved raw/rep*.csv files, for every unit under llm_cv and llm_ablation.

    Needed once, because the row store was populated when the band was the
    half-range (max-min)/2 rather than the sd. Every individual call was saved to
    raw/repN.csv, so the sd is recoverable exactly: NO API calls, no cost. Only
    the band changes; predictions (means) and therefore OME/log-R2 are untouched.

    Matching does NOT go through the dataset CSV. A raw row is matched to its
    store entry by (model, truth, mean-of-its-repeats): the store's cached `pred`
    IS the mean of exactly those repeats, so this is an internal, self-consistent
    identity. Regenerating prompts from the CSV instead would break whenever the
    CSV changed after inference (e.g. corrected SMILES, or RG split files holding
    the pre-correction strings).

    Raw files are walked BY ROW POSITION -- infer_repeats writes every repN.csv
    with the same rows in the same order, so position i across rep1..repN is one
    row's repeat set. Aggregating instead would pool byte-identical rows into a
    single 2N-sample sd.

    Nothing here blocks. Rows that rebuild are updated; rows that cannot (no raw
    repeats) keep their existing band and are reported. Prediction CSVs are
    all-or-nothing PER FILE -- fully patched or left completely untouched -- so no
    plotted file ever mixes sd with half-range, while a stale or unused file
    cannot stop the rest being fixed. A file whose (truth, pred) is shared with a
    row that could NOT be rebuilt is left untouched too: the CSVs carry no model
    column, so the two are indistinguishable."""
    plan, skipped = [], []
    n_units = n_rows = 0
    for sub_dir in ("llm_cv", "llm_ablation"):
        run_dir = os.path.join(args.out_dir, sub_dir)
        if not os.path.isdir(run_dir):
            continue
        m = _load_manifest(run_dir)
        store = _load_rowstore(run_dir)
        if not store:
            print(f"{sub_dir}: no row store; nothing to rebuild."); continue
        # index the store by model for matching
        by_model = {}
        for k, r in store.items():
            by_model.setdefault(str(r.get("model")), []).append(k)
        updated = {}
        for udir in sorted(glob.glob(os.path.join(run_dir, "*__split_*"))
                           + glob.glob(os.path.join(run_dir, "prod"))):
            u = os.path.basename(udir)
            model = m["units"].get(u, {}).get("fine_tuned_model")
            reps = sorted(glob.glob(os.path.join(udir, "raw", "rep*.csv")),
                          key=lambda p: int("".join(ch for ch in os.path.basename(p)
                                                    if ch.isdigit()) or 0))
            if not model or not reps:
                continue
            frames = [pd.read_csv(f) for f in reps]
            n = len(frames[0])
            if any(len(d) != n for d in frames):
                skipped.append(f"{sub_dir}/{u}: repeat files disagree on row count "
                               f"{[len(d) for d in frames]}; unit skipped")
                continue
            cand = by_model.get(str(model), [])
            hit = 0
            for i in range(n):                      # one row = position i across reps
                vals = []
                for d in frames:
                    try:
                        vals.append(float(str(d["pred"].iloc[i]).strip()))
                    except (ValueError, TypeError):
                        vals.append(np.nan)
                a = np.array(vals, float)
                nv = int(np.sum(~np.isnan(a)))
                if nv == 0:
                    continue                        # every repeat failed to parse
                mu = float(np.nanmean(a))
                try:
                    t_i = float(str(frames[0]["truth"].iloc[i]).strip())
                except (ValueError, TypeError):
                    skipped.append(f"{sub_dir}/{u}: raw row {i} has an unparseable truth; row skipped")
                    continue
                # The row store keeps truth at full precision (target(te)), but
                # row_to_record rounds to 3dp before writing it into raw/repN.csv,
                # so compare at the raws' precision. `pred` (the cached mean) is
                # the discriminating term.
                match = [k for k in cand
                         if np.isclose(round(float(store[k]["truth"]), 3), t_i,
                                       rtol=0, atol=1e-12)
                         and np.isclose(float(store[k]["pred"]), mu,
                                        rtol=1e-9, atol=1e-9)]
                if not match:
                    skipped.append(f"{sub_dir}/{u}: raw row {i} (truth {t_i}, mean "
                                   f"{mu:.6f}) matches no row-store entry; row skipped")
                    continue
                k = next((x for x in match if x not in updated), match[0])
                rec = dict(store[k])
                rec["band"] = float(np.nanstd(a, ddof=1)) if nv > 1 else 0.0
                updated[k] = rec; hit += 1          # duplicate identity -> last wins,
                                                    # matching _append_rowstore
            if hit:
                n_units += 1; n_rows += hit
                print(f"  {sub_dir}/{u}: rebuilt sd for {hit} rows from {len(reps)} repeat files")
        # Coverage report. A cached row whose unit has no raw repeats cannot be
        # rebuilt; it keeps its existing band and is reported below. This does
        # NOT block -- an unrebuildable row must not stop the rest being fixed.
        stale = [k for k in store if k not in updated]
        if stale:
            models = sorted({str(store[k].get("model")) for k in stale})
            skipped.append(f"{sub_dir}: {len(stale)} cached row(s) had no matching raw "
                           f"repeats; their band is left as-is. Model(s): "
                           f"{', '.join(x[-18:] for x in models[:4])}"
                           + (" ..." if len(models) > 4 else ""))
        if updated:
            plan.append((run_dir, updated))
    # ---- plan the prediction-CSV patches (still WITHOUT writing anything) ----
    # These are the files plot-only / validation-fig read. Patching them here means the
    # plots can be redrawn without llm-report, which would rebuild folds from
    # --data_csv and re-infer any row whose SMILES changed since training.
    #
    # row_predictions.csv is deliberately EXCLUDED from patching: it is written
    # below from `updated`, keyed exactly and per model. Re-patching it through
    # the (truth, pred) map would be redundant, and could overwrite a correct
    # model-specific band if two models ever shared a (truth, pred).
    csv_plan = []
    for run_dir, updated in plan:
        store = _load_rowstore(run_dir)
        band_of, ambiguous = {}, set()
        for r in updated.values():
            k = (round(float(r["truth"]), 9), round(float(r["pred"]), 9))
            if k in band_of and not np.isclose(band_of[k], float(r["band"]),
                                               rtol=0, atol=1e-12):
                ambiguous.add(k)
            band_of[k] = float(r["band"])
        # A (truth, pred) that ALSO belongs to a store row we could not rebuild is
        # unsafe: the prediction CSVs carry no model column, so such a row is
        # indistinguishable from the rebuilt one and would silently receive the
        # wrong band. Any file containing one is left untouched.
        unsafe = {(round(float(r["truth"]), 9), round(float(r["pred"]), 9))
                  for k, r in store.items() if k not in updated}
        if ambiguous:
            skipped.append(f"{os.path.basename(run_dir)}: {len(ambiguous)} (truth, pred) "
                           f"pair(s) map to more than one band; prediction CSVs in this "
                           f"directory left UNTOUCHED rather than risk a wrong band")
            continue
        store_name = os.path.basename(_rowstore_path(run_dir))
        targets = [f for f in
                   (glob.glob(os.path.join(run_dir, "*_predictions.csv"))
                    + glob.glob(os.path.join(run_dir, "*__split_*", "predictions.csv"))
                    + glob.glob(os.path.join(run_dir, "prod", "predictions.csv")))
                   if os.path.basename(f) != store_name]
        for f in sorted(set(targets)):
            try:
                d = pd.read_csv(f)
            except Exception as e:
                skipped.append(f"{os.path.relpath(f, args.out_dir)}: unreadable ({e})")
                continue
            if not {"truth", "pred", "band"} <= set(d.columns):
                continue
            new_band, miss, risky = [], 0, 0
            for t, pr, b in zip(d["truth"], d["pred"], d["band"]):
                k = (round(float(t), 9), round(float(pr), 9))
                if k in unsafe:
                    risky += 1; new_band.append(b)
                elif k in band_of:
                    new_band.append(band_of[k])
                else:
                    new_band.append(b); miss += 1
            if risky:
                skipped.append(f"{os.path.relpath(f, args.out_dir)}: {risky} row(s) share a "
                               f"(truth, pred) with a row that could NOT be rebuilt -- "
                               f"indistinguishable, so this file is left UNTOUCHED")
                continue
            if miss:
                # All-or-nothing per file: leave it completely untouched rather
                # than half-patch it, so no file ever MIXES sd and half-range.
                # This is only a warning, not a failure -- a stale/unused file
                # (e.g. an old partial aggregate) must not block rebuilding the
                # files that are actually in use.
                skipped.append(f"{os.path.relpath(f, args.out_dir)}: {miss} of {len(d)} "
                               f"rows not in the rebuild -- left UNTOUCHED (stale file?)")
                continue
            d["band"] = new_band
            csv_plan.append((f, d))
    # Write what rebuilt. Nothing here blocks: anything that could not be rebuilt
    # is reported and left exactly as it was. Prediction CSVs are all-or-nothing
    # PER FILE, so no plotted file ever mixes sd with half-range.
    for run_dir, updated in plan:
        _append_rowstore(run_dir, list(updated.values()))
        print(f"{os.path.basename(run_dir)}: row store updated ({len(updated)} rows)")
    for f, d in csv_plan:
        d.to_csv(f, index=False)
        print(f"  patched {os.path.relpath(f, args.out_dir)}")
    for w in skipped:
        print(f"  SKIPPED {w}")
    print(f"\nrebuilt bands for {n_rows} row(s) across {n_units} unit(s); "
          f"{len(csv_plan)} prediction CSV(s) patched"
          + (f"; {len(skipped)} item(s) skipped and left unchanged (above)"
             if skipped else "; nothing skipped")
          + ".\nNo API calls made. Redraw with plot-only / validation-fig. Do NOT use "
            "llm-report if --data_csv has changed since the models were trained -- "
            "it would re-infer the changed rows.")

def mode_ablation_fig(args):
    """Three parity panels -- baseline / +layers / +thickness -- for the LLM
    ablation, drawn exactly like the restricted-group panel (circles coloured by
    Split 1..5, dashed 1:1, log-log, metrics box). It calls the same
    _draw_parity_panel as validation-fig, so the styling is the identical code path,
    not a copy.

    The per-point data is reconstructed from runs/llm_ablation/<cond>__split_XX/
    raw/rep*.csv, which already hold every individual call (question, prediction,
    truth) from the ablation run. truth = the raw's truth column, pred = mean of
    the repeats, band = sd (ddof=1) across them -- the same 1-sigma convention as
    everywhere else. NO API calls; nothing is inferred or retrained."""
    run_dir = os.path.join(args.out_dir, "llm_ablation")
    if not os.path.isdir(run_dir):
        sys.exit(f"no {run_dir}; run llm-ablation first")
    conds = [("baseline", "Baseline"), ("layers", "+ Layers"), ("thickness", "+ Thickness")]
    panels, missing = {}, []
    for cond, _ in conds:
        pts = []
        for udir in sorted(glob.glob(os.path.join(run_dir, f"{cond}__split_*"))):
            k = int(os.path.basename(udir).split("_")[-1])
            reps = sorted(glob.glob(os.path.join(udir, "raw", "rep*.csv")),
                          key=lambda p: int("".join(ch for ch in os.path.basename(p)
                                                    if ch.isdigit()) or 0))
            if not reps:
                continue
            frames = [pd.read_csv(f) for f in reps]
            n = len(frames[0])
            if any(len(d) != n for d in frames):
                print(f"  {cond}__split_{k:02d}: repeat files disagree on row count; skipped")
                continue
            for i in range(n):
                vals = []
                for d in frames:
                    try:
                        vals.append(float(str(d["pred"].iloc[i]).strip()))
                    except (ValueError, TypeError):
                        vals.append(np.nan)
                a = np.array(vals, float)
                nv = int(np.sum(~np.isnan(a)))
                if nv == 0:
                    continue
                try:
                    t_i = float(str(frames[0]["truth"].iloc[i]).strip())
                except (ValueError, TypeError):
                    continue
                pts.append(dict(strategy="restricted-group", split=k, truth=t_i,
                                pred=float(np.nanmean(a)),
                                band=float(np.nanstd(a, ddof=1)) if nv > 1 else 0.0,
                                orientation="", fluence=np.nan))
        if pts:
            panels[cond] = pd.DataFrame(pts)
        else:
            missing.append(cond)
    if not panels:
        sys.exit(f"no ablation raw repeats found under {run_dir}; nothing to draw")
    if missing:
        print(f"  no raw repeats for: {', '.join(missing)} -- those panels left blank")

    # SHARED PARITY CELL (see PARITY_PANEL_W_IN): same per-panel inches as
    # validation-fig / gpr-ablation-fig, fonts width-compensated by
    # scaled_style(fig_w) so every text element renders at the same size as
    # every other figure when all are included at width=\textwidth.
    wr = [PARITY_AXES_IN + PARITY_YLAB_IN] + [PARITY_AXES_IN + PARITY_TICK_IN] * 2
    fig_w = sum(wr)
    st = scaled_style(fig_w)
    fig_h = (PARITY_AXES_IN + PARITY_TITLE_IN + PARITY_XLAB_IN
             + PARITY_SUPTITLE_IN + PARITY_LABEL_PAD_IN)
    # The Split 1..5 entries are identical across all three panels, so the legend
    # is drawn ONCE, inside the first drawn panel (Q8) -- no figure-level legend
    # and no reserved right margin, so the panels get the full width.
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), squeeze=False,
                             gridspec_kw={"width_ratios": wr})
    first = True
    for ci, (cond, nice) in enumerate(conds):
        ax = axes[0][ci]
        sub = panels.get(cond)
        if sub is None:
            ax.axis("off"); continue
        ome, r2 = _draw_parity_panel(
            ax, fig, sub, "restricted-group", nice, show_legend=first, style=st,
            show_ylabel=(ci == 0), show_xlabel=True)
        first = False
        print(f"  {cond:9s}: OME={ome:.3f} logR2={r2:+.3f} "
              f"({sub['split'].nunique()} folds, n={len(sub)})")
    # one model title for the whole figure rather than repeating it per panel.
    # The suptitle lives in its own reserved PARITY_SUPTITLE_IN strip:
    # tight_layout keeps the panels below rect_top, and the suptitle hangs from
    # the top edge, so the two can never collide.
    # Pin every square axes to the TOP of its slot: set_box_aspect shrinks the
    # axes inside the slot tight_layout allocated, and the default center
    # anchor dumps half that slack ABOVE the panel -- i.e. as a dead gap under
    # the suptitle. Anchoring north drops the slack below the panels instead.
    for _ax in axes.flat:
        _ax.set_anchor("N")
    fig.suptitle(st["model_display"]["llm"],
                 fontsize=st["suptitle_size"], y=0.99, va="top")
    fig_h = _equalize_parity_axes(fig, axes, PARITY_SUPTITLE_IN,
                                  extra_bot_in=0.15)
    # Snap the suptitle to just above the tallest panel title: the reserved
    # strip only guarantees ROOM, while tight_layout's internal headroom would
    # otherwise leave a dead gap between suptitle and titles.
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _tops = [ax.title.get_window_extent(_r).y1 for ax in fig.axes
             if ax.get_visible() and ax.get_title()]
    if _tops:
        fig._suptitle.set_va("bottom")
        fig._suptitle.set_y(min(max(_tops) / (fig.dpi * fig_h) + 0.10 / fig_h,
                                1.0 - 0.50 / fig_h))
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "ablation_figure.svg")
    _savefig_multi(fig, out)
    plt.close(fig)
    pd.concat([d.assign(condition=c) for c, d in panels.items()], ignore_index=True).to_csv(
        os.path.join(run_dir, "ablation_predictions.csv"), index=False)
    print(f"wrote ablation_figure.svg/.pdf/.eps to {args.out_dir}/ and "
          f"ablation_predictions.csv to {run_dir}/")

def _load_epoch_arm(run_dir, ep, rg):
    """Concatenate one epoch arm's five per-fold prediction files into a single
    frame with a `split` column. Returns (df, problems)."""
    probs, parts = [], []
    for k in range(len(rg)):
        f = os.path.join(run_dir, f"ep{ep}__split_{k:02d}", "predictions.csv")
        if not os.path.exists(f):
            probs.append(f"{ep} epochs, fold {k}: missing {os.path.relpath(f)}")
            continue
        try:
            d = pd.read_csv(f)
        except Exception as e:
            probs.append(f"{ep} epochs, fold {k}: unreadable ({e})"); continue
        need = {"truth", "pred", "band"}
        if not need <= set(d.columns):
            probs.append(f"{ep} epochs, fold {k}: missing column(s) "
                         f"{sorted(need - set(d.columns))}"); continue
        exp = len(rg[k][1])
        if len(d) != exp:
            probs.append(f"{ep} epochs, fold {k}: {len(d)} rows, expected {exp} "
                         f"(from rg/)"); continue
        for c in ("truth", "pred", "band"):
            if not np.all(np.isfinite(pd.to_numeric(d[c], errors="coerce").to_numpy())):
                probs.append(f"{ep} epochs, fold {k}: non-finite {c}"); break
        else:
            d = d.copy(); d["split"] = k; d["strategy"] = "restricted-group"
            parts.append(d)
    if probs:
        return None, probs
    return pd.concat(parts, ignore_index=True), []

def mode_tuning_fig(args):
    """2 x 2 tuning figure, reporting only -- no client, no API calls, no fitting.

    Top row: parity for each epoch arm, drawn by the SAME _draw_parity_panel as
    every other figure (s="restricted-group"), sharing one axis range so the arms
    cannot look different merely because their limits differ. Bottom row: the
    temperature sweep, log-R2 and OME.

    Sources, all already on disk:
      runs/llm_epoch_tune/ep{E}__split_XX/predictions.csv   (per-fold, per arm)
      runs/epoch_tune_results.csv                           (arm metrics)
      runs/temp_tune_results.csv                            (sweep)

    It will NOT fall back to the llm_cv predictions if the epoch-tune artifacts
    are absent: both panels must come from the same tuning run at the same
    temperature, or the comparison is not what it claims to be."""
    rg = load_rg_dir(args.rg_dir)
    et_dir = os.path.join(args.out_dir, "llm_epoch_tune")
    tt_csv = os.path.join(args.out_dir, "temp_tune_results.csv")
    et_csv = os.path.join(args.out_dir, "epoch_tune_results.csv")

    problems = []
    if not os.path.exists(tt_csv):
        problems.append(f"missing {os.path.relpath(tt_csv)} -- run: python pipeline.py temp-tune")
    if not os.path.exists(et_csv):
        problems.append(f"missing {os.path.relpath(et_csv)} -- run: "
                        f"python pipeline.py epoch-tune")
    if problems:
        sys.exit("tuning-fig needs completed tuning runs; it will not substitute the "
                 f"defaults (temperature {DEFAULT_TEMPERATURE}, {DEFAULT_EPOCHS} epochs) "
                 "or fall back to the llm_cv predictions:\n"
                 + "\n".join("  " + p for p in problems))

    tt = pd.read_csv(tt_csv)
    et = pd.read_csv(et_csv)
    if sorted(tt.temperature.round(6)) != sorted(round(float(t), 6) for t in TEMP_GRID):
        problems.append(f"temp_tune_results.csv has temperatures {sorted(tt.temperature)}, "
                        f"expected TEMP_GRID {TEMP_GRID}")
    n_expected = sum(len(te) for _, te in rg)
    if "n_rows" in tt and not (tt.n_rows == n_expected).all():
        problems.append(f"temp_tune_results.csv: not every temperature covers "
                        f"{n_expected} rows -> {dict(zip(tt.temperature, tt.n_rows))}")
    if sorted(et.epochs) != sorted(EPOCH_GRID):
        problems.append(f"epoch_tune_results.csv has epochs {sorted(et.epochs)}, "
                        f"expected EPOCH_GRID {EPOCH_GRID}")

    arms = {}
    for ep in EPOCH_GRID:
        d, probs = _load_epoch_arm(et_dir, ep, rg)
        problems += probs
        if d is not None:
            arms[ep] = d
            # the panels recompute their own metrics; if those disagree with the
            # saved summary the files are stale or mismatched
            row = et[et.epochs == ep]
            if len(row):
                ome, r2 = metrics(d.truth.to_numpy(float), d.pred.to_numpy(float))
                if not np.isclose(ome, float(row.OME.iloc[0]), atol=1e-6):
                    problems.append(f"{ep} epochs: OME from the per-fold files ({ome:.6f}) "
                                    f"disagrees with epoch_tune_results.csv "
                                    f"({float(row.OME.iloc[0]):.6f}) -- stale files?")
    if problems:
        sys.exit("tuning-fig: refusing to draw a figure from incomplete or inconsistent "
                 "inputs:\n" + "\n".join("  " + p for p in dict.fromkeys(problems)))

    # one axis range across both parity panels
    allv = np.concatenate([10 ** np.concatenate([d.truth.to_numpy(float),
                                                 d.pred.to_numpy(float)])
                           for d in arms.values()])
    allv = allv[np.isfinite(allv) & (allv > 0)]
    lim = [10 ** np.floor(np.log10(allv.min())), 10 ** np.ceil(np.log10(allv.max()))]
    print(f"tuning-fig: shared parity range {lim[0]:.0e} .. {lim[1]:.0e}")

    from matplotlib.gridspec import GridSpec
    T = TUNING_FIG
    fig = plt.figure(figsize=T["figsize"])
    st = scaled_style(T["figsize"][0])
    gs = GridSpec(2, 2, figure=fig, height_ratios=T["height_ratios"],
                  hspace=0.32, wspace=0.28)
    # Q8: the Split 1..5 legend is identical across both epoch panels, so it is
    # drawn once, inside the first one.
    for ci, ep in enumerate(EPOCH_GRID):
        ax = fig.add_subplot(gs[0, ci])
        ome, r2 = _draw_parity_panel(ax, fig, arms[ep], "restricted-group",
                                     f"{ep} Epochs", show_legend=(ci == 0), style=st,
                                     show_ylabel=(ci == 0), show_xlabel=True,
                                     lim_override=lim)
        print(f"  {ep:>2} epochs: OME={ome:.3f} logR2={r2:+.3f} n={len(arms[ep])}")

    tt = tt.sort_values("temperature")
    for ci, (col, color, marker, ylab) in enumerate(
            [("logR2", T["logr2_color"], T["logr2_marker"], "Log R$^2$"),
             ("OME",   T["ome_color"],   T["ome_marker"],   "OME")]):
        ax = fig.add_subplot(gs[1, ci])
        # No error bars, deliberately. temp_tune_results.csv now carries real
        # bootstrap CIs (OME_ci_lo/hi, logR2_ci_lo/hi), so bars COULD be drawn
        # here -- the reason they are not is a presentation choice, not a missing
        # number. (The old reason was that the only available quantity was
        # OME_std_rows, the dispersion of individual row errors, which is not an
        # uncertainty of the plotted value and must never be drawn as one.) To
        # add them: yerr from the CI columns as a 2xN array, since the intervals
        # are asymmetric.
        ax.plot(tt.temperature, tt[col], color=color, marker=marker,
                lw=T["line_width"], ms=T["marker_size"], zorder=3)
        ax.set_xlabel("Temperature", fontsize=st["axis_label_size"])
        ax.set_ylabel(ylab, fontsize=st["axis_label_size"])
        ax.set_xticks(list(tt.temperature))
        ax.tick_params(axis="both", labelsize=st["tick_label_size"])
        ax.grid(True, which="both", ls=":", lw=.6, alpha=.7)

    fig.suptitle(st["model_display"]["llm"],
                 fontsize=st["suptitle_size"], y=st["suptitle_y"])
    fig.tight_layout(rect=[0, 0, 1, st["suptitle_rect_top"]])
    os.makedirs(args.out_dir, exist_ok=True)
    _savefig_multi(fig, os.path.join(args.out_dir, "tuning_figure.svg"))
    plt.close(fig)
    best_t = tt.sort_values(["OME", "logR2"], ascending=[True, False]).temperature.iloc[0]
    best_e = et.sort_values(["OME", "logR2"], ascending=[True, False]).epochs.iloc[0]
    print(f"  selected: temperature {best_t}, {int(best_e)} epochs")
    print(f"wrote tuning_figure.svg/.pdf/.eps to {args.out_dir}/")

def mode_gpr_ablation_fig(args):
    """GPR ablation figure: rows = split methods, columns = baseline / +layers /
    +thickness. Mirrors validation-fig's structure (which is rows = models,
    columns = splits) and uses the SAME _draw_parity_panel, so every panel is the
    identical code path -- RG rows get split colours, random/variable rows get the
    orientation-marker + fluence-colourbar encoding, exactly as in the CV figures.

    A separate figure from the LLM ablation-fig, which stays a single
    restricted-group row because llm-ablation is RG-only by design.

    Reads runs/gpr_ablation/gpr_ablation_<split>_predictions.csv, written by
    gpr-ablation. No refitting, no API calls. Rows are whichever splits have a
    predictions file; --fig_splits selects a subset in canonical order.

    band = the GP predictive sd -- the same 1-sigma convention as the LLM figure,
    but NOT the same quantity (posterior predictive sd vs scatter across
    stochastic decodes). Say so in the caption."""
    run_dir = os.path.join(args.out_dir, "gpr_ablation")
    want = _fig_cols(args)
    rows = []
    for s in want:
        f = os.path.join(run_dir, f"gpr_ablation_{s}_predictions.csv")
        if os.path.exists(f):
            rows.append((s, pd.read_csv(f)))
        else:
            print(f"  {s}: no {os.path.basename(f)} -- run "
                  f"'python pipeline.py gpr-ablation --split {s}'; row omitted")
    if not rows:
        sys.exit(f"no gpr_ablation_*_predictions.csv under {run_dir} -- run: "
                 f"python pipeline.py gpr-ablation")

    conds = [("baseline", "Baseline"), ("layers", "+ Layers"), ("thickness", "+ Thickness")]
    # every condition within a row must cover the same rows, or the comparison
    # the figure is making does not hold
    for s, d in rows:
        n = {c: int((d.condition == c).sum()) for c, _ in conds if (d.condition == c).any()}
        if len(set(n.values())) != 1:
            sys.exit(f"{s}: conditions cover different row counts {n} -- they must be "
                     f"the same matched-control rows for the comparison to mean anything")

    nr = len(rows)
    # SHARED PARITY CELL (see PARITY_PANEL_W_IN): same per-panel inches as
    # validation-fig / ablation-fig, fonts width-compensated by
    # scaled_style(fig_w) -- see ablation-fig.
    wr = [PARITY_AXES_IN + PARITY_YLAB_IN] + [PARITY_AXES_IN + PARITY_TICK_IN] * 2
    fig_w = sum(wr)
    st = scaled_style(fig_w)
    row_t = PARITY_AXES_IN + PARITY_TITLE_IN
    row_b = PARITY_AXES_IN + PARITY_TITLE_IN + PARITY_XLAB_IN
    fig_h = row_t * (nr - 1) + row_b + PARITY_SUPTITLE_IN + PARITY_LABEL_PAD_IN
    fig, axes = plt.subplots(nr, 3, figsize=(fig_w, fig_h), squeeze=False,
                             gridspec_kw={"height_ratios": [row_t] * (nr - 1) + [row_b],
                                          "width_ratios": wr})
    # ONE legend of each KIND per figure, IN-PANEL: the split legend inside the
    # first panel of the (single) RG row, the orientation legend inside the
    # first panel of the FIRST orientation-encoded row only -- a second
    # orientation row would just repeat it. The fluence colourbar (a
    # quantitative scale, not a legend) stays on each orientation row's last
    # column, at the figure's right edge, exactly like the validation figure.
    seen_orient = False
    for ri, (s, d) in enumerate(rows):
        need_leg = (s == "restricted-group") or not seen_orient
        for ci, (cond, nice) in enumerate(conds):
            ax = axes[ri][ci]
            sub = d[d.condition == cond]
            if sub.empty:
                ax.axis("off"); continue
            ome, r2 = _draw_parity_panel(
                ax, fig, sub, s,
                # single line, COMPACT split name: the full display name
                # cannot fit one line -- its rendered width grows faster with
                # figure width than the column pitch does (0.34*fig_w vs
                # fig_w/3), so no figure width fixes it. "RG Split" etc. does.
                f"{PARITY_SPLIT_SHORT.get(s, s)} | {nice}", style=st,
                show_legend=need_leg, show_colorbar=(ci == 2),
                show_ylabel=(ci == 0), show_xlabel=(ri == nr - 1))
            if need_leg:
                need_leg = False
                if s != "restricted-group":
                    seen_orient = True
            print(f"  {s:16s} {cond:9s}: OME={ome:.3f} logR2={r2:+.3f} "
                  f"({sub['split'].nunique()} folds, n={len(sub)})")
    # Pin every square axes to the TOP of its slot: set_box_aspect shrinks the
    # axes inside the slot tight_layout allocated, and the default center
    # anchor dumps half that slack ABOVE the panel -- i.e. as a dead gap under
    # the suptitle. Anchoring north drops the slack below the panels instead.
    for _ax in axes.flat:
        _ax.set_anchor("N")
    # suptitle in its own reserved strip -- see ablation-fig
    fig.suptitle(st["model_display"]["gpr"],
                 fontsize=st["suptitle_size"], y=0.99, va="top")
    fig_h = _equalize_parity_axes(fig, axes, PARITY_SUPTITLE_IN,
                                  extra_bot_in=0.15)
    # Snap the suptitle to just above the tallest panel title: the reserved
    # strip only guarantees ROOM, while tight_layout's internal headroom would
    # otherwise leave a dead gap between suptitle and titles.
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _tops = [ax.title.get_window_extent(_r).y1 for ax in fig.axes
             if ax.get_visible() and ax.get_title()]
    if _tops:
        fig._suptitle.set_va("bottom")
        fig._suptitle.set_y(min(max(_tops) / (fig.dpi * fig_h) + 0.10 / fig_h,
                                1.0 - 0.50 / fig_h))
    os.makedirs(args.out_dir, exist_ok=True)
    stem = ("gpr_ablation_figure" if [s for s, _ in rows] == list(STRATEGIES)
            else "gpr_ablation_figure_" + "_".join(s.replace("restricted-group", "rg")
                                                   for s, _ in rows))
    _savefig_multi(fig, os.path.join(args.out_dir, f"{stem}.svg"))
    plt.close(fig)
    print(f"wrote {stem}.svg/.pdf/.eps to {args.out_dir}/ "
          f"(rows: {[s for s, _ in rows]}; columns: baseline / +layers / +thickness)")

MODES = {
    "temp-tune":    mode_llm_temp_tune,
    "epoch-tune":   mode_llm_epoch_tune,
    "llm-cv":       mode_llm_cv,
    "llm-report":   mode_llm_report,
    "llm-prod":     mode_llm_prod,
    "llm-ablation": mode_llm_ablation,
    "gpr-opt":      mode_gpr_opt,
    "gpr-cv":       mode_gpr_cv,
    "gpr-prod":     mode_gpr_prod,
    "gpr-ablation": mode_gpr_ablation,
    "plot-only":    mode_plot_only,
    "validation-fig": mode_validation_fig,
    "data-fig":     mode_data_fig,
    "rebuild-bands": mode_rebuild_bands,
    "ablation-fig": mode_ablation_fig,
    "tuning-fig":   mode_tuning_fig,
    "gpr-ablation-fig": mode_gpr_ablation_fig,
}

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--data_csv", default="polymer_Ey_dataset_final.csv",
                    help="canonical 201-row dataset (default: polymer_Ey_dataset_final.csv)")
    ap.add_argument("--rg_dir", default="rg",
                    help="restricted-group split_* folders (default: rg)")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override winning epochs (else read from llm_best.json)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override inference temperature (else llm_temp.json, else 0.2)")
    ap.add_argument("--split", choices=["restricted-group", "random", "variable"],
                    default=None, help="run only this one split scheme: restricted-group, random, or variable (default: all)")
    ap.add_argument("--pred_csv", default=None,
                    help="plot-only: path to a saved *_predictions.csv to redraw")
    ap.add_argument("--force", action="store_true",
                    help="gpr-cv: refit even if saved results already exist")
    ap.add_argument("--fig_splits", default=None,
                    help="validation-fig: comma-separated splits to draw as columns, e.g. "
                         "'restricted-group,random' to exclude variable. Default: all three.")
    ap.add_argument("--retry_tries", type=int, default=None,
                    help="llm-cv / llm-ablation: transient-error retry budget per unit. "
                         "0 = retry forever. Backoff ramps to a 10 min ceiling for 5xx and a "
                         "4 h ceiling for rate-limit/daily-cap. Default: 8.")
    ap.add_argument("--prep_only", action="store_true",
                    help="LLM modes: write jsonl and stop (no API calls)")
    args = ap.parse_args()
    needs_rg = args.mode in ("temp-tune", "epoch-tune", "llm-cv", "llm-report", "llm-ablation",
                             "gpr-opt", "gpr-cv", "gpr-ablation", "tuning-fig")
    if needs_rg and not args.rg_dir:
        sys.exit(f"{args.mode} requires --rg_dir")
    if args.retry_tries is not None:
        global MAX_TRANSIENT_TRIES, MAX_RATELIMIT_TRIES
        MAX_TRANSIENT_TRIES = args.retry_tries
        MAX_RATELIMIT_TRIES = args.retry_tries
        print(f"submit retries: UNLIMITED -- 5xx ramps to a "
              f"{TRANSIENT_BACKOFF_CAP//60} min ceiling, rate-limit/daily-cap ramps to "
              f"{RATELIMIT_BACKOFF_CAP//3600}h and holds there. Ctrl-C to stop."
              if args.retry_tries == 0 else
              f"submit retries: {args.retry_tries} per unit")
    os.makedirs(args.out_dir, exist_ok=True)
    MODES[args.mode](args)

if __name__ == "__main__":
    main()