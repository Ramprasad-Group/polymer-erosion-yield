#!/usr/bin/env python3
"""
LEO erosion-yield pipeline: LLM fine-tuning and traditional-ML benchmark.

The split, eligibility rule, canonicalizer, and metrics are defined once and
shared by both paths, so the LLM and ML benchmarks always use the same test set.
The path is selected by mode.

Shared core:
  * canonicalizer: local canonicalize.canonicalize (not rdkit)
  * rare-chemistry eligibility: a chemistry (canonical pSMILES) is test-eligible
    if it occurs <= MAX_TEST_COUNT times, all its occurrences are MISSE, and all
    its rows have known layer count and thickness. The last clause keeps test rows
    present under every feature-ablation mask, so the LLM and ML test sets contain
    the same chemistries.
  * build_feature_ablation: single split, all eligible rows in test
  * metrics: OME, logR2, and chemistry-averaged OME_chem/logR2_chem

Modes:
  prep       canonicalize, build splits (CSV + JSONL), idempotent.   [LLM/shared]
  train      submit fine-tune jobs.                                  [LLM]
  retrieve   poll jobs, save fine-tuned model ids + result files.    [LLM]
  infer      run inference, aggregate combined CSVs.                 [LLM]
  figures    parity plots.                                           [LLM]
  autotune   resumable model/temp/epoch sweep.                       [LLM]
  all        prep -> train -> retrieve -> infer -> figures.          [LLM]
  ml-bench   ablation on the single split, then 5-fold production on the best
             config; ranked CSV by chemistry-averaged OME.           [ML]

Both paths build the split in-memory from DATASET_CSV via the shared builder, so
the split is identical regardless of which mode runs first; ml-bench also writes
the production fold CSVs to SPLITS_DIR for inspection.
"""

import os, re, sys, csv, json, time, glob, pickle, hashlib, argparse, warnings, traceback
from datetime import datetime, timezone
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

# Heavy ML deps are optional: only the ml-bench path needs them. Import guarded so
# the LLM modes work even if scikit-learn / rdkit are absent.
try:
    from sklearn.model_selection import GroupKFold, RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.base import BaseEstimator, RegressorMixin, clone
    from sklearn.exceptions import ConvergenceWarning
    from scipy.stats import loguniform, uniform, randint
    warnings.filterwarnings("default", category=ConvergenceWarning)
    _ML_DEPS = True
except Exception as _ml_imp_err:               # pragma: no cover
    _ML_DEPS = False
    class BaseEstimator:  # minimal fallbacks so top-level class defs still load
        pass
    class RegressorMixin:
        pass
    def clone(*a, **k):
        raise RuntimeError("scikit-learn not installed; ml-bench mode unavailable")
    GroupKFold = RandomizedSearchCV = Pipeline = StandardScaler = SimpleImputer = None
    VarianceThreshold = None
    loguniform = uniform = randint = None
try:
    from rdkit import Chem
except Exception:
    Chem = None


# SHARED CONFIG (paths, columns, split rule)
# Paths default to repo-relative locations so a fresh clone runs without edits;
# override any of them with the matching env var to point at another filesystem.
DATASET_CSV = os.environ.get("LEO_DATASET_CSV", "data/polymer_Ey_dataset_final.csv")
SPLITS_DIR  = os.environ.get("LEO_SPLITS_DIR",  "splits")   # CSV + JSONL splits + manifests
RESULTS_DIR = os.environ.get("LEO_RESULTS_DIR", "results")  # LLM inference CSVs + figures

# Column names.
COL_PSMILES   = "psmiles"
COL_POLYNAME  = "polymer name"
COL_COATING   = "coating name"
COL_MISSION   = "mission name"
COL_ORIENT    = "orientation"
COL_MTIME     = "mission time (yr)"
COL_SOLAR     = "solar exposure (esh)"
COL_FLUENCE   = "ao fluence (atoms/cm2)"
COL_LOGEY     = "log(e_y)"
COL_THICK     = "thickness (mm)"
COL_LAYERS    = "layers"
# ML-benchmark aliases (same underlying columns, the ML code uses these names).
PSMILES_COL   = COL_PSMILES
TARGET_COL    = COL_LOGEY
MISSION_COL   = COL_MISSION
LAYERS_COL    = COL_LAYERS
THICKNESS_COL = COL_THICK

# Split rule (shared by LLM and ML).
N_SPLITS    = 5
SEED        = 42
MISSE_MATCH = "misse"     # case-insensitive substring match on mission name
# Rare-chemistry RG threshold: a chemistry is test-eligible only if it occurs at
# most this many times (1 = singletons only; 2 = singletons + doubletons).
MAX_TEST_COUNT = 2


# SHARED CORE: canonicalizer + split + metrics

def _normalize_stars(s):
    # bare * -> [*]; leave [*] intact.
    return str(s).strip().replace("*", "[*]").replace("[[*]]", "[*]")


def canonicalize_psmiles(s):
    """Canonicalize one psmiles via canonicalize.canonicalize; None on failure.
    Imported lazily so non-prep modes don't require the module."""
    from canonicalize import canonicalize as _canon
    try:
        return _canon(_normalize_stars(s))
    except Exception:
        return None

# Alias so the ML-benchmark code uses the same canonicalizer as the LLM path.
canon = canonicalize_psmiles


def _misse_mask(df):
    """Case-insensitive MISSE match on the mission-name column (shared rule)."""
    return df[COL_MISSION].astype(str).str.contains(MISSE_MATCH, case=False, na=False)


def _eligibility(df, canon_col):
    """Row-level test-eligibility mask (rare-chemistry RG).

    A CHEMISTRY (canonical pSMILES) is test-eligible iff:
      (1) it occurs at most MAX_TEST_COUNT times in df,
      (2) every one of its occurrences is from a MISSE mission, AND
      (3) every one of its rows has a known layer count AND known thickness.
    Clause (3) guarantees eligible (test) rows survive every feature-ablation mask,
    so the LLM and ML test sets are the identical set of chemistries. All rows of an
    eligible chemistry go to test together; everything else is train-only."""
    counts = df[canon_col].value_counts()
    size = df[canon_col].map(counts)
    is_misse = _misse_mask(df)
    misse_per_chem = is_misse.astype(int).groupby(df[canon_col]).transform("sum")
    all_misse = misse_per_chem.eq(size)

    lay_known = pd.to_numeric(df[COL_LAYERS], errors="coerce").notna()
    thk_known = pd.to_numeric(df[COL_THICK], errors="coerce").notna()
    feat_known = (lay_known & thk_known).astype(int)
    all_known = feat_known.groupby(df[canon_col]).transform("min").eq(1)

    rare = size.le(MAX_TEST_COUNT)
    eligible = rare & all_misse & all_known

    n_excl_mixed = int((rare & all_known & ~all_misse).sum())
    n_excl_feat  = int((rare & all_misse & ~all_known).sum())
    if n_excl_mixed:
        print(f"  {n_excl_mixed} row(s) in rare chemistries excluded (non-MISSE occurrence); train-only.")
    if n_excl_feat:
        print(f"  {n_excl_feat} row(s) in rare MISSE chemistries excluded (missing layers/thickness); "
              f"train-only so test rows are identical across all conditions.")
    elig_chems = df.loc[eligible, canon_col]
    n_singletons = int(elig_chems.map(counts).eq(1).sum())
    n_doubletons = int(eligible.sum()) - n_singletons
    print(f"  test-eligible (count<={MAX_TEST_COUNT} & all-MISSE & known layers+thickness): "
          f"{int(eligible.sum())} rows / {elig_chems.nunique()} chemistries "
          f"({n_singletons} from singletons, {n_doubletons} from doubletons)")
    return eligible


def build_grouped_cv_splits(df, canon_col):
    """Deterministic chemistry-grouped N_SPLITS-fold CV over the ENTIRE dataset.
    Every row participates; the only grouping key is canonical pSMILES, so all
    measurements of a polymer (any mission/orientation) share one fold and a
    chemistry never appears in both train and test of any fold (no leakage).
    Each chemistry -- hence each row -- is tested exactly once across the folds.
    No RNG: chemistries are assigned largest-first to the currently-smallest fold
    (by row count), so folds are as size-balanced as the group sizes allow and the
    partition is identical on every run and for both the LLM and ML consumers."""
    sizes = df.groupby(canon_col).size()
    # largest-first; ties broken by canonical string for full determinism
    chems = sorted(sizes.index, key=lambda c: (-int(sizes[c]), str(c)))
    fold_chems = [[] for _ in range(N_SPLITS)]
    fold_rows  = [0] * N_SPLITS
    for c in chems:
        k = min(range(N_SPLITS), key=lambda j: (fold_rows[j], j))   # smallest fold, lowest index on tie
        fold_chems[k].append(c)
        fold_rows[k] += int(sizes[c])
    out = []
    for k in range(N_SPLITS):
        test_chems = set(fold_chems[k])
        test_idx = df.index[df[canon_col].isin(test_chems)].to_numpy()
        train_df = df.drop(index=test_idx)
        out.append((train_df, df.loc[test_idx]))
    return out


def build_feature_ablation(df, canon_col):
    """Single strict split: every test-eligible row (chemistry count<=MAX_TEST_COUNT
    & all-MISSE) in the test set; everything else in train. No eligible
    chemistry appears in train (both rows of a doubleton go to test)."""
    eligible = _eligibility(df, canon_col)
    test_df = df.loc[eligible]
    train_df = df.loc[~eligible]
    return train_df, test_df


def calculate_ome(t, p):
    return float(np.mean(np.abs(np.asarray(t) - np.asarray(p))))


def calculate_log_r2(t, p):
    t = np.asarray(t); p = np.asarray(p)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot

def chem_metrics(true, pred, canon_labels):
    """Chemistry-weighted metrics: every chemistry gets one vote total, so
    doubletons don't count twice, and per-row residuals are used directly (no
    collapsing/cancellation).
      OME_chem  : each row's |true-pred| averaged within a chemistry, then
                  averaged across chemistries (abs-then-average).
      logR2_chem: weighted R2 with row weight 1/n_c (chemistry c's n_c rows share
                  unit weight), computed on per-row residuals about the weighted
                  mean -- the R2 analogue of the same one-vote-per-chemistry
                  weighting (NOT a collapse of each chemistry to its mean).
    Returns (OME_chem, logR2_chem, n_chemistries)."""
    df = pd.DataFrame({"c": list(canon_labels),
                       "t": np.asarray(true, float),
                       "p": np.asarray(pred, float)})
    df["ae"] = (df["t"] - df["p"]).abs()
    df["w"]  = 1.0 / df.groupby("c")["c"].transform("count")   # 1/n_c -> one vote per chemistry
    per_chem_ae = df.groupby("c")["ae"].mean()
    n = int(len(per_chem_ae))
    ome = float(per_chem_ae.mean()) if n else None
    if n >= 2:
        w, t, p = df["w"].values, df["t"].values, df["p"].values
        tbar = float(np.sum(w * t) / np.sum(w))
        ss_res = float(np.sum(w * (t - p) ** 2))
        ss_tot = float(np.sum(w * (t - tbar) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        r2 = None
    return ome, r2, n


# LLM-SPECIFIC CONFIG
BASE_MODEL  = "gpt-4o-2024-08-06"
N_EPOCHS    = 5          # fine-tune epochs (int, or "auto" to let OpenAI pick)
MAX_ACTIVE_JOBS = 3      # OpenAI cap on simultaneous active fine-tune jobs per model
TEMPERATURE = 0.05
N_REPEATS   = 20          # inference repeats per split -> mean/std
INFER_TRAIN = True      # also run inference on the train split (figures only need test)
POLL_SECS   = 60         # retrieve: seconds between job polls (0 = single check, no loop)

# --- Prompt variants for the feature-ablation experiment ----------------
# All three share the identical rules suffix; only the feature list (system)
# and the corresponding clause (user) differ. base == the prompt used for the
# main runs (unchanged).
_RULES = (
    " Rules: 1) Use only the provided input fields. 2) Keep reasoning internal; "
    "do not explain your steps. 3) Do not include text, labels, or units. 4) Output "
    "only the final numeric value with 3 significant figures. 5) Negative values are "
    "allowed. 6) If prediction is not possible, output exactly null."
)
_PREDICT = (
    " predict the base-10 logarithm of the atomic oxygen erosion yield in "
    "(Angstroms^3/atom)."
)
_SYS_HEAD = (
    "You are a materials scientist specializing in low Earth orbit atomic oxygen "
    "interactions with polymers. Given a structured input describing a polymer sample, "
    "including the polymer name, the SMILES string, a description of the coating, "
)
_SYS_TAIL = (
    "the NASA mission from which the sample's data were obtained, the orientation of the "
    "sample during space exposure, the mission time (years of direct space exposure "
    "while attached to the ISS), the solar exposure (equivalent sun hours) and the "
    "atomic oxygen fluence (atoms/cm^2),"
)

SYSTEM_BASE  = _SYS_HEAD + _SYS_TAIL + _PREDICT + _RULES
SYSTEM_THICK = _SYS_HEAD + "the film thickness in millimeters, " + _SYS_TAIL + _PREDICT + _RULES
SYSTEM_LAYERS = _SYS_HEAD + "the number of stacked thin-film layers, " + _SYS_TAIL + _PREDICT + _RULES

USER_BASE = (
    "What is the base-10 logarithm of the atomic oxygen erosion yield of the polymer {} "
    "represented by SMILES {}, with {} coating, flown on the {} mission oriented in the "
    "{} direction for a mission time of {} years, subjected to a solar exposure of {} "
    "equivalent sun hours and an atomic oxygen fluence of {} atom/cm^2?"
)
USER_THICK = (
    "What is the base-10 logarithm of the atomic oxygen erosion yield of the polymer {} "
    "represented by SMILES {}, with {} coating, with a film thickness of {} mm, flown on "
    "the {} mission oriented in the {} direction for a mission time of {} years, subjected "
    "to a solar exposure of {} equivalent sun hours and an atomic oxygen fluence of {} "
    "atom/cm^2?"
)
USER_LAYERS = (
    "What is the base-10 logarithm of the atomic oxygen erosion yield of the polymer {} "
    "represented by SMILES {}, with {} coating, composed of {} stacked thin-film layers, "
    "flown on the {} mission oriented in the {} direction for a mission time of {} years, "
    "subjected to a solar exposure of {} equivalent sun hours and an atomic oxygen fluence "
    "of {} atom/cm^2?"
)

PROMPTS = {
    "base":      (SYSTEM_BASE,   USER_BASE),
    "thickness": (SYSTEM_THICK,  USER_THICK),
    "layers":    (SYSTEM_LAYERS, USER_LAYERS),
}

# --- The layers/thickness experiment: 4 matched-control conditions, each its own
# fine-tune. Mirrors the ML-bench ABLATION_CONDS exactly (no imputation):
#   baseline   = all rows, neither feature
#   control    = rows with BOTH layers & thickness known, neither feature (matched control)
#   layers     = those same rows + layer-count feature
#   thickness  = those same rows + thickness(mm) feature
# Each condition = (prompt_variant, row_filter, layers_fill). Run via
# `--split-type feature_ablation`. All four share the same MISSE count<=MAX_TEST_COUNT
# test set (eligibility guarantees every test row has known layers+thickness, so the
# has_both mask never drops a test point); test size is printed at prep.
#   row_filter:  "all" (201) | "has_both" (drop rows missing layers or thickness)
#   layers_fill: None
ABLATION_CONDITIONS = {
    #  name           prompt        rows          layers_fill
    "baseline":      ("base",       "all",        None),    # all rows, neither feature
    "control":       ("base",       "has_both",   None),    # both-known rows, neither feature (matched control)
    "layers":        ("layers",     "has_both",   None),    # same rows + layer count
    "thickness":     ("thickness",  "has_both",   None),    # same rows + thickness(mm)
}
ABLATION_VARIANTS = list(ABLATION_CONDITIONS)   # the 4 condition names = the iteration units

# Condition applied to the production 5-fold runs. Any of the four
# ABLATION_CONDITIONS keys: the chosen prompt + row-filter are applied to every
# fold. e.g. "control" = base prompt, drop rows missing layers/thickness;
# "baseline" = base prompt on all rows.
MAIN_CONDITION = "baseline"
assert MAIN_CONDITION in ABLATION_CONDITIONS, f"MAIN_CONDITION must be one of {list(ABLATION_CONDITIONS)}"

# Prompt variant for the main runs is derived from MAIN_CONDITION.
MAIN_VARIANT = ABLATION_CONDITIONS[MAIN_CONDITION][0]

# Back-compat alias (main runs use the base prompt).
SYSTEM_PROMPT = SYSTEM_BASE
USER_TEMPLATE = USER_BASE



# PATH HELPERS
def _unit_name(u):
    return f"split_{u:02d}" if isinstance(u, int) else str(u)


def split_units(split_type):
    """Iteration units: integer fold indices for production, prompt-variant
    names for the feature_ablation experiment (which prompt variant per model)."""
    if split_type == "feature_ablation":
        return list(ABLATION_VARIANTS)
    return list(range(1, N_SPLITS + 1))


def variant_for(split_type, unit):
    """Which prompt variant a given unit uses."""
    if split_type == "feature_ablation":
        return ABLATION_CONDITIONS[unit][0]   # condition -> prompt variant
    return MAIN_VARIANT


def split_dir(split_type, u):
    return os.path.join(SPLITS_DIR, split_type, _unit_name(u))


def manifest_path(split_type):
    return os.path.join(SPLITS_DIR, f"manifest_{split_type}.json")


def results_split_dir(split_type, u):
    return os.path.join(RESULTS_DIR, split_type, _unit_name(u))


def combined_csv_path(split_type, u, subset):
    prefix = {"feature_ablation": "FA", "production": "CV"}.get(split_type, "X")
    fname = f"{prefix}logall_4o_{_unit_name(u)}_combined_{subset}.csv"
    return os.path.join(results_split_dir(split_type, u), "combined", fname)


def load_manifest(split_type):
    p = manifest_path(split_type)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"base_model": BASE_MODEL, "splits": {}}


def save_manifest(split_type, m):
    os.makedirs(os.path.dirname(manifest_path(split_type)), exist_ok=True)
    with open(manifest_path(split_type), "w") as f:
        json.dump(m, f, indent=2)


# JSONL GENERATION
def _fmt_thickness(row):
    v = row.get(COL_THICK)
    if v is None or pd.isna(v):
        return "unspecified"
    return f"{float(v):g}"


def _fmt_layers(row):
    v = row.get(COL_LAYERS)
    if v is None or pd.isna(v):
        return "an unspecified number of"
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v)


def row_to_record(row, variant="base"):
    system, user_tmpl = PROMPTS[variant]
    coating = row[COL_COATING]
    if pd.isna(coating) or str(coating).strip() == "":
        coating = "no"
    solar = round(float(row[COL_SOLAR]))
    fluence = f"{float(row[COL_FLUENCE]):.3e}"
    erosion_yield = round(float(row[COL_LOGEY]), 3)

    name, smiles = row[COL_POLYNAME], row[COL_PSMILES]
    mission, orient = row[COL_MISSION], row[COL_ORIENT]
    mtime = row[COL_MTIME]

    if variant == "thickness":
        user = user_tmpl.format(name, smiles, coating, _fmt_thickness(row),
                                mission, orient, mtime, solar, fluence)
    elif variant == "layers":
        user = user_tmpl.format(name, smiles, coating, _fmt_layers(row),
                                mission, orient, mtime, solar, fluence)
    else:
        user = user_tmpl.format(name, smiles, coating, mission, orient,
                                mtime, solar, fluence)

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(erosion_yield)},
        ]
    }


def write_jsonl(df, path, variant="base"):
    with open(path, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row_to_record(row, variant)) + "\n")


def _has_layers(df):
    return pd.to_numeric(df[COL_LAYERS], errors="coerce").notna()

def _has_thick(df):
    return pd.to_numeric(df[COL_THICK], errors="coerce").notna()

def apply_condition(train_df, test_df, cond):
    """Return (train, test, prompt_variant) for one ablation condition:
    apply the row filter to both train and test. (No layer imputation: every
    current condition uses fill=None; the leakage-safe fill branch below is
    retained but dormant.)"""
    prompt, rows, fill = ABLATION_CONDITIONS[cond]
    tr, te = train_df.copy(), test_df.copy()
    if rows == "has_layers":
        tr, te = tr[_has_layers(tr)].copy(), te[_has_layers(te)].copy()
    elif rows == "has_thick":
        tr, te = tr[_has_thick(tr)].copy(), te[_has_thick(te)].copy()
    elif rows == "has_both":   # matched control: rows with BOTH layers & thickness known (mirrors ML assemble)
        tr = tr[_has_layers(tr) & _has_thick(tr)].copy()
        te = te[_has_layers(te) & _has_thick(te)].copy()
    if fill is not None:
        known = pd.to_numeric(tr[COL_LAYERS], errors="coerce").dropna()
        fillval = 1 if fill == "one" else int(round(known.mean()))   # integer layer count for the prompt
        for d in (tr, te):
            lay = pd.to_numeric(d[COL_LAYERS], errors="coerce")
            d[COL_LAYERS] = lay.where(lay.notna(), fillval)
    return tr, te, prompt


# SPLIT CONSTRUCTION




def _split_rule_sig():
    """Signature of the parameters that determine split membership, so a change to
    any of them invalidates split files on disk instead of silently reusing them."""
    return {"max_test_count": MAX_TEST_COUNT, "misse_match": MISSE_MATCH,
            "n_splits": N_SPLITS, "seed": SEED}


def _prompt_hash(variant):
    """Hash of the actual system+user prompt text for a variant, so editing the
    prompt wording invalidates on-disk JSONL even when the variant name is unchanged."""
    system, user_tmpl = PROMPTS[variant]
    return hashlib.sha1((system + "\n" + user_tmpl).encode()).hexdigest()[:12]


def _write_split(d, train_df, test_df, canon_col, variant, force, label, dataset_hash):
    os.makedirs(d, exist_ok=True)
    train_csv = os.path.join(d, "train.csv")
    test_csv  = os.path.join(d, "test.csv")
    train_jsonl = os.path.join(d, "train.jsonl")
    test_jsonl  = os.path.join(d, "test.jsonl")
    meta_path = os.path.join(d, "_split_meta.json")
    meta = {"dataset_hash": dataset_hash, "variant": variant,
            "prompt_hash": _prompt_hash(variant),
            "n_train": int(len(train_df)), "n_test": int(len(test_df)),
            "rule": _split_rule_sig()}
    files_exist = all(os.path.exists(p) for p in
                      [train_csv, test_csv, train_jsonl, test_jsonl])
    if (not force) and files_exist and os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        if old == meta:
            print(f"  {label}: exists and matches, skipping (use --force to rebuild)")
            return
        print(f"  {label}: on-disk split is stale (dataset/prompt/rule changed); rebuilding.")
    train_df.drop(columns=[canon_col]).to_csv(train_csv, index=False)
    test_df.drop(columns=[canon_col]).to_csv(test_csv, index=False)
    write_jsonl(train_df, train_jsonl, variant)
    write_jsonl(test_df, test_jsonl, variant)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {label}: train={len(train_df)}  test={len(test_df)}  variant={variant}  -> {d}")


def mode_prep(split_type, force=False):
    df = pd.read_csv(DATASET_CSV)
    dataset_hash = _file_sha(DATASET_CSV)
    print(f"Loaded {len(df)} rows from {DATASET_CSV}")

    # Canonicalize the psmiles column once.
    canon_col = "_canon_psmiles"
    print("Canonicalizing psmiles ...")
    df[canon_col] = df[COL_PSMILES].apply(canonicalize_psmiles)
    n_bad = int(df[canon_col].isna().sum())
    if n_bad:
        print(f"  WARNING: {n_bad} rows failed canonicalization; dropping them.")
        df = df.dropna(subset=[canon_col]).reset_index(drop=True)
    df[COL_PSMILES] = df[canon_col]  # carry canonical form into the CSVs

    if split_type == "feature_ablation":
        train_df, test_df = build_feature_ablation(df, canon_col)
        print(f"  base strict split: test={len(test_df)} eligible / train={len(train_df)}; "
              f"{len(ABLATION_CONDITIONS)} conditions, each its own fine-tune")
        for cond in ABLATION_VARIANTS:
            tr, te, prompt = apply_condition(train_df, test_df, cond)
            _write_split(split_dir(split_type, cond), tr, te, canon_col, prompt, force,
                         f"cond {cond} [{prompt}]", dataset_hash)
        print(f"prep done ({split_type}).")
        return

    if split_type == "production":
        splits = build_grouped_cv_splits(df, canon_col)
    else:
        raise ValueError(f"unknown split_type for prep: {split_type!r} "
                         f"(expected 'feature_ablation' or 'production')")

    for i, (train_df, test_df) in enumerate(splits, start=1):
        tr, te, prompt = apply_condition(train_df, test_df, MAIN_CONDITION)
        _write_split(split_dir(split_type, i), tr, te,
                     canon_col, prompt, force, f"split {i} [{MAIN_CONDITION}]", dataset_hash)
    print(f"prep done ({split_type}, condition='{MAIN_CONDITION}', prompt='{MAIN_VARIANT}').")


# OPENAI HELPERS
def _resolve_api_key():
    """Resolve the OpenAI API key: openai_api_key.txt (one line, gitignored)
    first, then the OPENAI_API_KEY env var. No key is stored in this file."""
    p = "openai_api_key.txt"
    if os.path.exists(p):
        with open(p) as f:
            k = f.read().strip()
        if k:
            return k
    env = os.environ.get("OPENAI_API_KEY")
    if env:
        return env
    raise RuntimeError(
        "No OpenAI API key found. Create openai_api_key.txt (one line: sk-proj-...) "
        "or export OPENAI_API_KEY."
    )


def _client():
    from openai import OpenAI
    return OpenAI(api_key=_resolve_api_key())


def _ts(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


_TERMINAL = {"succeeded", "failed", "cancelled"}


def _file_sha(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def _llm_fingerprint(split_type, u):
    """Provenance fingerprint for one unit: hashes the train/test jsonl together
    with the base model, epoch count, and prompt variant. A change to any of these
    yields a different fingerprint, so stale fine-tunes or cached predictions from
    a different config are detectable instead of being reused silently."""
    variant = variant_for(split_type, u)
    d = split_dir(split_type, u)
    payload = json.dumps({
        "train_sha": _file_sha(os.path.join(d, "train.jsonl")),
        "test_sha":  _file_sha(os.path.join(d, "test.jsonl")),
        "base_model": BASE_MODEL,
        "n_epochs": N_EPOCHS,
        "variant": variant,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _check_fingerprint(rec, split_type, u, kind):
    """Compare a manifest record's stored fingerprint to the current one. Returns
    True if it matches (or is absent, treated as legacy and backfilled by caller).
    Warns on mismatch so a changed prompt/split/epoch/dataset can't be reused
    unnoticed."""
    cur = _llm_fingerprint(split_type, u)
    old = rec.get("fingerprint")
    if old is None:
        return True   # legacy entry; caller backfills
    if old != cur:
        print(f"  [warn] {kind} for unit {u}: fingerprint mismatch "
              f"(stored {old} != current {cur}). The train/test jsonl, base model, "
              f"epochs, or prompt changed since this was produced; pass --force to "
              f"regenerate, or the stored result may be stale.")
        return False
    return True


def _submit_split(client, split_type, m, u):
    """Upload train.jsonl and create one fine-tune job for unit u.
    Returns True if submitted, "ratelimit" if the org cap was hit, False if skipped."""
    from openai import RateLimitError
    key = str(u)
    train_jsonl = os.path.join(split_dir(split_type, u), "train.jsonl")
    if not os.path.exists(train_jsonl):
        print(f"  {key}: missing {train_jsonl}; run prep first.")
        return False
    try:
        with open(train_jsonl, "rb") as f:
            up = client.files.create(file=f, purpose="fine-tune")
        job = client.fine_tuning.jobs.create(
            training_file=up.id,
            model=BASE_MODEL,
            method={
                "type": "supervised",
                "supervised": {"hyperparameters": {"n_epochs": N_EPOCHS}},
            },
        )
    except RateLimitError:
        print(f"  {key}: org active-job cap hit; will retry as slots free.")
        return "ratelimit"
    except Exception as e:
        # 403/401/400/etc. — not a transient cap. Stop cleanly; rerun resumes.
        print(f"  {key}: submission failed ({type(e).__name__}: {e}). "
              f"Not retrying; fix access and rerun to resume.")
        return "error"
    rec = m["splits"].get(key, {})
    rec.update({"train_file_id": up.id, "job_id": job.id, "status": job.status,
                "fingerprint": _llm_fingerprint(split_type, u)})
    m["splits"][key] = rec
    save_manifest(split_type, m)
    print(f"  {key}: submitted job {job.id} (file {up.id})")
    return True


def _poll_split(client, split_type, m, u):
    """Poll one submitted job; on success save model + result file. Returns status."""
    key = str(u)
    rec = m["splits"][key]
    job = client.fine_tuning.jobs.retrieve(rec["job_id"])
    rec["status"] = job.status
    if job.status == "succeeded":
        rec["fine_tuned_model"] = job.fine_tuned_model
        if getattr(job, "result_files", None):
            rid = job.result_files[0]
            resp = client.files.with_raw_response.retrieve_content(rid)
            out = os.path.join(split_dir(split_type, u), f"{job.id}_result.csv")
            with open(out, "wb") as f:
                f.write(resp.content)
            rec["result_file"] = out
        print(f"  {key}: SUCCEEDED -> {job.fine_tuned_model}")
    elif job.status in ("failed", "cancelled"):
        print(f"  {key}: {job.status.upper()} {getattr(job, 'error', '')}")
    m["splits"][key] = rec
    save_manifest(split_type, m)
    return job.status


def _orchestrate(split_type, submit=True, wait=True):
    """Cap-aware submit+poll loop. Submits up to MAX_ACTIVE_JOBS, and as jobs
    reach a terminal state, submits the next pending unit. With wait=True it
    loops until every unit is done or terminal; with wait=False it does one pass."""
    client = _client()
    m = load_manifest(split_type)
    m["base_model"] = BASE_MODEL
    units = split_units(split_type)

    # Provenance check. A legacy entry (no stored fingerprint) is only trusted and
    # backfilled when --accept-legacy is passed; otherwise it is left unfingerprinted
    # so downstream inference refuses to reuse it. On a fingerprint mismatch, --force
    # drops the stale job/model for re-submission.
    for u in units:
        rec = m["splits"].get(str(u))
        if not rec or not rec.get("job_id"):
            continue
        if rec.get("fingerprint") is None:
            if FORCE and rec.get("fine_tuned_model"):
                print(f"  [force] unit {u}: clearing legacy no-fingerprint job/model for re-submission.")
                for fld in ("job_id", "fine_tuned_model", "status", "result_file", "train_file_id"):
                    rec.pop(fld, None)
                rec["fingerprint"] = _llm_fingerprint(split_type, u)
            elif ACCEPT_LEGACY:
                rec["fingerprint"] = _llm_fingerprint(split_type, u)
                print(f"  [accept-legacy] unit {u}: backfilled fingerprint for existing model.")
            elif rec.get("fine_tuned_model"):
                print(f"  [warn] unit {u}: fine-tuned model has no provenance fingerprint; "
                      f"it will not be trusted for inference. Pass --accept-legacy to bless it, "
                      f"or --force to retrain.")
        elif not _check_fingerprint(rec, split_type, u, "fine-tune") and FORCE:
            print(f"  [force] unit {u}: clearing stale job/model for re-submission.")
            for fld in ("job_id", "fine_tuned_model", "status", "result_file", "train_file_id"):
                rec.pop(fld, None)
            rec["fingerprint"] = _llm_fingerprint(split_type, u)
    save_manifest(split_type, m)

    def done(u):
        r = m["splits"].get(str(u), {})
        return bool(r.get("fine_tuned_model")) or r.get("status") in _TERMINAL

    while True:
        for u in units:
            rec = m["splits"].get(str(u), {})
            if rec.get("job_id") and not rec.get("fine_tuned_model") \
               and rec.get("status") not in _TERMINAL:
                _poll_split(client, split_type, m, u)

        active = sum(1 for u in units
                     if m["splits"].get(str(u), {}).get("job_id") and not done(u))
        unsubmitted = [u for u in units if not m["splits"].get(str(u), {}).get("job_id")]

        hit_error = False
        if submit and unsubmitted and active < MAX_ACTIVE_JOBS:
            for u in list(unsubmitted):
                if active >= MAX_ACTIVE_JOBS:
                    break
                res = _submit_split(client, split_type, m, u)
                if res == "ratelimit":
                    break
                if res == "error":
                    hit_error = True
                    break
                if res is True:
                    active += 1

        all_done = all(done(u) for u in units)
        if all_done or not wait:
            break

        # Don't spin forever if we can't submit and nothing is in flight to wait on.
        if hit_error and active == 0:
            print("  Cannot submit and no active jobs to wait on; stopping. "
                  "Manifest is preserved — rerun to resume once access is restored.")
            break

        n_left = sum(1 for u in units if not done(u))
        if POLL_SECS > 0:
            print(f"  {n_left} unit(s) outstanding; sleeping {POLL_SECS}s ...")
            time.sleep(POLL_SECS)
        else:
            break

    n_models = sum(1 for u in units
                   if m["splits"].get(str(u), {}).get("fine_tuned_model"))
    print(f"orchestrate done ({split_type}): {n_models}/{len(units)} models ready.")


def mode_train(split_type):
    # Submit up to the cap, single pass, no waiting.
    _orchestrate(split_type, submit=True, wait=False)


def mode_retrieve(split_type):
    # Poll to completion, backfilling new submissions as slots free.
    _orchestrate(split_type, submit=True, wait=True)


# INFERENCE + AGGREGATION
def _run_inference(client, model, jsonl_path):
    """Return list of dicts {question, pred (str), truth (str)} for one pass.
    Transient API/network/5xx errors are waited out and retried indefinitely
    (same policy as the autotune path), so a dropped connection mid-pass resumes
    instead of aborting the run. A genuine non-transient failure (e.g. a 400 for
    one input) records pred=None for that row and moves on."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            sys_msg = obj["messages"][0]["content"]
            user_msg = obj["messages"][1]["content"]
            truth = obj["messages"][2]["content"]

            def _do():
                return client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=TEMPERATURE,
                )
            try:
                resp = _retry_forever(_do, what=f"chat({model})")
                pred = resp.choices[0].message.content
            except Exception as e:
                print(f"    hard error on one row ({type(e).__name__}: {e}); pred=None.")
                pred = None
            rows.append({"question": user_msg, "pred": pred, "truth": truth})
    return rows


def _combine_repeats(repeats):
    """Aggregate a list of per-pass row-lists (same row order) into a DataFrame
    with columns row_id, question, true, meanpred, stdpred, n_valid, n_invalid,
    invalid_rate. _parse_pred handles null/blank/messy outputs. Rows with no valid
    prediction across any repeat are excluded from the returned frame and reported
    separately (second return value) rather than dropped silently. A row's
    invalid_rate exposes low-confidence rows (e.g. n_valid=1, where stdpred is 0
    only because a single repeat survived)."""
    base = repeats[0]
    n = len(base)
    n_rep = len(repeats)
    out, audit = [], []
    for j in range(n):
        q = base[j]["question"]
        truth = _parse_pred(base[j]["truth"])
        preds = [_parse_pred(r[j]["pred"]) for r in repeats]
        preds = [p for p in preds if p is not None]
        n_valid = len(preds)
        n_invalid = n_rep - n_valid
        if not preds or truth is None:
            audit.append({"row_id": j, "question": q,
                          "true": (float(truth) if truth is not None else None),
                          "n_valid": n_valid, "n_invalid": n_invalid,
                          "invalid_rate": n_invalid / n_rep if n_rep else None})
            continue
        out.append({
            "row_id": j,
            "question": q,
            "true": float(truth),
            "meanpred": float(np.mean(preds)),
            "stdpred": float(np.std(preds, ddof=0)),
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "invalid_rate": n_invalid / n_rep if n_rep else None,
        })
    if audit:
        print(f"    {len(audit)}/{n} row(s) had no valid prediction across "
              f"{n_rep} repeat(s); recorded in the audit CSV.")
    return pd.DataFrame(out), pd.DataFrame(audit)


def _attach_row_metadata(df, split_type, u, subset, model):
    """Add authoritative provenance columns to a combined/audit frame using the
    split's on-disk {subset}.csv, joined by row order (write_jsonl emits rows in
    the same order as the CSV, and row_id is that line index). This carries
    canon_psmiles, orientation, split_id, condition, and model_id directly, so
    downstream metrics don't have to recover chemistry/orientation from prompt
    text. No-op if df is empty or the CSV is missing."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df["split_id"] = _unit_name(u)
    df["subset"] = subset
    df["model_id"] = model
    df["condition"] = u if split_type == "feature_ablation" else MAIN_CONDITION
    csv_path = os.path.join(split_dir(split_type, u), f"{subset}.csv")
    if os.path.exists(csv_path) and "row_id" in df.columns:
        src = pd.read_csv(csv_path)
        get = lambda col: (src[col].reindex(df["row_id"]).to_numpy()
                           if col in src.columns else None)
        canon = get(COL_PSMILES)
        orient = get(COL_ORIENT)
        if canon is not None:
            df["canon_psmiles"] = canon
        if orient is not None:
            df["orientation"] = orient
    return df


def mode_infer(split_type):
    client = _client()
    m = load_manifest(split_type)

    subsets = ["test"] + (["train"] if INFER_TRAIN else [])
    for u in split_units(split_type):
        key = str(u)
        rec = m["splits"].get(key, {})
        model = rec.get("fine_tuned_model")
        if not model:
            print(f"  {key}: no fine_tuned_model in manifest; run retrieve.")
            continue
        # Block inference on a model whose training provenance no longer matches the
        # current split/prompt/epochs, unless --force is set to override.
        if rec.get("fingerprint") is not None \
                and rec["fingerprint"] != _llm_fingerprint(split_type, u) and not FORCE:
            print(f"  {key}: manifest fingerprint {rec['fingerprint']} != current "
                  f"{_llm_fingerprint(split_type, u)}; the split/prompt/epochs changed "
                  f"since this model was trained. Skipping (re-run train, or pass "
                  f"--force to infer with the existing model anyway).")
            continue
        # A model with no stored fingerprint has unverifiable provenance; refuse
        # unless explicitly accepted or forced.
        if rec.get("fingerprint") is None and not ACCEPT_LEGACY and not FORCE:
            print(f"  {key}: fine-tuned model has no provenance fingerprint; skipping. "
                  f"Pass --accept-legacy to use it, or --force.")
            continue
        rdir = results_split_dir(split_type, u)
        raw_dir = os.path.join(rdir, "raw")
        comb_dir = os.path.join(rdir, "combined")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(comb_dir, exist_ok=True)

        for subset in subsets:
            jsonl = os.path.join(split_dir(split_type, u), f"{subset}.jsonl")
            if not os.path.exists(jsonl):
                print(f"  {key} [{subset}]: missing {jsonl}; skipping.")
                continue
            out_csv = combined_csv_path(split_type, u, subset)
            meta_path = os.path.splitext(out_csv)[0] + ".meta.json"
            cur_meta = {"model_id": model, "temperature": TEMPERATURE,
                        "n_repeats": N_REPEATS, "variant": variant_for(split_type, u),
                        "fingerprint": _llm_fingerprint(split_type, u)}
            # If a combined result exists under different settings, do NOT silently
            # recombine stale raw repeats under fresh metadata. Refuse unless --force,
            # and when forced, regenerate every raw repeat rather than reuse it.
            stale = False
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    old_meta = json.load(f)
                diff = {k: (old_meta.get(k), v) for k, v in cur_meta.items()
                        if old_meta.get(k) != v}
                if diff:
                    stale = True
                    if not FORCE:
                        print(f"  {key} [{subset}]: cached predictions were made under "
                              f"different settings {diff}; refusing to reuse. Pass --force "
                              f"to regenerate from scratch.")
                        continue
                    print(f"  {key} [{subset}]: settings changed {diff}; regenerating all "
                          f"repeats (raw cache ignored).")
            elif os.path.exists(out_csv):
                # Combined result with no provenance sidecar: can't confirm the
                # settings it was made under, so don't trust it.
                stale = True
                if not FORCE:
                    print(f"  {key} [{subset}]: cached combined has no metadata sidecar; "
                          f"refusing to reuse. Pass --force to regenerate from scratch.")
                    continue
                print(f"  {key} [{subset}]: no metadata sidecar; regenerating all repeats.")
            if os.path.exists(out_csv) and not FORCE and not stale:
                print(f"  {key} [{subset}]: combined exists, skipping (use --force to redo).")
                continue
            reuse_raw = (not FORCE) and (not stale)   # stale settings never reuse raw
            repeats = []
            for r in range(1, N_REPEATS + 1):
                rep_csv = os.path.join(raw_dir, f"{subset}_rep{r}.csv")
                if os.path.exists(rep_csv) and reuse_raw:
                    print(f"  {key} [{subset}] repeat {r}/{N_REPEATS}: reusing saved {rep_csv}")
                    rows = pd.read_csv(rep_csv).to_dict("records")
                    repeats.append(rows)
                    continue
                print(f"  {key} [{subset}] repeat {r}/{N_REPEATS} ...")
                rows = _run_inference(client, model, jsonl)
                repeats.append(rows)
                pd.DataFrame(rows).to_csv(
                    rep_csv, index=False, quoting=csv.QUOTE_ALL, escapechar="\\")
            combined, audit = _combine_repeats(repeats)
            combined = _attach_row_metadata(combined, split_type, u, subset, model)
            combined.to_csv(out_csv, index=False)
            with open(meta_path, "w") as f:
                json.dump(cur_meta, f, indent=2)
            if not audit.empty:
                audit_csv = os.path.splitext(out_csv)[0] + "_invalid.csv"
                _attach_row_metadata(audit, split_type, u, subset, model).to_csv(audit_csv, index=False)
                print(f"  {key} [{subset}]: {len(audit)} invalid row(s) -> {audit_csv}")
            print(f"  {key} [{subset}]: combined {len(combined)} rows -> {out_csv}")

    print(f"infer done ({split_type}).")


# FIGURES (GPT-4o only)
_EPS = 1e-300

ORIENT_STYLE = {
    "ram":    {"color": "#8B246C", "marker": "o"},
    "nadir":  {"color": "#934f06", "marker": "s"},
    "wake":   {"color": "#17becf", "marker": "D"},
    "zenith": {"color": "#bcbd22", "marker": "^"},
}
RG_COLORS  = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
RG_MARKERS = ["o", "s", "D", "^", "v"]

FULL_MIN, FULL_MAX = 0.001, 100
ZOOM_MIN, ZOOM_MAX = 0.1, 10

MAIN_TITLE_FONT = 30
SUB_TITLE_FONT = 24
AXIS_LABEL_FONT = 22
TICK_LABEL_FONT = 18
LEGEND_FONT = 18
METRIC_FONT = 18
SUBPLOT_LABEL_FONT = 20

CUSTOM_XLABEL = r"True Value ($\mathrm{\AA^3/atom}$)"
CUSTOM_YLABEL = r"Predicted Value ($\mathrm{\AA^3/atom}$)"





def _load_combined(path):
    df = pd.read_csv(path)
    for c in ["true", "meanpred"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "stdpred" in df.columns:
        df = df.rename(columns={"stdpred": "errbar"})
    if "errbar" in df.columns:
        df["errbar"] = pd.to_numeric(df["errbar"], errors="coerce")
        df = df.dropna(subset=["true", "meanpred", "errbar"]).copy()
        df["errbar"] = df["errbar"].clip(lower=0)
    else:
        df = df.dropna(subset=["true", "meanpred"]).copy()
        df["errbar"] = 0.0
    df["true_lin"] = np.power(10.0, df["true"])
    df["meanpred_lin"] = np.power(10.0, df["meanpred"])
    df["errbar_lin"] = df["meanpred_lin"] * np.log(10) * df["errbar"]
    df.loc[(df["meanpred_lin"] - df["errbar_lin"]) <= 0, "errbar_lin"] = 0.0
    df["x_true"] = df["true_lin"].clip(lower=_EPS)
    df["y_pred"] = df["meanpred_lin"].clip(lower=_EPS)
    df["yerr"] = df["errbar_lin"]
    return df


def _extract_orientation(q):
    ql = str(q).lower()
    for o in ["ram", "nadir", "wake", "zenith"]:
        if o in ql:
            return o
    return "unknown"


def _format_ax(ax, xmin, xmax, subtitle=None, show_ylabel=False, show_xlabel=False):
    from matplotlib.ticker import LogLocator, NullLocator, NullFormatter, FuncFormatter
    ax.plot([xmin, xmax], [xmin, xmax], "--", color="k", alpha=0.7, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xmin, xmax); ax.set_ylim(xmin, xmax)
    ax.set_aspect("equal", adjustable="box")
    plain = FuncFormatter(lambda v, pos: f"{v:g}")
    ax.xaxis.set_major_formatter(plain); ax.yaxis.set_major_formatter(plain)
    ax.xaxis.set_major_locator(LogLocator(base=10.0)); ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT)
    ax.grid(True, which="major", linestyle=":")
    if subtitle:
        ax.set_title(subtitle, fontsize=SUB_TITLE_FONT, pad=10)
    if show_ylabel:
        ax.set_ylabel(CUSTOM_YLABEL, fontsize=AXIS_LABEL_FONT)
    if show_xlabel:
        ax.set_xlabel(CUSTOM_XLABEL, fontsize=AXIS_LABEL_FONT)


_QSMILES_RE = re.compile(r"represented by SMILES (.*?), with .*? coating")

def _canon_from_question(q):
    """Recover the chemistry key for a combined-CSV row: pull the pSMILES out of
    the question prompt and canonicalize it. Lets figure metrics be chemistry-
    averaged without re-inference (the combined CSVs predate a stored chem key)."""
    m = _QSMILES_RE.search(str(q))
    return canonicalize_psmiles(m.group(1)) if m else None

def _canon_series(sub):
    """Chemistry key per row: use the authoritative canon_psmiles column when the
    CSV carries it, falling back to parsing the prompt only for older CSVs that
    predate that column."""
    if "canon_psmiles" in sub.columns:
        return sub["canon_psmiles"].astype("string")
    return sub["question"].map(_canon_from_question)

def _chem_om_r2(sub):
    """Chemistry-weighted (OME_chem, logR2_chem) for a combined-CSV subset: OME is
    per-row abs error averaged within each chemistry then across chemistries; logR2
    uses per-row residuals with weight 1/n_c so each chemistry contributes one vote
    (doubletons aren't double-weighted). Returns NaNs if no usable rows."""
    c = _canon_series(sub)
    mask = c.notna() & sub["true"].notna() & sub["meanpred"].notna()
    if not mask.any():
        return float("nan"), float("nan")
    ome, r2, _ = chem_metrics(sub["true"][mask].to_numpy(),
                              sub["meanpred"][mask].to_numpy(), c[mask].tolist())
    return ome, (r2 if r2 is not None else float("nan"))


def _ablation_figure():
    """3-panel parity comparison for the feature_ablation
    experiment (base / thickness / layers), each on the same rare-chemistry test set."""
    import matplotlib.pyplot as plt

    have = {}
    for variant in ABLATION_VARIANTS:
        p = combined_csv_path("feature_ablation", variant, "test")
        if os.path.exists(p):
            have[variant] = _load_combined(p)
    if not have:
        print("No feature_ablation results found. Run infer --split-type feature_ablation first.")
        return

    titles = {"baseline": "Baseline", "control": "Control (matched)",
              "layers": "+ Layers", "thickness": "+ Thickness"}
    # Fixed 2x2: baseline / control on top, thickness / layers on the bottom.
    grid = [["baseline", "control"], ["thickness", "layers"]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), squeeze=False)
    print("\n=== Feature-ablation metrics (rare-chemistry test set) ===")
    for r, row in enumerate(grid):
        for c, variant in enumerate(row):
            ax = axes[r][c]
            d = have.get(variant)
            if d is None:
                ax.axis("off"); continue
            ax.errorbar(d["x_true"], d["y_pred"], yerr=d["yerr"], fmt="o", markersize=7,
                        linewidth=1.2, color="#1f77b4", alpha=0.85, capsize=3, zorder=3)
            _format_ax(ax, FULL_MIN, FULL_MAX, subtitle=titles.get(variant, variant),
                       show_ylabel=(c == 0), show_xlabel=(r == 1))
            ome, r2 = _chem_om_r2(d)
            ax.text(0.97, 0.03, f"OME = {ome:.3f}\nLogR\u00b2 = {r2:.3f}",
                    transform=ax.transAxes, fontsize=METRIC_FONT, va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
            print(f"  {variant:10s} n={len(d):4d}  OME_chem={ome:.3f}  logR2_chem={r2:.3f}")

    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_svg = os.path.join(RESULTS_DIR, "feature_ablation_figure.svg")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(os.path.splitext(out_svg)[0] + ".eps", format="eps", bbox_inches="tight")
    fig.savefig(os.path.splitext(out_svg)[0] + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Saved: {out_svg}")


def mode_figures(split_type="production"):
    if split_type == "autotune":
        _autotune_figure()
        _autotune_model_parity_figure()
        _autotune_final_parity_figure()
    elif split_type == "feature_ablation":
        _ablation_figure()
    elif split_type == "production":
        _production_cv_figure()
    else:
        raise ValueError(f"no figure for split_type {split_type!r}")


def _chem_om_r2_ci(sub, n_boot=2000, seed=0):
    """Chemistry-resampled bootstrap 95% CI for (OME_chem, logR2_chem): resample the
    UNIQUE chemistries with replacement and recompute chem_metrics each draw, so the
    interval is uncertainty over which test chemistries were sampled (n = #chemistries)."""
    c = _canon_series(sub)
    mask = c.notna() & sub["true"].notna() & sub["meanpred"].notna()
    if not mask.any():
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    t = sub["true"][mask].to_numpy(); p = sub["meanpred"][mask].to_numpy(); cc = c[mask].to_numpy()
    by = {}
    for k, ci in enumerate(cc):
        by.setdefault(ci, []).append(k)
    chems = list(by); rng = np.random.default_rng(seed)
    omes, r2s = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(chems), len(chems), replace=True)
        tb, pb, lab = [], [], []
        for m, j in enumerate(pick):                     # unique label per draw preserves multiplicity
            rows = by[chems[j]]
            tb.append(t[rows]); pb.append(p[rows])
            lab.append(np.full(len(rows), f"{chems[j]}__{m}"))
        o, r, _ = chem_metrics(np.concatenate(tb), np.concatenate(pb), np.concatenate(lab))
        if o is not None: omes.append(o)
        if r is not None: r2s.append(r)
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a else (float("nan"), float("nan"))
    return q(omes), q(r2s)


def _production_cv_figure():
    """Single-panel parity for the chemistry-grouped 5-fold CV over the whole dataset,
    pooled out-of-fold; chem-weighted OME/logR2 with chemistry-resampled 95% CIs."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    splits = []
    for i in range(1, N_SPLITS + 1):
        p = combined_csv_path("production", i, "test")
        if os.path.exists(p):
            d = _load_combined(p); d["split"] = f"Fold {i}"; splits.append(d)
    if not splits:
        print("No production combined results found. Run infer --split-type production first.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    handles = []
    for i, d in enumerate(splits):
        ax.errorbar(d["x_true"], d["y_pred"], yerr=d["yerr"],
                    fmt=RG_MARKERS[i % len(RG_MARKERS)], markersize=7, linewidth=1.2,
                    color=RG_COLORS[i % len(RG_COLORS)], alpha=0.9, capsize=3, zorder=3)
        handles.append(mlines.Line2D([], [], color=RG_COLORS[i % len(RG_COLORS)],
                       marker=RG_MARKERS[i % len(RG_MARKERS)], linestyle="None",
                       markersize=8, label=f"Fold {i+1}"))
    _format_ax(ax, FULL_MIN, FULL_MAX, subtitle="Chemistry-Grouped 5-Fold CV",
               show_ylabel=True, show_xlabel=True)
    comb = pd.concat(splits, ignore_index=True)
    ome, r2 = _chem_om_r2(comb)
    (ol, oh), (rl, rh) = _chem_om_r2_ci(comb)
    ax.text(0.97, 0.03,
        f"OME = {ome:.3f} [{ol:.3f}, {oh:.3f}]\nLogR\u00b2 = {r2:.3f} [{rl:.3f}, {rh:.3f}]",
        transform=ax.transAxes, fontsize=METRIC_FONT - 2, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(handles=handles, loc="upper left", fontsize=LEGEND_FONT,
              frameon=True, title="Fold", title_fontsize=LEGEND_FONT)

    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_svg = os.path.join(RESULTS_DIR, "production_cv_figure.svg")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(os.path.splitext(out_svg)[0] + ".eps", format="eps", bbox_inches="tight")
    print(f"Saved: {out_svg}  (OME_chem={ome:.3f} [{ol:.3f},{oh:.3f}], "
          f"logR2_chem={r2:.3f} [{rl:.3f},{rh:.3f}])")





# AUTOTUNE FIGURE (temp + epoch sweeps)
# 2x2 line-plot figure of the autotune sweeps (the model sweep is reported as a
# LaTeX table, not here). Both sweeps mirror each other: OME on the LEFT,
# LogR2 on the RIGHT; the best point in each panel is ringed.
#   (a) OME vs epochs   (b) LogR2 vs epochs
#   (c) OME vs T        (d) LogR2 vs T
# Reads only autotune_state.json. Toggle USE_CHEM for chem-averaged vs row-level.
USE_CHEM = True   # True -> OME_chem/logR2_chem (selection metrics), False -> row-level

def _autotune_load_eval_recs():
    """Return {key: rec} for completed eval recs belonging to the CURRENT run
    (key prefix = current dataset_hash|split_hash). Filtering by prefix drops
    stale recs left in the state file by earlier runs on a different dataset/
    split, which would otherwise duplicate every epoch/temperature panel."""
    if not os.path.exists(AUTOTUNE_STATE):
        return {}
    st = json.load(open(AUTOTUNE_STATE))
    pfx = f"{st.get('dataset_hash')}|{st.get('split_hash')}"
    return {k: r for k, r in st.get("eval", {}).items()
            if r.get("status") == "complete" and k.startswith(pfx)}

def _autotune_dedupe(items, axis_key):
    """Collapse {key: rec} to one rec per distinct axis value (epochs or
    temperature), preferring the most-complete cell (max total_calls) so a
    partial resume never wins over a finished one."""
    best = {}
    for r in items.values():
        v = r.get(axis_key)
        if v is None:
            continue
        cur = best.get(v)
        if cur is None or (r.get("total_calls") or 0) > (cur.get("total_calls") or 0):
            best[v] = r
    return [best[v] for v in sorted(best)]

def _autotune_figure():
    """Epoch and temperature sweeps as mirrored line plots (OME left, LogR2
    right), read straight from the autotune eval recs."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    recs = _autotune_load_eval_recs()
    if not recs:
        print(f"No completed eval recs in {AUTOTUNE_STATE}. Run autotune first."); return
    ome_k = "OME_chem" if USE_CHEM else "OME"
    r2_k  = "logR2_chem" if USE_CHEM else "logR2"

    cfg_temps = {round(float(t), 4) for t in AUTOTUNE_TEMPS}
    epo  = _autotune_dedupe({k: r for k, r in recs.items()
                             if r.get("stage") == "epoch_sweep"
                             and r.get(ome_k) is not None}, "epochs")
    temp = _autotune_dedupe({k: r for k, r in recs.items()
                             if r.get("stage") == "temperature_sweep"
                             and r.get(ome_k) is not None
                             and r.get("temperature") is not None
                             and round(float(r["temperature"]), 4) in cfg_temps},
                            "temperature")
    sweeps = [("epochs", "Epochs", epo), ("temperature", "Temperature", temp)]
    sweeps = [s for s in sweeps if s[2]]
    if not sweeps:
        print("No epoch_sweep / temperature_sweep recs found."); return

    AXIS, TICK, LAB = 18, 14, 20
    r2_lab  = r"LogR$^2$" + (" (chem)" if USE_CHEM else "")
    ome_lab = "OME" + (" (chem)" if USE_CHEM else "")
    n_rows = len(sweeps)
    fig = plt.figure(figsize=(13, 6.0 * n_rows))
    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.32, wspace=0.32)

    def _panel(a, x, y, xlabel, ylabel, color, marker, best, int_x=False):
        yv = [np.nan if v is None else v for v in y]
        a.plot(x, yv, marker + "-", color=color, markersize=8, linewidth=2, zorder=3)
        pts = [(xi, yi) for xi, yi in zip(x, yv) if yi == yi]   # drop NaN cells
        if pts:
            bx, by = (min if best == "min" else max)(pts, key=lambda p: p[1])
            a.scatter([bx], [by], s=260, facecolors="none", edgecolors="k",
                      linewidths=2.0, zorder=5)                 # ring the selected point
        a.set_xlabel(xlabel, fontsize=AXIS); a.set_ylabel(ylabel, fontsize=AXIS)
        a.tick_params(axis="both", labelsize=TICK); a.grid(True, linestyle=":")
        if int_x:
            a.set_xticks(sorted(x))
        a.set_aspect(1.0 / a.get_data_ratio(), adjustable="box")

    labels = iter([f"({chr(97 + i)})" for i in range(2 * n_rows)])
    for row, (axis_key, xlabel, data) in enumerate(sweeps):
        x  = [r[axis_key] for r in data]
        o  = [r[ome_k] for r in data]
        r2 = [r[r2_k] for r in data]
        int_x = (axis_key == "epochs")
        ax_ome = fig.add_subplot(gs[row, 0])   # OME on the LEFT
        ax_r2  = fig.add_subplot(gs[row, 1])   # LogR2 on the RIGHT
        _panel(ax_ome, x, o,  xlabel, ome_lab, "#d62728", "s", "min", int_x)
        _panel(ax_r2,  x, r2, xlabel, r2_lab,  "#1f77b4", "o", "max", int_x)
        for a in (ax_ome, ax_r2):
            a.text(-0.10, 1.02, next(labels), transform=a.transAxes,
                   fontsize=LAB, fontweight="bold")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_svg = os.path.join(RESULTS_DIR, "autotune_temp_epoch_figure.svg")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(os.path.splitext(out_svg)[0] + ".eps", format="eps", bbox_inches="tight")
    fig.savefig(os.path.splitext(out_svg)[0] + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Saved: {out_svg}")


# AUTOTUNE PARITY PLOTS (appendix)
# Parity (predicted-vs-true) plots for the model sweep and the finally selected
# config, rebuilt from the per-repeat raw predictions saved under AUTOTUNE_DIR/raw
# during the sweep -- so NO re-inference is needed. Row order in every rep*.csv
# matches the test jsonl, so a row's screen prediction is the mean over its valid
# repeats. Same log-log parity style, orientation colors, and chem-weighted
# OME/logR2 box as the production figure, so these drop straight into the SI.

def _autotune_short_model(name):
    """Drop the trailing -YYYY-MM-DD snapshot date for a readable panel title."""
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", str(name))


def _autotune_raw_dir_for_key(key):
    """Recompute a cell's raw dir from its eval key (identical to the path used
    when the sweep wrote it). Deriving it from the key -- not a stored absolute
    path -- keeps parity plots working after the results tree is moved."""
    return os.path.join(AUTOTUNE_DIR, "raw", re.sub(r"[^A-Za-z0-9._-]", "_", key))


def _autotune_parity_df(key):
    """Rebuild a parity DataFrame for one eval cell from its rep*.csv files.
    Columns mirror what _format_ax / chem_metrics expect: question, true, meanpred
    (all log10), plus linear x_true/y_pred/yerr and orientation. A row's meanpred
    is the mean over its VALID repeats (invalid parses dropped, never zero-filled).
    Returns None if the raw dir is missing or yields no valid rows."""
    rd = _autotune_raw_dir_for_key(key)
    reps = sorted(glob.glob(os.path.join(rd, "rep*.csv")))
    if not reps:
        return None
    per_row = {}
    for rp in reps:
        try:
            d = pd.read_csv(rp, escapechar="\\")
        except Exception as e:
            print(f"[parity] skipping unreadable {rp} ({type(e).__name__})")
            continue
        for i, r in d.iterrows():
            slot = per_row.setdefault(i, {"q": r.get("question"),
                                          "truth": r.get("truth"), "preds": []})
            v = _parse_pred(r.get("pred"))
            if v is not None:
                slot["preds"].append(v)
    q, tr, mp = [], [], []
    for i in sorted(per_row):
        s = per_row[i]
        truth = _parse_pred(s["truth"])
        if truth is None or not s["preds"]:
            continue
        q.append(s["q"]); tr.append(truth); mp.append(float(np.mean(s["preds"])))
    if not tr:
        return None
    df = pd.DataFrame({"question": q, "true": tr, "meanpred": mp})
    df["orientation"] = df["question"].map(_extract_orientation)
    df["x_true"] = np.power(10.0, df["true"]).clip(lower=_EPS)
    df["y_pred"] = np.power(10.0, df["meanpred"]).clip(lower=_EPS)
    df["yerr"] = 0.0
    return df


_AUTOTUNE_UNKNOWN_COLOR = "#b0b0b0"   # rows whose chemistry key can't be recovered

def _autotune_chem_colors(dfs, cmap_name="turbo"):
    """Canonical-pSMILES -> color map shared across panels, so a chemistry keeps
    one hue everywhere (a doubleton's two rows share it). Sorted keys for
    determinism; no legend, since points are grouped by chemistry rather than
    orientation or fold and individual identities aren't decoded."""
    import matplotlib.pyplot as plt
    keys = set()
    for df in dfs:
        keys |= set(df["question"].map(_canon_from_question).dropna().unique())
    keys = sorted(keys)
    cmap = plt.get_cmap(cmap_name)
    n = max(len(keys), 1)
    return {k: cmap((i + 0.5) / n) for i, k in enumerate(keys)}


def _autotune_parity_panel(ax, df, subtitle, show_ylabel, show_xlabel,
                           metric_fs=None, winner=False, chem_colors=None):
    """Draw one log-log parity panel colored by chemistry (canonical pSMILES),
    with a chem-weighted OME/logR2 box (masking rows whose chemistry key can't be
    recovered, exactly like _chem_om_r2). Winner panels get a green bold title."""
    canon = df["question"].map(_canon_from_question)
    cmap = chem_colors or {}
    pt = [cmap.get(c, _AUTOTUNE_UNKNOWN_COLOR) for c in canon]
    ax.scatter(df["x_true"], df["y_pred"], s=48, marker="o", c=pt,
               alpha=0.9, edgecolors="none", zorder=3)
    _format_ax(ax, FULL_MIN, FULL_MAX, subtitle=subtitle,
               show_ylabel=show_ylabel, show_xlabel=show_xlabel)
    if winner:
        ax.title.set_color("#2ca02c"); ax.title.set_fontweight("bold")
    c = df["question"].map(_canon_from_question)
    m = c.notna() & df["true"].notna() & df["meanpred"].notna()
    if m.any():
        ome, r2, _ = chem_metrics(df["true"][m].to_numpy(),
                                  df["meanpred"][m].to_numpy(), c[m].to_numpy())
        r2 = float("nan") if r2 is None else r2
        ax.text(0.97, 0.03, f"OME = {ome:.3f}\nLogR\u00b2 = {r2:.3f}",
                transform=ax.transAxes,
                fontsize=metric_fs if metric_fs is not None else METRIC_FONT,
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="gray", alpha=0.9))


def _autotune_model_parity_figure():
    """Appendix: per-base-model parity for the stage-1 model sweep, one panel per
    candidate model (AUTOTUNE_MODELS order), winner titled in green/bold."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.gridspec import GridSpec

    recs = _autotune_load_eval_recs()
    ms = {k: r for k, r in recs.items() if r.get("stage") == "model_sweep"}
    if not ms:
        print("No model_sweep eval recs. Run autotune first."); return
    best = {}                                   # base_model -> most-complete key
    for k, r in ms.items():
        bm = r.get("base_model")
        if bm is None:
            continue
        if bm not in best or (r.get("total_calls") or 0) > (ms[best[bm]].get("total_calls") or 0):
            best[bm] = k
    order = [m for m in AUTOTUNE_MODELS if m in best] + \
            [m for m in best if m not in AUTOTUNE_MODELS]
    winner_bm = min(best, key=lambda m: ms[best[m]].get("OME_chem")
                    if ms[best[m]].get("OME_chem") is not None else 1e18)

    panels = []
    for bm in order:
        df = _autotune_parity_df(best[bm])
        if df is not None:
            panels.append((bm, df))
        else:
            print(f"[parity] no raw preds for model sweep cell: {bm}")
    if not panels:
        print("No raw predictions found for the model sweep; nothing to plot."); return

    chem_colors = _autotune_chem_colors([df for _, df in panels])

    # Page-friendly layout: 2 columns (portrait). Non-winners fill the top block
    # row-major; the winner sits alone on a fresh bottom row at column 0. For the
    # common 5-model case this is a 2x2 of non-winners with the winner beneath
    # column 0 (3 rows x 2 cols, bottom-right cell empty) -> fits one sheet.
    nonwin = [(bm, df) for bm, df in panels if bm != winner_bm]
    win    = [(bm, df) for bm, df in panels if bm == winner_bm]
    ncols  = 2
    rows_nw = (len(nonwin) + ncols - 1) // ncols
    nrows   = rows_nw + (1 if win else 0)
    nrows   = max(nrows, 1)

    placements = []                              # (bm, df, row, col, is_winner)
    for i, (bm, df) in enumerate(nonwin):
        r, c = divmod(i, ncols)
        placements.append((bm, df, r, c, False))
    if win:
        placements.append((win[0][0], win[0][1], rows_nw, 0, True))

    bottom_of_col = {}                           # lowest occupied row per column
    for _, _, r, c, _ in placements:
        bottom_of_col[c] = max(bottom_of_col.get(c, -1), r)

    fig = plt.figure(figsize=(6.2 * ncols, 6.0 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.30)
    for idx, (bm, df, r, c, is_win) in enumerate(placements):
        ax = fig.add_subplot(gs[r, c])
        _autotune_parity_panel(ax, df, _autotune_short_model(bm),
                               show_ylabel=(c == 0),
                               show_xlabel=(r == bottom_of_col[c]),
                               metric_fs=METRIC_FONT - 4,
                               winner=is_win, chem_colors=chem_colors)
        ax.text(-0.15, 1.05, f"({chr(97 + idx)})", transform=ax.transAxes,
                fontsize=SUBPLOT_LABEL_FONT, fontweight="bold")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_svg = os.path.join(RESULTS_DIR, "autotune_model_parity_figure.svg")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(os.path.splitext(out_svg)[0] + ".eps", format="eps", bbox_inches="tight")
    fig.savefig(os.path.splitext(out_svg)[0] + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Saved: {out_svg}  (winner={_autotune_short_model(winner_bm)})")


def _autotune_final_parity_figure():
    """Appendix: parity for the finally selected config -- the epoch-sweep winner,
    falling back to the temperature/model sweep if a later stage is absent."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    recs = _autotune_load_eval_recs()
    if not recs:
        print("No eval recs. Run autotune first."); return
    key = rec = None
    for stage in ("epoch_sweep", "temperature_sweep", "model_sweep"):
        cand = [(k, r) for k, r in recs.items()
                if r.get("stage") == stage and r.get("OME_chem") is not None]
        if cand:
            key, rec = min(cand, key=lambda kr: kr[1]["OME_chem"])
            break
    if key is None:
        print("No scored eval recs to plot."); return

    df = _autotune_parity_df(key)
    if df is None:
        print(f"No raw predictions under {_autotune_raw_dir_for_key(key)}."); return

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    bits = [_autotune_short_model(rec.get("base_model"))]
    if rec.get("temperature") is not None:
        bits.append(f"T={rec['temperature']:g}")
    if rec.get("epochs") is not None:
        bits.append(f"{rec['epochs']} ep")
    _autotune_parity_panel(ax, df, "Tuned Config (" + ", ".join(bits) + ")",
                           show_ylabel=True, show_xlabel=True,
                           chem_colors=_autotune_chem_colors([df]))
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_svg = os.path.join(RESULTS_DIR, "autotune_final_parity_figure.svg")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(os.path.splitext(out_svg)[0] + ".eps", format="eps", bbox_inches="tight")
    fig.savefig(os.path.splitext(out_svg)[0] + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Saved: {out_svg}")


# AUTOTUNE (cheap pre-screen)
# A one-shot, resumable, tmux-friendly sweep run BEFORE the production runs.
# Test set = the feature_ablation baseline test set (every test-eligible row:
#   canonical psmiles count<=MAX_TEST_COUNT, all-MISSE, and known layers+thickness;
#   both rows of a doubleton go to test together), so the screen runs on the same
#   test rows as the reported preliminary-holdout numbers.
# Train set = the MISSE-only common-chemistry pool (MISSE rows whose chemistry
#   occurs > MAX_TEST_COUNT times), NOT the feature_ablation train remainder. This
#   is deliberate: it keeps the training data fixed across dataset edits so existing
#   fine-tunes stay valid and are reused. base prompt, no feature.
# This is a SCREEN for picking hyperparameters, not a separately reported metric.
#
# Stages (greedy, in order; no full grid):
#   1) model sweep   (epochs=5, temp=0.2)        -> best base model
#   2) temperature sweep (reuse winning model)   -> best inference temperature
#   3) epoch sweep   (best base, best temp)       -> final config
# Selection at every stage: lowest OME, then highest logR2, then lowest invalid rate.

AUTOTUNE_DIR        = os.path.join(RESULTS_DIR, "autotune")
AUTOTUNE_SPLIT_DIR  = os.path.join(SPLITS_DIR, "autotune", "baseline")
AUTOTUNE_STATE      = os.path.join(AUTOTUNE_DIR, "autotune_state.json")

# Fixed candidate list of base models to ATTEMPT supervised fine-tuning on. This
# is NOT auto-discovery of every fine-tunable model (OpenAI exposes no reliable
# endpoint for that) — it is a minimum candidate set. Entries the account cannot
# fine-tune (unavailable / non-fine-tunable / deprecated) are logged as skipped
# and ignored. Edit this list to add/remove candidates.
AUTOTUNE_MODELS = [
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
]
AUTOTUNE_FT_EPOCHS = 5            # stage-1 fine-tune epochs
AUTOTUNE_FT_TEMP   = 0.2         # stage-1 inference temperature
# Inference-temperature sweep grid: 0.0 -> 0.5 in 0.05 steps. Edit the range to rescale.
AUTOTUNE_TEMPS     = [round(0.05 * i, 2) for i in range(0, 11)]   # [0.0, 0.05, ..., 0.5]
AUTOTUNE_EPOCHS    = [3, 5, 7]
AUTOTUNE_N_REPEATS = 20
AUTOTUNE_POLL_SECS = max(POLL_SECS, 30)   # never busy-loop the FT queue


def _sha8(b):
    return hashlib.sha1(b).hexdigest()[:8]


def _autotune_keypfx(state):
    """Cache-key prefix tying every FT/eval cell to the exact dataset + split
    (and therefore prompt_variant, which is baked into the split jsonl). A
    changed dataset/split/prompt yields new keys -> cache miss -> recompute,
    never silent reuse of stale results."""
    return f"{state['dataset_hash']}|{state['split_hash']}"


def _is_transient(e):
    """True for rate limits, timeouts, connection drops, and 5xx — things that
    should be waited out, not treated as model invalidity or a fatal run error."""
    name = type(e).__name__
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError",
                "APIConnectionTimeoutError", "InternalServerError"}:
        return True
    code = getattr(e, "status_code", None)
    return code is not None and (code == 429 or code >= 500)


def _retry_forever(fn, what, cap=None):
    """Call fn(), retrying transient API/network errors indefinitely with capped
    exponential backoff. Re-raises immediately on non-transient errors (e.g. a
    400/401/403), which the caller handles. Keeps overnight runs alive through
    outages without recording garbage."""
    cap = cap or AUTOTUNE_POLL_SECS
    i = 0
    while True:
        try:
            return fn()
        except Exception as e:
            if not _is_transient(e):
                raise
            wait = min(cap, 2.0 * (2 ** min(i, 6)))
            print(f"[autotune] transient {type(e).__name__} on {what}; "
                  f"retrying in {wait:.0f}s ...")
            time.sleep(wait)
            i += 1


def _autotune_save(state):
    os.makedirs(AUTOTUNE_DIR, exist_ok=True)
    tmp = AUTOTUNE_STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, AUTOTUNE_STATE)


def _autotune_load_state():
    if os.path.exists(AUTOTUNE_STATE):
        with open(AUTOTUNE_STATE) as f:
            return json.load(f)
    return {"ft": {}, "eval": {}, "skipped": {}}


def _parse_pred(s):
    """Parse a model output into a float, or None if invalid.
    Invalid = null / empty / non-numeric / NaN / ambiguous (0 or >1 numbers)."""
    if s is None:
        return None
    t = str(s).strip()
    if not t or t.lower() == "null":
        return None
    try:
        v = float(t)
        return None if np.isnan(v) else v
    except ValueError:
        pass
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
    if len(nums) == 1:
        try:
            v = float(nums[0])
            return None if np.isnan(v) else v
        except ValueError:
            return None
    return None


def _autotune_build_split():
    """Build (idempotently) the baseline feature_ablation split (identical test
    set to the ablation/production runs) and return a state dict carrying hashes
    + jsonl paths. Rebuilds the jsonl from data each run (deterministic); warns
    if the resulting split_hash changed."""
    os.makedirs(AUTOTUNE_SPLIT_DIR, exist_ok=True)
    train_csv   = os.path.join(AUTOTUNE_SPLIT_DIR, "train.csv")
    test_csv    = os.path.join(AUTOTUNE_SPLIT_DIR, "test.csv")
    train_jsonl = os.path.join(AUTOTUNE_SPLIT_DIR, "train.jsonl")
    test_jsonl  = os.path.join(AUTOTUNE_SPLIT_DIR, "test.jsonl")

    with open(DATASET_CSV, "rb") as f:
        dataset_hash = _sha8(f.read())
    df = pd.read_csv(DATASET_CSV)
    print(f"[autotune] loaded {len(df)} rows from {DATASET_CSV}")

    canon_col = "_canon_psmiles"
    df[canon_col] = df[COL_PSMILES].apply(canonicalize_psmiles)
    n_bad = int(df[canon_col].isna().sum())
    if n_bad:
        print(f"[autotune] WARNING: {n_bad} rows failed canonicalization; dropping.")
        df = df.dropna(subset=[canon_col]).reset_index(drop=True)
    df[COL_PSMILES] = df[canon_col]

    is_misse = df[COL_MISSION].astype(str).str.contains(MISSE_MATCH, case=False, na=False)
    mdf = df[is_misse].reset_index(drop=True)   # MISSE-only screen pool (unchanged)
    print(f"[autotune] MISSE-only subset: {len(mdf)} rows")

    # Train: UNCHANGED original screen train (MISSE rows whose chemistry occurs
    # > MAX_TEST_COUNT times within the MISSE pool) so existing fine-tunes stay
    # valid and are reused (no retraining).
    counts = mdf[canon_col].value_counts()
    chem_size = mdf[canon_col].map(counts)
    train_df = mdf[chem_size.gt(MAX_TEST_COUNT)]
    # Test: the canonical feature_ablation test set, computed over the FULL dataset
    # exactly as build_feature_ablation does (count<=MAX_TEST_COUNT & all-MISSE &
    # known layers+thickness) -> identical rows to every reported number. These are
    # all-MISSE & rare, so none sit in train above; no leakage.
    test_df = df.loc[_eligibility(df, canon_col)]
    print(f"[autotune] split: train={len(train_df)} (unchanged)  "
          f"test={len(test_df)} (canonical baseline rows)")
    tr, te, prompt = apply_condition(train_df, test_df, "baseline")
    tr.drop(columns=[canon_col]).to_csv(train_csv, index=False)
    te.drop(columns=[canon_col]).to_csv(test_csv, index=False)
    write_jsonl(tr, train_jsonl, prompt)
    write_jsonl(te, test_jsonl, prompt)

    with open(train_jsonl, "rb") as f1, open(test_jsonl, "rb") as f2:
        split_hash = _sha8(f1.read() + f2.read())

    state = _autotune_load_state()
    state.setdefault("ft", {}); state.setdefault("eval", {}); state.setdefault("skipped", {})
    state["dataset_hash"] = dataset_hash
    state["split_hash"]   = split_hash
    state["split_type"]   = "feature_ablation_baseline"
    state["prompt_variant"] = prompt
    state["train_jsonl"]  = train_jsonl
    state["test_jsonl"]   = test_jsonl
    state["test_n"]       = len(te)
    # Canonical psmiles per test row, in the same order as test.jsonl lines
    # (write_jsonl iterates te in order). Enables chemistry-averaged metrics so
    # doubletons aren't weighted twice.
    state["test_canon"]   = te[canon_col].astype(str).tolist()

    # Re-key fine-tunes onto THIS split and drop everything tied to any other
    # split. The training data is unchanged, so existing models stay valid and are
    # reused (no retraining); only the test set / eval results differ. Pruning
    # stale eval cells means the state + summary reflect only the canonical test
    # set, leaving no record of any prior one.
    pfx = _autotune_keypfx(state)
    migrated = {}
    for v in state["ft"].values():
        bm, ep = v.get("base_model"), v.get("epochs")
        if bm is None or ep is None:
            continue
        nk = f"{pfx}|{bm}|ep{ep}"
        if v.get("fine_tuned_model"):
            migrated[nk] = {"base_model": bm, "epochs": ep,
                            "job_id": v.get("job_id"), "status": "succeeded",
                            "fine_tuned_model": v["fine_tuned_model"]}
        elif v.get("skipped") or v.get("status") in ("failed", "cancelled"):
            migrated.setdefault(nk, {"base_model": bm, "epochs": ep,
                                     "skipped": True, "status": v.get("status", "failed"),
                                     "error": v.get("error", "")})
    state["ft"] = migrated
    state["eval"] = {k: r for k, r in state["eval"].items() if k.startswith(pfx + "|")}
    _autotune_save(state)
    return state


def _autotune_ensure_ft(state, client, base_model, epochs):
    """Submit (if needed) and wait for one fine-tune (base_model, epochs).
    Returns the fine_tuned_model id, or None if the model was skipped/failed.
    Resumable: reuses any existing job/model. Active-job cap and transient
    errors are waited out and retried indefinitely."""
    key = f"{_autotune_keypfx(state)}|{base_model}|ep{epochs}"
    rec = state["ft"].setdefault(key, {"base_model": base_model, "epochs": epochs})

    if rec.get("fine_tuned_model"):
        return rec["fine_tuned_model"]
    if rec.get("skipped") or rec.get("status") in ("failed", "cancelled"):
        return None

    # Train data does not depend on which rows are held out for TEST, so reuse any
    # existing succeeded fine-tune for this (base_model, epochs) even if the split
    # hash changed (e.g. the test set was repointed). Prevents needless retraining.
    for v in state["ft"].values():
        if (v.get("base_model") == base_model and v.get("epochs") == epochs
                and v.get("fine_tuned_model")):
            rec["fine_tuned_model"] = v["fine_tuned_model"]
            rec["job_id"] = v.get("job_id", rec.get("job_id"))
            rec["status"] = "succeeded"
            _autotune_save(state)
            print(f"[autotune] {key}: reusing existing model {v['fine_tuned_model']}")
            return rec["fine_tuned_model"]

    # Upload train.jsonl once PER SPLIT; the file id is shared by every FT job
    # for that split and re-uploaded if the split changes.
    train_files = state.setdefault("train_files", {})
    if not train_files.get(state["split_hash"]):
        with open(state["train_jsonl"], "rb") as f:
            up = client.files.create(file=f, purpose="fine-tune")
        train_files[state["split_hash"]] = up.id
        _autotune_save(state)
        print(f"[autotune] uploaded train file {up.id}")
    train_file_id = train_files[state["split_hash"]]

    # Submit if no job yet (transient errors are waited out; hard errors skip).
    if not rec.get("job_id"):
        def _do_create():
            return client.fine_tuning.jobs.create(
                training_file=train_file_id,
                model=base_model,
                method={"type": "supervised",
                        "supervised": {"hyperparameters": {"n_epochs": epochs}}},
            )
        try:
            job = _retry_forever(_do_create, what=f"ft.create {key}")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            rec["status"] = "failed"; rec["skipped"] = True; rec["error"] = msg
            state["skipped"][base_model] = msg
            _autotune_save(state)
            print(f"[autotune] {key}: SKIPPED ({msg})")
            return None
        rec["job_id"] = job.id; rec["status"] = job.status
        _autotune_save(state)
        print(f"[autotune] {key}: submitted job {job.id}")

    # Poll to terminal (transient retrieve errors are waited out).
    while True:
        job = _retry_forever(lambda: client.fine_tuning.jobs.retrieve(rec["job_id"]),
                             what=f"ft.poll {key}")
        rec["status"] = job.status
        if job.status == "succeeded":
            rec["fine_tuned_model"] = job.fine_tuned_model
            _autotune_save(state)
            print(f"[autotune] {key}: SUCCEEDED -> {job.fine_tuned_model}")
            return job.fine_tuned_model
        if job.status in ("failed", "cancelled"):
            rec["skipped"] = True
            rec["error"] = str(getattr(job, "error", "") or job.status)
            state["skipped"][base_model] = rec["error"]
            _autotune_save(state)
            print(f"[autotune] {key}: {job.status.upper()} {rec['error']}")
            return None
        _autotune_save(state)
        print(f"[autotune] {key}: {job.status}; sleeping {AUTOTUNE_POLL_SECS}s ...")
        time.sleep(AUTOTUNE_POLL_SECS)


def _autotune_call(client, model, obj, temperature):
    """Return the model's raw output string. Transient API/network/5xx errors
    are waited out indefinitely (never counted as model invalidity). Only a
    genuine non-transient failure (e.g. a 400 for this input) returns None, which
    the caller counts as an invalid prediction."""
    sys_msg = obj["messages"][0]["content"]
    user_msg = obj["messages"][1]["content"]

    def _do():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            temperature=temperature,
        )
    try:
        r = _retry_forever(_do, what=f"chat({model})")
    except Exception as e:
        print(f"[autotune] hard error on chat({model}) ({type(e).__name__}: {e}); "
              f"counting this prediction invalid.")
        return None
    return r.choices[0].message.content


def _autotune_eval(state, client, stage, base_model, epochs, fine_tuned_model,
                   temperature, n_repeats):
    """Inference + metrics for one cell. Resumable: returns cached result if
    already complete. Tracks invalid predictions at BOTH the call level
    (invalid_call_n/total_calls) and the row level (a row is invalid only if
    every repeat failed). Saves raw per-repeat predictions for debugging.
    Nothing is silently dropped."""
    key = (f"{_autotune_keypfx(state)}|{stage}|{base_model}|ep{epochs}"
           f"|t{temperature:.2f}|r{n_repeats}")
    cached = state["eval"].get(key)
    if cached and cached.get("status") == "complete":
        return cached

    with open(state["test_jsonl"]) as f:
        items = [json.loads(line) for line in f]
    test_n = len(items)

    raw_dir = os.path.join(AUTOTUNE_DIR, "raw", re.sub(r"[^A-Za-z0-9._-]", "_", key))
    os.makedirs(raw_dir, exist_ok=True)

    valids = [[] for _ in items]
    invalid_call_n = 0
    total_calls = 0
    for rpt in range(1, n_repeats + 1):
        print(f"[autotune] eval {key}: repeat {rpt}/{n_repeats}")
        raw_rows = []
        for i, obj in enumerate(items):
            raw = _autotune_call(client, fine_tuned_model, obj, temperature)
            v = _parse_pred(raw)
            total_calls += 1
            if v is None:
                invalid_call_n += 1
            else:
                valids[i].append(v)
            raw_rows.append({"question": obj["messages"][1]["content"],
                             "pred": raw, "truth": obj["messages"][2]["content"]})
        pd.DataFrame(raw_rows).to_csv(
            os.path.join(raw_dir, f"rep{rpt}.csv"),
            index=False, quoting=csv.QUOTE_ALL, escapechar="\\")

    test_canon = state.get("test_canon")
    trues, means, invalid_row_n, row_canon = [], [], 0, []
    for i, obj in enumerate(items):
        truth = _parse_pred(obj["messages"][2]["content"])
        if truth is None or not valids[i]:
            invalid_row_n += 1
            continue
        trues.append(truth)
        means.append(float(np.mean(valids[i])))
        row_canon.append(test_canon[i] if test_canon and i < len(test_canon) else str(i))
    valid_n = len(trues)
    ome   = calculate_ome(trues, means) if valid_n else None
    logr2 = calculate_log_r2(trues, means) if valid_n >= 2 else None
    invalid_call_rate = (invalid_call_n / total_calls) if total_calls else None

    # Chemistry-averaged metrics: collapse each chemistry to its mean (true, pred)
    # first, so doubletons count once rather than twice.
    # Chemistry-weighted metrics via chem_metrics (abs-then-average OME, so
    # within-chemistry over/under-predictions don't cancel; one vote per chemistry).
    ome_chem = logr2_chem = None
    n_chem = None
    if valid_n:
        ome_chem, logr2_chem, n_chem = chem_metrics(trues, means, row_canon)

    rec = {
        "stage": stage, "base_model": base_model, "epochs": epochs,
        "temperature": temperature, "n_repeats": n_repeats,
        "fine_tuned_model": fine_tuned_model,
        "test_n": test_n, "valid_n": valid_n, "n_chem": n_chem,
        "invalid_row_n": invalid_row_n,
        "invalid_row_rate": (invalid_row_n / test_n) if test_n else None,
        "invalid_call_n": invalid_call_n, "total_calls": total_calls,
        # invalid_rate (reported + used for selection) = call-level rate.
        "invalid_n": invalid_call_n,
        "invalid_rate": invalid_call_rate,
        # Chemistry-averaged metrics (OME_chem/logR2_chem) drive selection so
        # doubletons aren't double-weighted; row-weighted OME/logR2 kept for SI.
        "OME": ome, "logR2": logr2,
        "OME_chem": ome_chem, "logR2_chem": logr2_chem,
        "status": "complete", "error": "",
        "raw_dir": raw_dir,
    }
    state["eval"][key] = rec
    _autotune_save(state)
    print(f"[autotune] eval {key}: OME={ome}  logR2={logr2}  "
          f"OME_chem={ome_chem}  logR2_chem={logr2_chem}  "
          f"invalid_calls={invalid_call_n}/{total_calls}  invalid_rows={invalid_row_n}/{test_n}")
    return rec


def _autotune_pick(state, stage):
    """Best cell for a stage: lowest OME_chem, then highest logR2_chem, then
    lowest invalid rate. Chemistry-averaged so doubletons aren't double-weighted."""
    cand = [r for r in state["eval"].values()
            if r.get("stage") == stage and r.get("status") == "complete"
            and r.get("OME_chem") is not None]
    if not cand:
        return None
    return sorted(cand, key=lambda r: (r["OME_chem"],
                                       -(r["logR2_chem"] if r["logR2_chem"] is not None else -1e18),
                                       r["invalid_rate"] if r["invalid_rate"] is not None else 1.0))[0]


def _autotune_write_outputs(state, best):
    os.makedirs(AUTOTUNE_DIR, exist_ok=True)
    summary = os.path.join(AUTOTUNE_DIR, "autotune_summary.csv")
    cols = ["stage", "base_model", "epochs", "temperature", "n_repeats",
            "fine_tuned_model", "test_n", "valid_n", "n_chem",
            "invalid_n", "invalid_rate", "invalid_call_n", "total_calls",
            "invalid_row_n", "invalid_row_rate",
            "OME", "logR2", "OME_chem", "logR2_chem", "status", "error"]
    order = {"model_sweep": 0, "temperature_sweep": 1, "epoch_sweep": 2}
    rows = sorted(state["eval"].values(),
                  key=lambda r: (order.get(r.get("stage"), 9),
                                 r.get("OME_chem") if r.get("OME_chem") is not None else 1e18))
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[autotune] wrote {summary}")

    skipped = os.path.join(AUTOTUNE_DIR, "autotune_skipped_models.csv")
    with open(skipped, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base_model", "error"])
        for mdl, err in state.get("skipped", {}).items():
            w.writerow([mdl, err])
    print(f"[autotune] wrote {skipped}")

    best_cfg = os.path.join(AUTOTUNE_DIR, "autotune_best_config.json")
    payload = {
        "best_base_model": best.get("base_model") if best else None,
        "best_fine_tuned_model": best.get("fine_tuned_model") if best else None,
        "best_temperature": best.get("temperature") if best else None,
        "best_epochs": best.get("epochs") if best else None,
        "OME_chem": best.get("OME_chem") if best else None,
        "logR2_chem": best.get("logR2_chem") if best else None,
        "OME": best.get("OME") if best else None,          # per-row, for reference/SI
        "logR2": best.get("logR2") if best else None,       # per-row, for reference/SI
        "invalid_rate": best.get("invalid_rate") if best else None,
        "split_type": state.get("split_type"),
        "dataset_hash": state.get("dataset_hash"),
        "split_hash": state.get("split_hash"),
    }
    with open(best_cfg, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[autotune] wrote {best_cfg}")


def mode_autotune():
    client = _client()
    state = _autotune_build_split()

    # --- Stage 1: model sweep (epochs=5, temp=0.2) ---
    print("\n[autotune] === Stage 1: model sweep ===")
    for mdl in AUTOTUNE_MODELS:
        ftm = _autotune_ensure_ft(state, client, mdl, AUTOTUNE_FT_EPOCHS)
        if not ftm:
            continue
        _autotune_eval(state, client, "model_sweep", mdl, AUTOTUNE_FT_EPOCHS,
                       ftm, AUTOTUNE_FT_TEMP, AUTOTUNE_N_REPEATS)
    best1 = _autotune_pick(state, "model_sweep")
    if not best1:
        print("[autotune] no fine-tunable models succeeded; nothing to tune.")
        _autotune_write_outputs(state, None)
        return
    best_base = best1["base_model"]
    best_ft5  = best1["fine_tuned_model"]
    print(f"[autotune] stage-1 winner: {best_base}  (OME_chem={best1['OME_chem']:.3f})")

    # --- Stage 2: temperature sweep (reuse winning model, no retrain) ---
    print("\n[autotune] === Stage 2: temperature sweep ===")
    for t in AUTOTUNE_TEMPS:
        _autotune_eval(state, client, "temperature_sweep", best_base,
                       AUTOTUNE_FT_EPOCHS, best_ft5, t, AUTOTUNE_N_REPEATS)
    best2 = _autotune_pick(state, "temperature_sweep")
    best_temp = best2["temperature"] if best2 else AUTOTUNE_FT_TEMP
    print(f"[autotune] stage-2 winner: temp={best_temp}")

    # --- Stage 3: epoch sweep (best base, best temp; reuse ep5 from stage 1) ---
    print("\n[autotune] === Stage 3: epoch sweep ===")
    for ep in AUTOTUNE_EPOCHS:
        ftm = _autotune_ensure_ft(state, client, best_base, ep)  # ep5 reused from stage 1
        if not ftm:
            continue
        _autotune_eval(state, client, "epoch_sweep", best_base, ep,
                       ftm, best_temp, AUTOTUNE_N_REPEATS)
    best3 = _autotune_pick(state, "epoch_sweep")

    final = best3 or best2 or best1
    _autotune_write_outputs(state, final)
    print(f"\n[autotune] FINAL: base={final['base_model']}  epochs={final['epochs']}  "
          f"temp={final['temperature']}  OME_chem={final['OME_chem']:.3f}  "
          f"logR2_chem={final['logR2_chem'] if final['logR2_chem'] is None else round(final['logR2_chem'],3)}")
    _autotune_figure()
    print(f"[autotune] done. outputs in {AUTOTUNE_DIR}")

# ML-BENCHMARK-SPECIFIC CONFIG
OUT_DIR     = os.environ.get("LEO_ML_OUT_DIR", "results/ml_bench")
# ML-owned split directory: ml-bench writes its production fold CSVs here rather
# than into the LLM split tree, so it can never leave the LLM's SPLITS_DIR in a
# half-updated state (CSV rewritten without matching JSONL/meta). The fold contents
# are identical (same deterministic builder), just kept in a separate location.
SPLITS_ROOT = os.path.join(OUT_DIR, "splits")
FAMILIES    = ["production"]
FEATURIZERS = ["morgan", "rdkit", "polybert"]
MODELS      = "all"
# Four matched-control ablation conditions (no imputation):
#   baseline   = all rows, neither feature
#   control    = rows with BOTH layers & thickness known, neither feature (matched control)
#   layers     = those same rows + layer-count feature
#   thickness  = those same rows + thickness(mm) feature
ABLATION_CONDS = ["baseline", "control", "layers", "thickness"]
TRAIN_FILTERS  = ["all", "misse_only"]

NESTED_TUNE    = True
N_INNER_SPLITS = 4
N_TRIALS       = 30
N_BOOT         = 1000
BOOTSTRAP_MODE = "chemistry"                  # CIs resample unique chemistries and recompute chem_metrics
PRIMARY_METRIC = "OME"                         # selection uses chemistry-averaged variant
RANDOM_STATE   = SEED
SEARCH_JOBS    = -1

USE_EXPOSURE   = True
LOG_EXPOSURE   = True
ORIENTATION_MODE = "auto"
MORGAN_BITS    = 1024
MORGAN_RADIUS  = 2
STAR_HANDLING  = "keep"
LOOP_CLOSE     = True      # loop-close the repeat unit before mol-based featurization
                          # (morgan/rdkit), identical to penn5 -> ML == PENN
DROP_INVALID_PSMILES = False

PG_SRC_PATHS   = ["/data/wschertzer3/PolymerGenome/src/fingerprinting",
                  "/data/wschertzer3/PolymerGenome/src/common_lib"]
PG_FP_VERSION  = 2
POLYBERT_PATH  = os.environ.get("LEO_POLYBERT_PATH", "kuelumbus/polyBERT")  # public HF checkpoint; set env var to a local cache to skip download
POLYBERT_DEVICE = "cpu"
POLYBERT_BATCH = 32

AGG_MODE = {"production": "pooled"}
AGG_DEFAULT_TEST_SIZE_CUTOFF = 10
FORCE = False
ACCEPT_LEGACY = False


def _loop_mol(s):
    """Loop-close the repeat unit into a closed monomer ring, matching
    penn5._smiles_to_loop_mol so the mol-based fingerprints (morgan/rdkit) agree
    with the PENN benchmark. PE ('[*]C[*]') is rewritten to '[*]CCC[*]' so its
    endpoints aren't adjacent (a 2-atom backbone can't ring-close); if the
    endpoints are already bonded (e.g. vinyl monomers) the bond isn't re-added.
    Returns None on failure rather than raising, so the caller can emit a NaN
    feature row for that pSMILES."""
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

def _mols(ps, cap):
    """Mols for the mol-based featurizers. With LOOP_CLOSE the repeat unit is
    ring-closed (penn5-identical); otherwise legacy behavior (keep stars as
    dummies, or H-cap). `cap` retained for signature compatibility."""
    if LOOP_CLOSE:
        return [_loop_mol(s) for s in ps]
    return [Chem.MolFromSmiles(s.replace("[*]", "[H]") if cap else s) for s in ps]

def feat_morgan(ps):
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    X = np.full((len(ps), MORGAN_BITS), np.nan, np.float32)   # nan row flags invalid mol
    for i, m in enumerate(_mols(ps, STAR_HANDLING == "to_h")):
        if m is not None: X[i] = gen.GetFingerprintAsNumPy(m)
    return X, [f"mfp_{i}" for i in range(MORGAN_BITS)]

def feat_rdkit(ps):
    from rdkit.Chem import Descriptors
    names = [n for n, _ in Descriptors.descList]                # FIXED order, always
    rows = []
    for m in _mols(ps, STAR_HANDLING == "to_h"):
        if m is None: rows.append([np.nan]*len(names)); continue
        r = []
        for _, fn in Descriptors.descList:
            try: v = fn(m); r.append(v if np.isfinite(v) else np.nan)
            except Exception: r.append(np.nan)
        rows.append(r)
    return np.array(rows, np.float32), names

_PG = {}
def feat_pg(ps):
    """PolymerGenome hierarchical fingerprint (v2). Calls the local PG code directly,
    same as any other featurizer; fixed-schema vector per pSMILES (NaN row if PG fails)."""
    import importlib
    if "fp" not in _PG:
        for p in PG_SRC_PATHS:
            if p not in sys.path: sys.path.append(p)
        _PG["fp"] = importlib.import_module("fp")
    fpmod, params = _PG["fp"], {"ismolecule": False, "polymer_fp_version": PG_FP_VERSION}
    rows = []
    for s in ps:
        try: d = fpmod.fingerprint_from_smiles(s, params)
        except Exception: d = None
        rows.append(d if isinstance(d, dict) else {})
    keys = _PG.get("keys")
    if keys is None:
        keys = sorted({k for d in rows for k in d}); _PG["keys"] = keys
    X = np.array([[d.get(k, np.nan) for k in keys] for d in rows], np.float32)
    return X, [f"pg_{k}" for k in keys]

_PB = {}
def feat_polybert(ps):
    """polyBERT embeddings: tokenize raw pSMILES (keep [*]), mean-pool token states -> 600-dim.
    Matches the standard polyBERT/sentence-transformers pooling; works on the bare HF checkpoint."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    if "model" not in _PB:
        _PB["tok"] = AutoTokenizer.from_pretrained(POLYBERT_PATH)
        _PB["model"] = AutoModel.from_pretrained(POLYBERT_PATH).to(POLYBERT_DEVICE).eval()
    tok, model = _PB["tok"], _PB["model"]
    out = []
    for i in range(0, len(ps), POLYBERT_BATCH):
        batch = list(ps[i:i + POLYBERT_BATCH])              # raw pSMILES, stars intact
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt").to(POLYBERT_DEVICE)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(emb.cpu().numpy())
    X = np.vstack(out).astype(np.float32)
    return X, [f"pb_{i}" for i in range(X.shape[1])]

FEAT = {"morgan": feat_morgan, "rdkit": feat_rdkit,
        "pg": feat_pg, "polybert": feat_polybert}

def _pg_source_fp():
    return "pg" + "".join(PG_SRC_PATHS) + f"v{PG_FP_VERSION}"
def feat_cache_key(featname):
    extra = ""
    if featname == "pg": extra = "|" + _pg_source_fp()
    elif featname == "polybert": extra = "|pb:" + POLYBERT_PATH
    elif featname == "rdkit": extra = "|rdkit" + Chem.rdBase.rdkitVersion
    return hashlib.md5(f"{featname}|{STAR_HANDLING}|loop{int(LOOP_CLOSE)}|{MORGAN_BITS}|{MORGAN_RADIUS}{extra}".encode()).hexdigest()[:8]

def chem_features(featname, ps, cache_dir):
    fp = os.path.join(cache_dir, f"feat_{featname}_{feat_cache_key(featname)}.pkl")
    cache = pickle.load(open(fp, "rb")) if (os.path.exists(fp) and not FORCE) else {"vec": {}, "cols": None}
    if featname == "pg" and "keys" not in _PG and (cache.get("pg_keys") is not None or cache["cols"] is not None):
        # resume: pin PG schema from persisted raw keys (or, for legacy caches w/o pg_keys, recover from cols)
        _PG["keys"] = list(cache["pg_keys"]) if cache.get("pg_keys") is not None else [c[3:] for c in cache["cols"]]
    uniq = list(dict.fromkeys(ps)); miss = [s for s in uniq if s not in cache["vec"]]
    if miss or cache["cols"] is None:
        Xm, cols = FEAT[featname](miss if cache["cols"] is not None else uniq)
        base = miss if cache["cols"] is not None else uniq
        if cache["cols"] is not None and cols != cache["cols"]:
            raise RuntimeError(f"{featname} schema drift: cached {len(cache['cols'])} vs new {len(cols)} cols")
        for s, v in zip(base, Xm): cache["vec"][s] = v
        cache["cols"] = cols
        if featname == "pg": cache["pg_keys"] = _PG.get("keys")   # persist raw PG keys for robust resume
        os.makedirs(cache_dir, exist_ok=True); pickle.dump(cache, open(fp, "wb"))
    return cache["cols"]

_VEC_CACHE = {}
def _get_vec(featname, cache_dir, ps):
    fp = os.path.join(cache_dir, f"feat_{featname}_{feat_cache_key(featname)}.pkl")
    mtime = os.path.getmtime(fp)
    hit = _VEC_CACHE.get(fp)
    if hit is None or hit[0] != mtime:                      # reload only when the pickle is new / changed
        hit = (mtime, pickle.load(open(fp, "rb"))); _VEC_CACHE[fp] = hit
    return np.vstack([hit[1]["vec"][s] for s in ps])

# exposure / orientation
def numeric_exposure(df):
    cols, parts = [], []
    for c, name in [("mission time (yr)", "mission_time"),
                    ("solar exposure (esh)", "esh"),
                    ("ao fluence (atoms/cm2)", "fluence")]:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        bad = ~np.isfinite(v)
        if bad.any():
            raise ValueError(f"{c} non-finite at rows {np.where(bad)[0].tolist()}")
        if LOG_EXPOSURE:
            if (v <= 0).any():
                raise ValueError(f"non-positive {c} can't be log10'd: rows {np.where(v<=0)[0].tolist()}")
            v = np.log10(v); name = "log_" + name
        parts.append(v.reshape(-1, 1)); cols.append(name)
    return np.hstack(parts), cols

ORIENT_LEVELS = ["ram", "zenith", "nadir", "wake", "unknown"]
def _norm_orient(df):
    s = df["orientation"].astype("string").str.strip().str.lower()
    return s.where(s.isin(ORIENT_LEVELS), "unknown").fillna("unknown").values
def orient_onehot(df):
    o = _norm_orient(df)
    oh = np.zeros((len(df), len(ORIENT_LEVELS)), float)
    for i, v in enumerate(o): oh[i, ORIENT_LEVELS.index(v)] = 1
    return oh, [f"orient_{l}" for l in ORIENT_LEVELS]

def layers_num(df): return pd.to_numeric(df[LAYERS_COL], errors="coerce").values
def thickness_num(df): return pd.to_numeric(df[THICKNESS_COL], errors="coerce").values



# assemble: raw fixed-schema features for the 4 matched-control conditions
def assemble(df, cond, featname, cache_dir, orient_native):
    """cond in {baseline, control, layers, thickness}. baseline -> all rows, no
    layers/thickness feature. control/layers/thickness -> rows with BOTH layers and
    thickness known (one shared matched row-set); control adds neither feature,
    layers adds the layer count, thickness adds thickness(mm). Test rows are
    guaranteed known by eligibility, so this mask never drops a test point."""
    mask = np.ones(len(df), bool)
    if cond in ("control", "layers", "thickness"):
        mask &= ~np.isnan(layers_num(df))
        mask &= ~np.isnan(thickness_num(df))
    d = df[mask]
    chem = _get_vec(featname, cache_dir, d[PSMILES_COL].tolist())
    blocks, names = [chem], [f"f{i}" for i in range(chem.shape[1])]
    if USE_EXPOSURE:
        ne, en = numeric_exposure(d); blocks.append(ne); names += en
        if not orient_native:
            oh, on = orient_onehot(d); blocks.append(oh); names += on
    has_layers = cond == "layers"
    if has_layers:
        blocks.append(layers_num(d).reshape(-1, 1)); names.append("layers")
    if cond == "thickness":
        blocks.append(thickness_num(d).reshape(-1, 1)); names.append("thickness_mm")
    Xnum = np.hstack(blocks)
    if orient_native and USE_EXPOSURE:
        X = pd.DataFrame(Xnum, columns=names)
        X["orientation"] = pd.Categorical(_norm_orient(d), categories=ORIENT_LEVELS)
        return X, has_layers, mask
    return Xnum, has_layers, mask


# clone-safe CatBoost (auto cat detection)
class CatBoostCat(BaseEstimator, RegressorMixin):
    def __init__(self, depth=6, learning_rate=0.1, iterations=300, l2_leaf_reg=3.0,
                 bagging_temperature=1.0, random_strength=1.0, random_state=42):
        self.depth=depth; self.learning_rate=learning_rate; self.iterations=iterations
        self.l2_leaf_reg=l2_leaf_reg; self.bagging_temperature=bagging_temperature
        self.random_strength=random_strength; self.random_state=random_state
    def fit(self, X, y):
        from catboost import CatBoostRegressor
        cats = [c for c in X.columns if str(X[c].dtype) == "category"] if hasattr(X, "columns") else []
        self.model_ = CatBoostRegressor(depth=self.depth, learning_rate=self.learning_rate,
            iterations=self.iterations, l2_leaf_reg=self.l2_leaf_reg,
            bagging_temperature=self.bagging_temperature, random_strength=self.random_strength,
            random_state=self.random_state, verbose=0, allow_writing_files=False,
            cat_features=cats, thread_count=1)
        self.model_.fit(X, y); return self
    def predict(self, X): return self.model_.predict(X)

# registry: (estimator, params, scale, nan_native, cat_native)
def build_models(requested):
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                                  GradientBoostingRegressor, HistGradientBoostingRegressor)
    from sklearn.neural_network import MLPRegressor
    L, U, I, rs = loguniform, uniform, randint, RANDOM_STATE
    R = {}
    R["ridge"]      = (Ridge(), {"alpha": L(1e-3, 1e3)}, True, False, False)
    R["lasso"]      = (Lasso(max_iter=50000), {"alpha": L(1e-4, 1e1)}, True, False, False)
    R["elasticnet"] = (ElasticNet(max_iter=50000), {"alpha": L(1e-4, 1e1), "l1_ratio": U(.05, .9)}, True, False, False)
    R["bayesridge"] = (BayesianRidge(), {"alpha_1": L(1e-7, 1e-3), "lambda_1": L(1e-7, 1e-3)}, True, False, False)
    R["ard"]        = (ARDRegression(), {"alpha_1": L(1e-7, 1e-3)}, True, False, False)
    R["pls"]        = (PLSRegression(), {"n_components": I(1, 20)}, True, False, False)
    R["krr"]        = (KernelRidge(kernel="rbf"), {"alpha": L(1e-4, 1e1), "gamma": L(1e-5, 1e1)}, True, False, False)
    R["svr"]        = (SVR(kernel="rbf"), {"C": L(1e-1, 1e3), "gamma": L(1e-5, 1e1), "epsilon": U(.005, .2)}, True, False, False)
    R["knn"]        = (KNeighborsRegressor(), {"n_neighbors": I(2, 20), "weights": ["uniform", "distance"], "p": [1, 2]}, True, False, False)
    R["dtree"]      = (DecisionTreeRegressor(random_state=rs), {"max_depth": I(2, 20), "min_samples_leaf": I(1, 10), "max_features": U(.3, .7), "ccp_alpha": L(1e-4, 1e-1)}, False, False, False)
    ts = {"n_estimators": I(100, 600), "max_depth": I(2, 20), "max_features": U(.3, .7), "min_samples_leaf": I(1, 8), "min_samples_split": I(2, 10)}
    R["rf"]         = (RandomForestRegressor(random_state=rs, n_jobs=1), ts, False, False, False)
    R["extratrees"] = (ExtraTreesRegressor(random_state=rs, n_jobs=1), ts, False, False, False)
    R["gbm"]        = (GradientBoostingRegressor(random_state=rs),
                       {"n_estimators": I(100, 500), "learning_rate": L(1e-2, 3e-1), "max_depth": I(2, 5),
                        "subsample": U(.6, .4), "min_samples_leaf": I(1, 8), "max_features": U(.3, .7)}, False, False, False)
    R["hgb"]        = (HistGradientBoostingRegressor(random_state=rs, categorical_features="from_dtype"),
                       {"learning_rate": L(1e-2, 3e-1), "max_iter": I(100, 500), "max_leaf_nodes": I(7, 63),
                        "l2_regularization": L(1e-6, 1e1), "min_samples_leaf": I(5, 30)},
                       False, True, True)
    R["mlp"]        = (MLPRegressor(random_state=rs, max_iter=1500, early_stopping=False),
                       {"hidden_layer_sizes": [(64,), (64, 64), (128, 64), (64, 64, 64)], "alpha": L(1e-5, 1e1),
                        "learning_rate_init": L(1e-4, 1e-2), "activation": ["relu", "tanh"]},
                       True, False, False)
    # optional libs: import lazily, skip the model (not the whole registry) if absent
    want = (lambda n: requested == "all" or n in requested)
    if want("xgb"):
        try:
            from xgboost import XGBRegressor
            R["xgb"] = (XGBRegressor(random_state=rs, n_jobs=1, verbosity=0, tree_method="hist", enable_categorical=True),
                        {"n_estimators": I(100, 600), "learning_rate": L(1e-2, 3e-1), "max_depth": I(2, 8),
                         "subsample": U(.6, .4), "colsample_bytree": U(.6, .4), "reg_lambda": L(1e-2, 1e2),
                         "reg_alpha": L(1e-3, 1e1), "min_child_weight": I(1, 10)}, False, True, True)
        except Exception as e: print(f"[skip xgb] {e}")
    if want("lgbm"):
        try:
            from lightgbm import LGBMRegressor
            R["lgbm"] = (LGBMRegressor(random_state=rs, n_jobs=1, verbose=-1, subsample_freq=1),
                         {"n_estimators": I(100, 600), "learning_rate": L(1e-2, 3e-1), "num_leaves": I(7, 63),
                          "subsample": U(.6, .4), "colsample_bytree": U(.6, .4), "reg_lambda": L(1e-2, 1e2),
                          "reg_alpha": L(1e-3, 1e1), "min_child_samples": I(2, 40)}, False, True, True)
        except Exception as e: print(f"[skip lgbm] {e}")
    if want("catboost"):
        try:
            import catboost  # noqa
            R["catboost"] = (CatBoostCat(random_state=rs),
                             {"depth": I(3, 8), "learning_rate": L(1e-2, 3e-1), "iterations": I(100, 600),
                              "l2_leaf_reg": L(1e-1, 1e2), "bagging_temperature": U(0.0, 1.0),
                              "random_strength": L(1e-2, 1e1)}, False, True, True)
        except Exception as e: print(f"[skip catboost] {e}")
    return R

# train-side curation
def apply_train_filter(tr, flt):
    if flt == "all": return tr
    if flt == "misse_only": return tr[_misse_mask(tr)]
    raise ValueError(flt)



# one cell: nested-tuned fit (or frozen-param fit) on the given splits
def _jsonify_params(d):
    out = {}
    for k, v in (d or {}).items():
        out[k] = v.item() if hasattr(v, "item") else v
    return out

def run_cell(est, pdist, scale, nan_native, cat_native, feat, cond, flt, splits,
             cache_dir, agg, frozen=None, collect_preds=False):
    """Fit/eval one (model, featurizer, condition, filter) cell across `splits`.
    If `frozen` (a {param: value} dict) is given, the hyperparameter search is
    SKIPPED and those exact params are used on every split -- this is how the
    5-fold production reuses the params tuned on the single ablation split. When
    searching, the chosen params are captured and returned in out["best_params"]
    so they can be frozen. With collect_preds=True the per-fold true/pred/canon
    are returned in out["_pred_rows"] (popped before any JSON save)."""
    orient_native = cat_native and ORIENTATION_MODE != "onehot"
    per_split = []   # (yte, pred, canon_te)
    captured = dict(frozen) if frozen is not None else None
    for tr_df, te_df in splits:
        tr_df = apply_train_filter(tr_df, flt)
        if len(tr_df) == 0:
            raise ValueError("empty training set after filter")
        Xtr, has_lay, mtr = assemble(tr_df, cond, feat, cache_dir, orient_native)
        Xte, _, mte = assemble(te_df, cond, feat, cache_dir, orient_native)
        ytr = tr_df[TARGET_COL].values[mtr].astype(float)
        yte = te_df[TARGET_COL].values[mte].astype(float)
        canon_te = te_df[PSMILES_COL].apply(canon).values[mte]
        groups = tr_df[PSMILES_COL].apply(canon).values[mtr]
        ntr = len(ytr)
        steps = []
        if not nan_native:
            steps += [("impute", SimpleImputer(strategy="mean")), ("var", VarianceThreshold(0.0))]
            if scale: steps.append(("scale", StandardScaler()))
        steps.append(("m", clone(est)))
        pipe = Pipeline(steps)
        p2 = {f"m__{k}": v for k, v in pdist.items()}
        if frozen is not None:
            # Frozen production fit: no search, use the params tuned on the ablation split.
            best = pipe.set_params(**{f"m__{k}": v for k, v in frozen.items()}).fit(Xtr, ytr)
        elif NESTED_TUNE:
            ng = min(N_INNER_SPLITS, len(np.unique(groups)))
            if ng < 2:
                best = pipe.fit(Xtr, ytr)
                if captured is None:
                    captured = {k: best.get_params().get(f"m__{k}") for k in pdist}
            else:
                # Manual GroupKFold random search scored by chemistry-weighted OME
                # (sklearn scorers can't see group labels, so we tune the same metric
                # we report). Deterministic sampling; trials evaluated in parallel.
                from joblib import Parallel, delayed
                cv_splits = list(GroupKFold(ng).split(Xtr, ytr, groups))
                min_inner = min(len(tr_idx) for tr_idx, _ in cv_splits)
                hi_nn = min(20, min_inner)
                hi_nc = min(20, min_inner - 1, Xtr.shape[1])
                if "n_neighbors" in pdist and hi_nn < 2:
                    raise ValueError(f"inner fold too small for KNN (min_inner={min_inner})")
                if "n_components" in pdist and hi_nc < 1:
                    raise ValueError(f"inner fold too small for PLS (min_inner={min_inner})")
                srng = np.random.RandomState(RANDOM_STATE)
                def _sample():
                    d = {}
                    for k, dist in pdist.items():
                        if k == "n_neighbors":     d[k] = int(srng.randint(2, hi_nn + 1))
                        elif k == "n_components":  d[k] = int(srng.randint(1, hi_nc + 1))
                        elif hasattr(dist, "rvs"): d[k] = dist.rvs(random_state=srng)
                        else:                      d[k] = dist[srng.randint(len(dist))]
                    return d
                trials = [_sample() for _ in range(N_TRIALS)]
                def _score(params):
                    fold_omes = []
                    for tr_idx, va_idx in cv_splits:
                        try:
                            pl = clone(pipe).set_params(**{f"m__{k}": v for k, v in params.items()})
                            pl.fit(Xtr[tr_idx], ytr[tr_idx])
                            pr = np.asarray(pl.predict(Xtr[va_idx])).ravel()
                            o, _, _ = chem_metrics(ytr[va_idx], pr, groups[va_idx])
                            fold_omes.append(o if o is not None else np.inf)
                        except Exception:
                            fold_omes.append(np.inf)
                    return float(np.mean(fold_omes))
                scores = Parallel(n_jobs=SEARCH_JOBS)(delayed(_score)(t) for t in trials)
                bp = trials[int(np.argmin(scores))]
                best = clone(pipe).set_params(**{f"m__{k}": v for k, v in bp.items()}).fit(Xtr, ytr)
                if captured is None:
                    captured = bp
        else:
            if "m__n_neighbors" in p2: p2["m__n_neighbors"] = randint(2, max(3, min(20, ntr - 1)))
            if "m__n_components" in p2: p2["m__n_components"] = randint(1, max(2, min(20, ntr - 1, Xtr.shape[1])))
            chosen = {k: (v.rvs(random_state=RANDOM_STATE) if hasattr(v, "rvs") else v[0]) for k, v in p2.items()}
            best = pipe.set_params(**chosen).fit(Xtr, ytr)
            if captured is None:
                captured = {k.replace("m__", ""): v for k, v in chosen.items()}
        per_split.append((yte, np.asarray(best.predict(Xte)).ravel(), canon_te))

    y  = np.concatenate([a for a, _, _ in per_split])
    yh = np.concatenate([b for _, b, _ in per_split])
    cc = np.concatenate([c for _, _, c in per_split])
    ome_c, r2_c, n_c = chem_metrics(y, yh, cc)
    out = dict(OME=float(calculate_ome(y, yh)), logR2=float(calculate_log_r2(y, yh)),
               OME_chem=ome_c, logR2_chem=r2_c, n=int(len(y)), n_chem=n_c)
    if agg == "pooled":
        # Chemistry-resampled bootstrap: resample the UNIQUE chemistries with
        # replacement and recompute chem_metrics each draw, so the CIs match the
        # reported chem-weighted metrics (and the single-split case is non-degenerate).
        rng = np.random.default_rng(RANDOM_STATE)
        by = {}
        for k, ci_ in enumerate(cc):
            by.setdefault(ci_, []).append(k)
        chems = list(by); omb, r2b = [], []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(chems), len(chems))
            yb, yhb, lab = [], [], []
            for m, j in enumerate(pick):                 # unique label per draw preserves multiplicity
                rows = by[chems[j]]
                yb.append(y[rows]); yhb.append(yh[rows])
                lab.append(np.full(len(rows), f"{chems[j]}__{m}"))
            o_, r_, _ = chem_metrics(np.concatenate(yb), np.concatenate(yhb), np.concatenate(lab))
            if o_ is not None: omb.append(o_)
            if r_ is not None: r2b.append(r_)
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a else [float("nan")] * 2
        out.update(agg="pooled", bootstrap="chemistry",
                   OME_chem_ci=ci(omb), logR2_chem_ci=ci(r2b))
    else:
        r2s  = [calculate_log_r2(a, b) for a, b, _ in per_split if len(np.unique(a)) > 1]
        omes = [calculate_ome(a, b) for a, b, _ in per_split]
        out.update(agg="per_split", n_splits=len(per_split), n_r2_splits=len(r2s),
                   logR2=float(np.mean(r2s)) if r2s else float("nan"),
                   logR2_std=float(np.std(r2s, ddof=1)) if len(r2s) > 1 else float("nan"),
                   OME=float(np.mean(omes)),
                   OME_std=float(np.std(omes, ddof=1)) if len(omes) > 1 else float("nan"))
    out["best_params"] = _jsonify_params(captured)
    if collect_preds:
        rows = []
        for si, (a, b, c) in enumerate(per_split):
            for tv, pv, cv in zip(a, b, c):
                rows.append({"split": si, "true": float(tv), "pred": float(pv), "canon": str(cv)})
        out["_pred_rows"] = rows
    return out


def split_fingerprint():
    h = hashlib.md5()
    for fam in FAMILIES:
        for d in sorted(glob.glob(os.path.join(SPLITS_ROOT, fam, "split_*"))):
            for f in ("train.csv", "test.csv"):
                p = os.path.join(d, f)
                if os.path.exists(p): h.update(open(p, "rb").read())
    return h.hexdigest()[:10]

def _dist_repr(v):
    """Process-stable serialization of a scipy frozen dist or a list (no memory address)."""
    if hasattr(v, "dist") and hasattr(v, "args"):           # scipy frozen distribution
        return f"{v.dist.name}|args={v.args}|kwds={sorted(v.kwds.items())}"
    return repr(v)                                           # plain list e.g. ['relu','tanh']

def search_space_fingerprint(reg):
    # hash both the estimator construction (v[0] repr -> catches e.g. early_stopping/depth defaults) and the search space
    blob = {k: {"est": repr(v[0]), "space": sorted({pk: _dist_repr(pv) for pk, pv in v[1].items()}.items())}
            for k, v in reg.items()}
    return hashlib.md5(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:10]

def code_fingerprint():
    """Hash the source of the core logic fns so code edits (not just config) bust the cache. Works in notebooks via inspect."""
    import inspect
    try:
        src = "".join(inspect.getsource(f) for f in
                      (assemble, run_cell, build_models, numeric_exposure,
                       feat_morgan, feat_rdkit, feat_pg, feat_polybert))
    except Exception:
        return "nosrc"
    return hashlib.md5(src.encode()).hexdigest()[:10]

def cfg_hash(reg):
    k = dict(nest=NESTED_TUNE, nin=N_INNER_SPLITS, nt=N_TRIALS, nb=N_BOOT, exp=USE_EXPOSURE,
             loge=LOG_EXPOSURE, om=ORIENTATION_MODE, star=STAR_HANDLING, mb=MORGAN_BITS, mr=MORGAN_RADIUS,
             rs=RANDOM_STATE, tgt=TARGET_COL, agg=str(sorted(AGG_MODE.items())),
             misse=MISSE_MATCH, maxtc=MAX_TEST_COUNT, aggcut=AGG_DEFAULT_TEST_SIZE_CUTOFF, dropinv=DROP_INVALID_PSMILES,
             boot=BOOTSTRAP_MODE, splits=split_fingerprint(), space=search_space_fingerprint(reg),
             code=code_fingerprint(), conds=tuple(ABLATION_CONDS), feat={f: feat_cache_key(f) for f in FEATURIZERS})
    return hashlib.md5(json.dumps(k, sort_keys=True).encode()).hexdigest()[:10]

def versions():
    import sklearn, scipy
    v = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
         "sklearn": sklearn.__version__, "scipy": scipy.__version__, "rdkit": Chem.rdBase.rdkitVersion}
    for m in ("xgboost", "lightgbm", "catboost"):
        try: v[m] = __import__(m).__version__
        except Exception: v[m] = None
    return v

def save_json(obj, path):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=2); os.replace(tmp, path)


def drop_invalid_rows(df):
    valid = df[PSMILES_COL].apply(lambda s: isinstance(s, str) and Chem.MolFromSmiles(s) is not None)
    if not valid.all(): print(f"  [warn] dropping {(~valid).sum()} invalid-pSMILES rows")
    return df.loc[valid].reset_index(drop=True)

def validate(splits, family):
    """Validate columns/targets/exposures; either drop invalid pSMILES rows or raise. Returns cleaned splits."""
    req = [PSMILES_COL, TARGET_COL, LAYERS_COL, MISSION_COL, "orientation",
           "mission time (yr)", "solar exposure (esh)", "ao fluence (atoms/cm2)"]
    req = req + [THICKNESS_COL]
    cleaned = []
    for tr, te in splits:
        for nm, d in [("train", tr), ("test", te)]:
            miss = [c for c in req if c not in d.columns]
            if miss: raise ValueError(f"{nm} missing columns {miss}")
            if not np.isfinite(pd.to_numeric(d[TARGET_COL], errors="coerce")).all():
                raise ValueError(f"{nm} has non-finite {TARGET_COL}")
        if DROP_INVALID_PSMILES:
            tr, te = drop_invalid_rows(tr), drop_invalid_rows(te)
        else:
            bad = [s for s in set(tr[PSMILES_COL]) | set(te[PSMILES_COL])
                   if not (isinstance(s, str) and Chem.MolFromSmiles(s))]
            if bad:
                raise ValueError(f"{len(bad)} invalid pSMILES e.g. {bad[:5]} "
                                 f"(set DROP_INVALID_PSMILES=True to drop)")
        ov = set(tr[PSMILES_COL].map(canon)) & set(te[PSMILES_COL].map(canon))
        if ov:
            if family in ("ablation", "production"):
                raise ValueError(f"[{family}] {len(ov)} train/test chemistry overlap e.g. {list(ov)[:5]} "
                                 f"(group-disjoint benchmark must have none)")
            print(f"  [warn] {len(ov)} chemistry overlap train/test in a split")
        cleaned.append((tr, te))
    return cleaned


def requested_models():
    if MODELS == "all": return "all"
    return [MODELS] if isinstance(MODELS, str) else list(MODELS)

def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False

def _code_dir():
    """Directory of this script (falls back to cwd when __file__ is undefined, e.g. in a notebook)."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

# shared, deterministic split provisioning (both scripts arrive at the same split)
def _load_canon_dataset():
    df = pd.read_csv(DATASET_CSV)
    df["_canon"] = df[COL_PSMILES].apply(canonicalize_psmiles)
    bad = int(df["_canon"].isna().sum())
    if bad:
        print(f"[split] dropping {bad} rows that failed canonicalization")
        df = df.dropna(subset=["_canon"]).reset_index(drop=True)
    df[COL_PSMILES] = df["_canon"]
    return df

def ensure_production_on_disk(force=False):
    """Chemistry-grouped 5-fold CV over the WHOLE dataset (deterministic, no RNG),
    identical to what the LLM production consumes. Writes fold CSVs to the ML-owned
    SPLITS_ROOT/production/split_NN (not the LLM split tree) and returns the splits."""
    df = _load_canon_dataset()
    splits = build_grouped_cv_splits(df, "_canon")
    for i, (tr, te) in enumerate(splits, start=1):
        d = os.path.join(SPLITS_ROOT, "production", f"split_{i:02d}")
        os.makedirs(d, exist_ok=True)
        tr.drop(columns=["_canon"]).to_csv(os.path.join(d, "train.csv"), index=False)
        te.drop(columns=["_canon"]).to_csv(os.path.join(d, "test.csv"), index=False)
    return [(tr.drop(columns=["_canon"]), te.drop(columns=["_canon"])) for tr, te in splits]

def feature_ablation_single():
    """The single ablation split: all eligible rows in test, everything else in train."""
    df = _load_canon_dataset()
    tr, te = build_feature_ablation(df, "_canon")
    return tr.drop(columns=["_canon"]), te.drop(columns=["_canon"])


# ml-bench driver: ablation on the single split, then 5-fold production on best
def mode_ml_bench():
    if not _ML_DEPS:
        print("ml-bench requires scikit-learn / scipy / rdkit, which are not installed.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_dir = os.path.join(OUT_DIR, "cache"); os.makedirs(cache_dir, exist_ok=True)
    req = requested_models(); reg = build_models(req)
    names = list(reg) if req == "all" else [m for m in req if m in reg]

    # Deterministic shared splits (build either-first):
    tr_ab, te_ab = feature_ablation_single()
    ablation_splits = [(tr_ab, te_ab)]
    ablation_splits = validate(ablation_splits, "ablation")   # train/test chemistry overlap must be empty
    prod_splits = ensure_production_on_disk(force=FORCE)
    prod_splits = validate(prod_splits, "production")
    print(f"[ml-bench] ablation single split: test={len(te_ab)} rows | "
          f"production grouped-CV folds={len(prod_splits)} (pooled out-of-fold test = full dataset)")

    res_path = os.path.join(OUT_DIR, f"ml_results_{cfg_hash(reg)}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}

    allps = pd.concat([pd.concat([t, e]) for t, e in (ablation_splits + prod_splits)])[PSMILES_COL].tolist()
    for feat in FEATURIZERS:
        chem_features(feat, allps, cache_dir)

    # ---- Phase 1: ablation grid on the single split ----
    rows = []
    for feat in FEATURIZERS:
        for flt in TRAIN_FILTERS:
            for cond in ABLATION_CONDS:
                for nm in names:
                    est, pd_, sc, nan_n, cat_n = reg[nm]
                    key = f"ablation|{feat}|{cond}|{flt}|{nm}"
                    if key in results and not FORCE:
                        rows.append(results[key]); continue
                    t0 = time.time()
                    try:
                        r = run_cell(est, pd_, sc, nan_n, cat_n, feat, cond, flt, ablation_splits, cache_dir, "pooled")
                        r.update(phase="ablation", featurizer=feat, condition=cond,
                                 train_filter=flt, model=nm, sec=round(time.time() - t0, 1))
                        results[key] = r; save_json(results, res_path); rows.append(r)
                        print(f"  {key:55s} OME_chem={r.get('OME_chem')} logR2_chem={r.get('logR2_chem')} ({r['sec']}s)")
                    except Exception as e:
                        results.setdefault("_failures", {})[key] = {"error": str(e),
                                                                    "traceback": traceback.format_exc()[-1200:]}
                        save_json(results, res_path)
                        print(f"  {key:55s} FAIL: {e}")
    if not rows:
        print("[ml-bench] no ablation cells succeeded."); return

    abdf = pd.DataFrame(rows)
    # best -> worst by chemistry-averaged error (OME_chem asc), tie-break logR2_chem desc
    abdf = abdf.sort_values(by=["OME_chem", "logR2_chem"], ascending=[True, False], na_position="last")
    cols = ["phase", "featurizer", "condition", "train_filter", "model",
            "OME_chem", "logR2_chem", "OME", "logR2", "n", "n_chem", "sec"]
    ranked = os.path.join(OUT_DIR, "ml_all_combinations_ranked.csv")
    abdf[[c for c in cols if c in abdf.columns]].round(4).to_csv(ranked, index=False)
    print(f"\n[ml-bench] wrote {len(abdf)} ablation cells (best->worst by OME_chem) -> {ranked}")
    print(abdf[[c for c in cols if c in abdf.columns]].head(12).round(3).to_string(index=False))

    best = abdf.iloc[0]
    print(f"\n[ml-bench] best ablation: feat={best.featurizer} cond={best.condition} "
          f"filter={best.train_filter} model={best.model} OME_chem={best.OME_chem:.3f}")

    # ---- Phase 2: frozen 5-fold production for the baseline-all list ----
    # Mirror the LLM procedure: the per-model winners from the SINGLE ablation split
    # (condition=baseline, train_filter=all, best featurizer per model) are taken with
    # their tuned hyperparameters FROZEN, then trained/tested on the chemistry-grouped
    # 5-fold CV folds (the identical on-disk folds the LLM production used) with NO
    # re-tuning. Per-row predictions are saved.
    base = abdf[(abdf["condition"] == "baseline") & (abdf["train_filter"] == "all")].copy()
    base = base.sort_values(by=["OME_chem", "logR2_chem"], ascending=[True, False], na_position="last")
    base = base.drop_duplicates(subset=["model"], keep="first")   # best featurizer per model
    agg = "pooled"
    pred_dir = os.path.join(OUT_DIR, "production_predictions"); os.makedirs(pred_dir, exist_ok=True)
    frozen_hparams, prod_rows = {}, []
    for _, brow in base.iterrows():
        nm, feat = brow["model"], brow["featurizer"]
        akey = f"ablation|{feat}|baseline|all|{nm}"
        hp = results.get(akey, {}).get("best_params", {})
        frozen_hparams[nm] = {"featurizer": feat, "hparams": hp}
        est, pd_, sc, nan_n, cat_n = reg[nm]
        t0 = time.time()
        try:
            r = run_cell(est, pd_, sc, nan_n, cat_n, feat, "baseline", "all",
                         prod_splits, cache_dir, agg, frozen=hp, collect_preds=True)
        except Exception as e:
            print(f"  [prod] {nm:14s} FAIL: {e}")
            continue
        preds = r.pop("_pred_rows", [])
        pd.DataFrame(preds).to_csv(os.path.join(pred_dir, f"{nm}_{feat}.csv"), index=False)
        oc = r.get("OME_chem_ci", [float("nan")] * 2); rc = r.get("logR2_chem_ci", [float("nan")] * 2)
        r["OME_chem_lo"], r["OME_chem_hi"] = oc[0], oc[1]
        r["logR2_chem_lo"], r["logR2_chem_hi"] = rc[0], rc[1]
        r.update(phase="production_cv", featurizer=feat, condition="baseline",
                 train_filter="all", model=nm, sec=round(time.time() - t0, 1))
        results[f"production_cv|{nm}"] = r
        prod_rows.append(r)
        print(f"  [prod] {nm:14s} ({feat}) OME_chem={r.get('OME_chem')} "
              f"logR2_chem={r.get('logR2_chem')} ({r['sec']}s)")
    save_json(results, res_path)
    save_json(frozen_hparams, os.path.join(OUT_DIR, "ml_frozen_hparams.json"))

    if prod_rows:
        pdf = pd.DataFrame(prod_rows).sort_values(by=["OME_chem", "logR2_chem"],
                                                  ascending=[True, False], na_position="last")
        pcols = ["model", "featurizer", "OME_chem", "OME_chem_lo", "OME_chem_hi",
                 "logR2_chem", "logR2_chem_lo", "logR2_chem_hi", "n", "n_chem", "sec"]
        prod_csv = os.path.join(OUT_DIR, "ml_production_ranking.csv")
        pdf[[c for c in pcols if c in pdf.columns]].round(4).to_csv(prod_csv, index=False)
        print(f"\n[ml-bench] PRODUCTION (frozen, 5-fold) for {len(pdf)} baseline models -> {prod_csv}")
        print(f"[ml-bench] per-row predictions -> {pred_dir}")
        print(f"[ml-bench] frozen hyperparameters -> {os.path.join(OUT_DIR, 'ml_frozen_hparams.json')}")
    print(f"[ml-bench] full results -> {res_path}")

    # Reproducibility metadata: software versions + identifying hashes.
    try:
        with open(os.path.abspath(__file__), "rb") as _f:
            code_hash = hashlib.md5(_f.read()).hexdigest()[:12]
    except Exception:
        code_hash = None
    meta = {"versions": versions(), "code_md5": code_hash, "cfg_hash": cfg_hash(reg),
            "dataset_csv": DATASET_CSV, "n_splits": N_SPLITS,
            "split_protocol": "chemistry-grouped 5-fold CV (deterministic, largest-first balance)",
            "bootstrap": "chemistry-resampled", "n_boot": N_BOOT}
    try:
        meta["dataset_md5"] = hashlib.md5(open(DATASET_CSV, "rb").read()).hexdigest()[:12]
    except Exception:
        meta["dataset_md5"] = None
    meta_path = os.path.join(OUT_DIR, "ml_run_metadata.json")
    save_json(meta, meta_path)
    print(f"[ml-bench] run metadata -> {meta_path}")


# UNIFIED CLI
def main():
    in_notebook = False
    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except Exception:
        in_notebook = False

    p = argparse.ArgumentParser(description="Unified LEO LLM + ML benchmark pipeline.",
                                allow_abbrev=False)
    p.add_argument("mode", choices=["prep", "train", "retrieve", "infer", "figures",
                                    "autotune", "all", "ml-bench"])
    p.add_argument("--split-type", choices=["feature_ablation", "autotune", "production"],
                   default="production")
    p.add_argument("--force", action="store_true", help="rebuild splits / recompute cached cells")
    p.add_argument("--accept-legacy", action="store_true",
                   help="trust fine-tuned models / results that predate provenance fingerprinting")
    p.add_argument("--yes", "-y", action="store_true",
                   help="skip the confirmation prompt (for non-interactive / tmux runs)")

    argv = None
    if in_notebook or "ipykernel" in sys.modules:
        argv = ["all", "--split-type", "production"]
    a, _unknown = p.parse_known_args(argv)   # tolerate Jupyter's -f kernel arg

    global FORCE, ACCEPT_LEGACY
    FORCE = FORCE or a.force
    ACCEPT_LEGACY = a.accept_legacy

    # Modes that submit fine-tune jobs and/or run paid inference. Gate them behind
    # an explicit y/n so an accidental %run / import in a notebook (which forces
    # mode="all") can't silently start spending money. --yes bypasses for tmux.
    SPENDS_MONEY = {"train", "retrieve", "infer", "autotune", "all"}
    if a.mode in SPENDS_MONEY and not a.yes:
        msg = (f"About to run '{a.mode}' (split-type={a.split_type}). "
               f"This submits fine-tune jobs and/or runs paid inference. Proceed? [y/N]: ")
        try:
            ans = input(msg).strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted (no API calls made). Re-run with --yes to skip this prompt.")
            return

    if a.mode == "prep":
        mode_prep(a.split_type, force=a.force)
    elif a.mode == "autotune":
        mode_autotune()
    elif a.mode == "train":
        mode_train(a.split_type)
    elif a.mode == "retrieve":
        mode_retrieve(a.split_type)
    elif a.mode == "infer":
        mode_infer(a.split_type)
    elif a.mode == "figures":
        mode_figures(a.split_type)
    elif a.mode == "ml-bench":
        mode_ml_bench()
    elif a.mode == "all":
        mode_prep(a.split_type, force=a.force)
        mode_train(a.split_type)
        mode_retrieve(a.split_type)
        mode_infer(a.split_type)
        mode_figures(a.split_type)


if __name__ == "__main__":
    main()