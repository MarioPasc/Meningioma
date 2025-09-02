#!/usr/bin/env python3
"""
meningioma_stats.py  –  v1.2 (2025-08-04)

v1.2
 • Fix global-mean aggregation bug.
 • Presence heat-map: colour-code by pulse + patient ordering.
 • Axial-spacing plots: unified 8 mm → 1 mm y-axis.
"""

# ..............................  imports  .................................... #
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
from typing import Dict, Generator, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nrrd
from matplotlib.colors import to_rgba
from dataclasses import dataclass, asdict
# ............................................................................ #

# -----------------------------  CONFIGURATION  ------------------------------ #
PULSES: Tuple[str, ...] = ("FLAIR", "T1", "T1SIN", "T2")
PULSE_COLOURS = {                # consistent colour palette
    "FLAIR":  "tab:green",
    "T1":     "tab:blue",
    "T1SIN":  "tab:orange",
    "T2":     "tab:red",
}
FIG_DIR = Path("/home/mpascual/research/tests/mening/figures")
CACHE_CSV = Path("/home/mpascual/research/tests/mening/stats.csv")
YLIM_SPACING = (8, 1)            # (top, bottom) – inverted axis

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger("meningioma_stats")

# -----------------------------  DATA CLASSES  ------------------------------- #
@dataclass(frozen=True, slots=True)
class PulseInstance:
    patient: str
    timepoint: int
    pulse: str
    path: Path
    def exists(self) -> bool: return self.path.is_file()

@dataclass(slots=True)
class PulseStatistic:
    patient: str; timepoint: int; pulse: str
    exists: bool; axial_spacing_mm: float 
    @classmethod
    def from_instance(cls, inst: "PulseInstance") -> "PulseStatistic":
        if inst.exists():
            try:
                hdr = nrrd.read_header(str(inst.path))
                dirs = hdr.get("space directions")
                dz = float(np.linalg.norm(dirs[2])) if dirs is not None else np.nan
                return cls(inst.patient, inst.timepoint, inst.pulse, True, dz)
            except Exception as exc:        # noqa: BLE001
                LOGGER.warning("Failed to load %s: %s", inst.path, exc)
        return cls(inst.patient, inst.timepoint, inst.pulse, False, np.nan)
    def to_dict(self) -> Dict[str, object]: return asdict(self)

# -------------------------  FILE-SYSTEM TRAVERSAL  -------------------------- #
def enumerate_instances(root: Path) -> Generator[PulseInstance, None, None]:
    baseline_dir = root / "baseline" / "RM"
    controls_dir = root / "controls"
    patients_with_controls = {d.name for d in controls_dir.iterdir() if d.is_dir()}
    LOGGER.info("Using %d patients with controls", len(patients_with_controls))

    # baseline
    for pulse in PULSES:
        for pdir in (baseline_dir / pulse).iterdir():
            if pdir.is_dir() and pdir.name in patients_with_controls:
                yield PulseInstance(pdir.name, 0, pulse, pdir / f"{pulse}_{pdir.name}.nrrd")

    # follow-ups
    for patient in patients_with_controls:
        for cdir in (controls_dir / patient).iterdir():
            if cdir.is_dir() and cdir.name.startswith("control"):
                try: tp = int(cdir.name.removeprefix("control"))
                except ValueError: continue
                for pulse in PULSES:
                    files = list(cdir.glob(f"{pulse}_*.nrrd"))
                    yield PulseInstance(patient, tp, pulse,
                                        files[0] if files else cdir / f"{pulse}.nrrd")

# -------------------------  DATA INGESTION / CACHE  ------------------------- #
def collect_metadata(root: Path) -> pd.DataFrame:
    rows = [PulseStatistic.from_instance(inst).to_dict()
            for inst in enumerate_instances(root)]
    df = pd.DataFrame(rows)
    LOGGER.info("Collected %d entries (%d patients)", len(df), df.patient.nunique())
    return df

def load_or_build_stats(root: Path, cache: Path, recompute: bool) -> pd.DataFrame:
    if cache.exists() and not recompute:
        LOGGER.info("Reading cached statistics: %s", cache); return pd.read_csv(cache)
    LOGGER.info("Cache miss → computing...");
    df = collect_metadata(root); df.to_csv(cache, index=False)
    LOGGER.info("Cached statistics → %s", cache); return df

# ----------------------  DERIVED TABLES & UTILITIES  ------------------------ #
def build_presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pres = (df.assign(present=df.exists.astype(int))
              .pivot_table(values="present",
                           index="patient",
                           columns=["timepoint", "pulse"],
                           aggfunc="max",
                           fill_value=0)
              .sort_index(axis=1, level=0))

    # ---------------- order patients by available-scan count ---------------- #
    pres["__count"] = pres.sum(axis=1)
    pres = pres.sort_values("__count", ascending=False).drop(columns="__count")
    return pres

def compute_spacing_stats(df: pd.DataFrame):
    ok = df[df.exists]
    by_pp   = ok.groupby(["patient", "pulse"]).axial_spacing_mm.agg(["mean", "std"]).reset_index()
    by_ptp  = ok.groupby(["patient", "timepoint", "pulse"]).axial_spacing_mm.agg(["mean", "std"]).reset_index()
    return by_pp, by_ptp

# ------------------------------  PLOTTING  ---------------------------------- #
def _ensure_fig_dir() -> None: FIG_DIR.mkdir(exist_ok=True)

def _set_spacing_ylim(ax):
    ax.set_ylim(*YLIM_SPACING)
    ax.set_yticks(np.arange(YLIM_SPACING[1], YLIM_SPACING[0]+1))
    ax.invert_yaxis()                         # 1 mm at bottom, 8 mm at top

# ──  1) presence heat-map  ────────────────────────────────────────────────── #
def plot_presence_heatmap(pres: pd.DataFrame) -> None:
    _ensure_fig_dir()
    # Remove '__count' column and empty pulse entries
    pres = pres.loc[:, pres.columns.get_level_values(1) != '']  # Remove empty pulse entries
    
    if len(pres.columns.levels) < 2:
        raise ValueError("Expected at least two levels in columns, got: "
                         f"{pres.columns.levels}")
    LOGGER.info(f"{pres.columns.levels[0].tolist()} time-points, {len(pres.index)} patients")

    # Ensure time-points are int and get max timepoint
    timepoints = pres.columns.get_level_values(0).astype(int)
    num_tp = timepoints.max()
    n_rows = pres.shape[0]; n_cols = (num_tp+1)*len(PULSES)

    # Build RGB image (white background)
    img = np.ones((n_rows, n_cols, 4))  # RGBA
    for (tp, pulse), col in pres.items():
        colour = to_rgba(PULSE_COLOURS[pulse])
        col_idx = tp*len(PULSES) + PULSES.index(pulse)
        mask = col.to_numpy(dtype=bool)
        img[mask, col_idx] = colour

    fig, ax = plt.subplots(figsize=(max(8, n_cols/4), max(6, n_rows/3)))
    ax.imshow(img, interpolation="none")

    # ticks
    ax.set_xticks([tp*len(PULSES)+1.5 for tp in range(num_tp+1)],
                  ["BL" if tp==0 else f"C{tp}" for tp in range(num_tp+1)],
                  rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows), pres.index)
    ax.set_title("Pulse presence per patient & time-point")

    # grid lines
    for x in range(0, n_cols+len(PULSES), len(PULSES)):
        ax.axvline(x-0.5, color="grey", linewidth=0.4)
    ax.axhline(-0.5, color="grey", linewidth=0.4)
    ax.set_xlim(-0.5, n_cols-0.5); ax.set_ylim(n_rows-0.5, -0.5)

    # legend proxy
    handles = [plt.Rectangle((0,0),1,1,color=PULSE_COLOURS[p]) for p in PULSES]
    ax.legend(handles, PULSES, title="Pulse", loc="upper right")
    fig.tight_layout(); fig.savefig(FIG_DIR/"01_presence_heatmap.png", dpi=300)
    plt.close(fig); LOGGER.info("Saved Figure 1 (presence heat-map)")

# ──  2) bar plots per patient/pulse  ──────────────────────────────────────── #
def plot_spacing_per_patient_pulse(stats: pd.DataFrame) -> None:
    _ensure_fig_dir()
    patients = stats.patient.unique(); per_page=5
    for i in range(0, len(patients), per_page):
        subset = patients[i:i+per_page]; pg=stats[stats.patient.isin(subset)]
        fig, ax = plt.subplots(figsize=(8, 4+len(subset)))
        x = np.arange(len(PULSES)); w = 0.15
        for j, pat in enumerate(subset):
            row = pg[pg.patient==pat]
            means = [row.loc[row.pulse==p,"mean"].values[0] if p in row.pulse.values else np.nan
                     for p in PULSES]
            ax.bar(x+j*w, means, width=w, label=pat)
        ax.set_xticks(x+(len(subset)-1)*w/2, PULSES)
        ax.set_ylabel("Axial spacing (mm)")
        _set_spacing_ylim(ax)
        ax.set_title("Average axial spacing per patient & pulse"); ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR/f"02_spacing_patient_pulse_pg{i//per_page+1}.png", dpi=300)
        plt.close(fig)
    LOGGER.info("Saved Figure 2 (patient-pulse bar-plots)")

# ──  3) global mean ± std  ────────────────────────────────────────────────── #
def plot_global_spacing_summary(stats: pd.DataFrame) -> None:
    _ensure_fig_dir()
    glob = (stats
            .groupby("pulse")["mean"]
            .agg(global_mean="mean", global_std="std")
            .reset_index())
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(glob.pulse, glob.global_mean, yerr=glob.global_std, capsize=4,
           color=[PULSE_COLOURS[p] for p in glob.pulse])
    ax.set_ylabel("Axial spacing (mm)")
    _set_spacing_ylim(ax)
    ax.set_title("Overall axial spacing (mean ± std) per pulse")
    fig.tight_layout(); fig.savefig(FIG_DIR/"03_global_spacing_summary.png", dpi=300)
    plt.close(fig); LOGGER.info("Saved Figure 3 (global summary)")

# ──  4) spacing vs time-point  ────────────────────────────────────────────── #
def plot_spacing_per_patient_timepoint(stats_tp: pd.DataFrame) -> None:
    _ensure_fig_dir()
    for pulse in PULSES:
        pdf = stats_tp[stats_tp.pulse==pulse]
        fig, ax = plt.subplots(figsize=(8,6))
        for pat, grp in pdf.groupby("patient"):
            ax.plot(grp.timepoint, grp["mean"], marker="o", label=pat)
        ax.set_xlabel("Time-point (0 = BL)"); ax.set_ylabel("Axial spacing (mm)")
        _set_spacing_ylim(ax)
        ax.set_title(f"{pulse} – axial spacing vs time-point")
        ax.legend(ncol=3, fontsize="small", title="Patient", loc="best")
        fig.tight_layout(); fig.savefig(FIG_DIR/f"04_spacing_timepoint_{pulse}.png", dpi=300)
        plt.close(fig)
    LOGGER.info("Saved Figure 4 (spacing vs time-point)")

# ----------------------------  CLI & MAIN  ---------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate meningioma study plots")
    p.add_argument("-r","--root", type=Path, required=True,
                   help="Root folder containing 'baseline/' & 'controls/'")
    p.add_argument("--recompute", action="store_true",
                   help="Ignore cache and recompute")
    p.add_argument("--cache", type=Path, default=CACHE_CSV)
    return p.parse_args()

# ─────────────────────  QUALIFYING CONTROLS REPORT  ────────────────────── #
def print_qualifying_controls(df: pd.DataFrame) -> None:
    """
    Print, for every patient, the controls that contain ≥ 3/4 pulses.

    Example line:
       Patient P42: Control 1 (missing only T2), Control 3 (all pulses present)  → Qualify
       Patient P50: None of the controls qualify
    """
    follow = df[df.timepoint > 0]
    summary_lines: list[str] = []

    for patient, grp in follow.groupby("patient"):
        msgs: list[str] = []
        for tp, tp_grp in grp.groupby("timepoint"):
            present = tp_grp[tp_grp.exists].pulse.tolist()
            missing = sorted(set(PULSES) - set(present))
            if len(present) >= 3:
                if missing:
                    msg = f"Control {tp} (missing only {missing[0]})"
                else:
                    msg = f"Control {tp} (all pulses present)"
                msgs.append(msg)
        if msgs:
            summary_lines.append(f"Patient {patient}: " + ", ".join(msgs) + "  → Qualify")
        else:
            summary_lines.append(f"Patient {patient}: None of the controls qualify")

    print("\n".join(summary_lines))


def main() -> None:
    args = parse_args()
    df = load_or_build_stats(args.root, args.cache, args.recompute)
    pres = build_presence_matrix(df)
    pp_stats, pt_stats = compute_spacing_stats(df)

    plot_presence_heatmap(pres)
    plot_spacing_per_patient_pulse(pp_stats)
    plot_global_spacing_summary(pp_stats)
    plot_spacing_per_patient_timepoint(pt_stats)
    LOGGER.info("All figures saved to %s", FIG_DIR.resolve())

    # -------- new report -------- #
    print_qualifying_controls(df)
    LOGGER.info("All figures saved to %s", FIG_DIR.resolve())

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: LOGGER.warning("Interrupted"); sys.exit(130)
