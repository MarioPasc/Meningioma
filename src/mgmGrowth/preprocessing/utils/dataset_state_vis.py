#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MenGrowth database state dashboard with per-control missing-pulse diagnostics.

Changes vs previous:
  • Removed the bar plot of controls with >2 missing pulses.
  • Second row order: Age at baseline / Sex distribution / Visits per year.
  • Bottom heatmap taller and patients ordered by completeness.
  • Heatmap legend moved below the axis, outside the plot.
  • “Two missing” and “>2 missing” fused → one category:
      label = "Only two or less pulses available", color = '#CC3311'.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# --------------------------- Matplotlib config ---------------------------

def configure_matplotlib() -> None:
    """
    Configure matplotlib and scienceplots with LaTeX and requested typography.
    Falls back gracefully if LaTeX or scienceplots are not available.
    """
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science'])  # base science style
    except Exception as e:
        logging.warning("scienceplots not available: %s. Continuing with default style.", e)

    plt.rcParams.update({
        'figure.dpi': 600,
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times'],
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'savefig.bbox': 'tight',
    })
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    except Exception as e:
        logging.warning("LaTeX not available: %s. Falling back to non-LaTeX text.", e)
        plt.rcParams['text.usetex'] = False


# ------------------------------ Data model ------------------------------

@dataclass(frozen=True)
class Visit:
    """A single visit (baseline or control)."""
    patient_p: str
    patient_num: str
    visit_index: int            # 0 = baseline, 1.. = c1..
    label: str                  # 'baseline','c1','c2',...
    date: Optional[datetime]


@dataclass
class PatientSummary:
    """Aggregated summary for one patient."""
    patient_p: str
    patient_num: str
    age: Optional[float]
    sex: Optional[int]          # code as in metadata
    n_visits: int
    n_controls: int
    spacing_months: List[float]
    pulses_per_visit: List[Set[str]]


# ------------------------------ Utilities -------------------------------

_DATE_FMT_CANDIDATES: Tuple[str, ...] = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y")
VISIT_ORDER = ["baseline", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
_NIFTI_RE = re.compile(r"^MenGrowth-(?P<num>\d{5})-(?P<visit>\d{4})-(?P<pulse>[A-Za-z0-9]+)\.nii\.gz$")

def parse_date_safe(x: Optional[str]) -> Optional[datetime]:
    """Parse date string or return None if invalid."""
    if x is None:
        return None
    s = str(x).strip()
    if not re.search(r"\d{4}", s):
        return None
    for fmt in _DATE_FMT_CANDIDATES:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None

def months_between(a: datetime, b: datetime) -> float:
    """Fractional months between two datetimes."""
    return (b - a).days / 30.4375

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------- IO and extraction ---------------------------

def build_maps(map_dict: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return P->num and num->P maps."""
    p2num = dict(map_dict)
    num2p = {v: k for k, v in p2num.items()}
    return p2num, num2p

def extract_visits_from_metadata(meta: dict, patient_p: str) -> List[Visit]:
    """
    Build ordered visits from metadata_clean for one patient P-code.
    Uses 'first_study.rm.date' as baseline if present, then c1..cN.
    """
    node = meta.get(patient_p, {})
    baseline_date = parse_date_safe((((node.get("first_study") or {}).get("rm") or {}).get("date")))
    visits: List[Visit] = []
    if baseline_date is not None:
        visits.append(("baseline", baseline_date))
    for lab in [k for k in node.keys() if k.startswith("c")]:
        date = parse_date_safe((node[lab] or {}).get("date"))
        visits.append((lab, date))

    def order_key(item: Tuple[str, Optional[datetime]]) -> Tuple[int, datetime]:
        lab, d = item
        idx = VISIT_ORDER.index(lab) if lab in VISIT_ORDER else 999
        return (idx, d or datetime(2100, 1, 1))

    visits_sorted = sorted(visits, key=order_key)
    out: List[Visit] = []
    for i, (lab, d) in enumerate(visits_sorted):
        out.append(Visit(patient_p=patient_p, patient_num="", visit_index=i, label=lab, date=d))
    return out

def scan_pulses(root: Path) -> Dict[str, Dict[int, Set[str]]]:
    """
    Walk root and record pulses per patient-num and visit-index inferred from filenames.
    Returns: { '00012': { 0: {'T1','T2',...}, 1: {...}, ... }, ... }
    """
    result: Dict[str, Dict[int, Set[str]]] = {}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        mdir = re.match(r"^MenGrowth-(\d{5})$", entry.name)
        if not mdir:
            continue
        num = mdir.group(1)
        result.setdefault(num, {})
        for fn in entry.iterdir():
            if not fn.is_file():
                continue
            m = _NIFTI_RE.match(fn.name)
            if not m:
                continue
            if m.group("num") != num:
                continue
            visit_idx = int(m.group("visit"))
            pulse = m.group("pulse").upper()
            result[num].setdefault(visit_idx, set()).add(pulse)
    return result


# --------------------------- Aggregation logic ---------------------------

def assemble_patient_summaries(
    meta: dict,
    p2num: Dict[str, str],
    pulses_fs: Dict[str, Dict[int, Set[str]]],
) -> Tuple[List[PatientSummary], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create per-patient summaries and four analysis tables:
      patients_df, intervals_df, visits_df, pulse_cov
    """
    patients: List[PatientSummary] = []
    visits_rows: List[Dict] = []
    intervals_rows: List[Dict] = []

    for pcode, num in p2num.items():
        if pcode not in meta:
            logging.warning("Patient %s present in map but not in metadata. Skipping.", pcode)
            continue

        vlist = extract_visits_from_metadata(meta, pcode)
        for i in range(len(vlist)):
            vlist[i] = Visit(
                patient_p=vlist[i].patient_p, patient_num=num,
                visit_index=vlist[i].visit_index, label=vlist[i].label, date=vlist[i].date
            )

        pulses_map = pulses_fs.get(num, {})
        pulses_per_visit: List[Set[str]] = []
        for i in range(len(vlist)):
            pulses_per_visit.append(set(pulses_map.get(i, set())))

        gen = (meta[pcode] or {}).get("general", {})
        try:
            age = float(gen.get("age")) if gen.get("age") not in (None, "") else None
        except Exception:
            age = None
        sex_val = gen.get("sex")
        try:
            sex = int(sex_val) if sex_val not in (None, "") else None
        except Exception:
            sex = None

        dates = [v.date for v in vlist if v.date is not None]
        dates_sorted = sorted(dates)
        gaps = [months_between(dates_sorted[i], dates_sorted[i+1]) for i in range(len(dates_sorted)-1)]

        for v in vlist:
            visits_rows.append({
                "patient_p": pcode, "patient_num": num,
                "visit_index": v.visit_index, "label": v.label, "date": v.date,
                "n_pulses": len(pulses_map.get(v.visit_index, set())),
                "pulses": sorted(list(pulses_map.get(v.visit_index, set()))),
            })
        for g in gaps:
            intervals_rows.append({"patient_p": pcode, "patient_num": num, "gap_months": g})

        patients.append(PatientSummary(
            patient_p=pcode, patient_num=num, age=age, sex=sex,
            n_visits=len(vlist), n_controls=max(0, len(vlist)-1),
            spacing_months=gaps, pulses_per_visit=pulses_per_visit,
        ))

    patients_df = pd.DataFrame([{
        "patient_p": p.patient_p, "patient_num": p.patient_num,
        "age": p.age, "sex": p.sex,
        "n_visits": p.n_visits, "n_controls": p.n_controls,
        "median_gap_months": np.median(p.spacing_months) if p.spacing_months else np.nan,
    } for p in patients]).sort_values(["patient_num"])

    intervals_df = pd.DataFrame(intervals_rows)
    visits_df = pd.DataFrame(visits_rows)
    if not visits_df.empty:
        visits_df["visit_has_date"] = visits_df["date"].notna()

    # pulse coverage
    all_pulses: List[str] = []
    for p in patients:
        for s in p.pulses_per_visit:
            all_pulses.extend(list(s))
    pulse_series = pd.Series(all_pulses, dtype="object")
    if not pulse_series.empty:
        pulse_counts = pulse_series.value_counts().sort_index()
        total_visits = int(visits_df.shape[0])
        pulse_cov = pd.DataFrame({"pulse": pulse_counts.index, "count": pulse_counts.values})
        pulse_cov["coverage_pct"] = 100.0 * pulse_cov["count"] / total_visits
    else:
        pulse_cov = pd.DataFrame(columns=["pulse", "count", "coverage_pct"])

    return patients, patients_df, intervals_df, visits_df, pulse_cov


# ---------------------- Missing-pulse diagnostics ------------------------

def build_missing_matrix(
    visits_df: pd.DataFrame,
    expected_pulses: List[str],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build a matrix codes[y, x] for patient P-IDs (rows) vs controls (cols).
    Coding:
      -1 → no visit
       0 → complete (no missing)
       1..K → exactly one missing pulse j (use pulse color j)
       S (=K+1) → missing_count >= 2  [fused “Only two or less pulses available”]
    """
    dfc = visits_df[visits_df["label"].str.startswith("c")].copy()
    if dfc.empty:
        return np.zeros((0, 0), dtype=int), [], []

    row_labels = sorted(dfc["patient_p"].unique().tolist())
    max_ctrl = int(dfc["visit_index"].max())
    col_labels = [f"c{j}" for j in range(1, max_ctrl + 1)]

    pulses_map = {(p, vi): set(ps) for p, vi, ps in zip(dfc["patient_p"], dfc["visit_index"], dfc["pulses"])}
    exp_set = set(expected_pulses)
    K = len(expected_pulses)
    severe_code = K + 1

    codes = np.full((len(row_labels), len(col_labels)), -1, dtype=int)
    for iy, p in enumerate(row_labels):
        for jx, c_lab in enumerate(col_labels, start=1):
            present = pulses_map.get((p, jx), None)
            if present is None:
                continue  # keep -1
            missing_count = len(exp_set - present)
            if missing_count == 0:
                codes[iy, jx-1] = 0
            elif missing_count == 1:
                # code of the missing pulse is 1..K, per expected_pulses order
                missing_pulse = list(exp_set - present)[0]
                codes[iy, jx-1] = 1 + expected_pulses.index(missing_pulse)
            else:
                codes[iy, jx-1] = severe_code
    return codes, row_labels, col_labels


def order_rows_by_completeness(codes: np.ndarray, row_labels: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Order rows from most complete to least complete.
    Score = proportion of complete cells among non-masked cells.
    Tie-breaker: fewer severe cells, more observed cells, alphabetical ID.
    """
    if codes.size == 0:
        return codes, row_labels
    K = codes.max()  # includes severe_code
    severe_code = K  # by construction in build_missing_matrix
    # compute metrics per row
    order = []
    for i, pid in enumerate(row_labels):
        row = codes[i]
        observed = (row >= 0)
        n_obs = int(observed.sum())
        n_complete = int((row == 0).sum())
        n_severe = int((row == severe_code).sum())
        score = n_complete / n_obs if n_obs > 0 else 0.0
        order.append(( -score, n_severe, -n_obs, pid, i))  # negatives for descending
    order.sort()
    idxs = [t[-1] for t in order]
    new_codes = codes[idxs, :]
    new_labels = [row_labels[i] for i in idxs]
    return new_codes, new_labels


# ------------------------------- Plotting --------------------------------

def draw_dashboard(
    patients_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    visits_df: pd.DataFrame,
    pulse_cov: pd.DataFrame,
    out_path: Path,
) -> None:
    """Compose a single figure with multiple axes and a tall heatmap."""
    configure_matplotlib()

    # KPIs
    n_patients = int(patients_df.shape[0])
    n_visits = int(visits_df.shape[0]) if not visits_df.empty else 0
    n_controls = int(patients_df["n_controls"].sum()) if not patients_df.empty else 0
    med_controls = float(patients_df["n_controls"].median()) if n_patients else float("nan")
    med_gap = float(intervals_df["gap_months"].median()) if not intervals_df.empty else float("nan")
    sex_counts = patients_df["sex"].value_counts(dropna=True).sort_index() if not patients_df.empty else pd.Series([], dtype=int)
    ages = patients_df["age"].dropna() if not patients_df.empty else pd.Series([], dtype=float)

    # expected pulses and palette
    expected_pulses = sorted(pulse_cov["pulse"].tolist()) if not pulse_cov.empty else []
    pulse_palette = ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311']
    color_map_for_pulses = {p: pulse_palette[i % len(pulse_palette)] for i, p in enumerate(expected_pulses)}
    color_complete = '#FFFFFF'
    color_novisit = '#F0F0F0'
    color_severe = '#CC3311'  # fused severe category

    # missing matrix
    codes, row_labels, col_labels = build_missing_matrix(visits_df=visits_df, expected_pulses=expected_pulses)
    if codes.size > 0:
        codes, row_labels = order_rows_by_completeness(codes, row_labels)

    # colormap aligned to codes:
    # 0 = white; 1..K = per-pulse color; K+1 = severe
    K = len(expected_pulses)
    lut = [color_complete] + [color_map_for_pulses[p] for p in expected_pulses] + [color_severe]
    cmap = ListedColormap(lut)
    M = np.ma.masked_where(codes < 0, codes)

    # Layout: 4 rows × 4 cols. Last row taller.
    fig = plt.figure(figsize=(12.0, 10.5))
    gs = fig.add_gridspec(4, 4, height_ratios=[0.65, 1.1, 1.1, 2.6], hspace=0.7, wspace=0.6)

    # Row 0: KPI text
    ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
    lines = [
        rf"\textbf{{Patients}}: {n_patients}",
        rf"\textbf{{Visits}}: {n_visits} \quad \textbf{{Controls}}: {n_controls}",
        rf"\textbf{{Median controls/patient}}: {med_controls:.1f}",
        rf"\textbf{{Median spacing}}: {med_gap:.1f} months" if not math.isnan(med_gap) else r"\textbf{Median spacing}: NA",
    ]
    if not pulse_cov.empty:
        top_pulses = pulse_cov.sort_values("coverage_pct", ascending=False).head(5)
        pulse_str = ", ".join([f"{r.pulse}: {r.coverage_pct:.0f}%" for _, r in top_pulses.iterrows()])
        lines.append(rf"\textbf{{Top pulse coverage}}: {pulse_str}")
    ax0.text(0.01, 0.92, "\n".join(lines), va="top", ha="left")

    # Row 1: Controls per patient / Spacing / Pulse coverage
    ax1 = fig.add_subplot(gs[1, 0])
    vals = patients_df["n_controls"].values if not patients_df.empty else []
    ax1.hist(vals, bins=range(0, int(max(vals, default=0))+2), edgecolor="black")
    ax1.set_title("Controls per patient"); ax1.set_xlabel("Controls"); ax1.set_ylabel("Patients")

    ax2 = fig.add_subplot(gs[1, 1])
    data_bp = [intervals_df["gap_months"].values] if not intervals_df.empty else [[]]
    ax2.boxplot(data_bp, vert=True, showmeans=True, whis=1.5)
    ax2.set_title("Inter-visit spacing"); ax2.set_ylabel("Months"); ax2.set_xticks([])

    ax3 = fig.add_subplot(gs[1, 2:])
    if not pulse_cov.empty:
        order = pulse_cov.sort_values(["coverage_pct", "pulse"], ascending=[False, True])
        ax3.bar(order["pulse"], order["coverage_pct"])
        ax3.set_ylim(0, max(100, min(100, order["coverage_pct"].max() * 1.1)))
        ax3.set_ylabel("Coverage (%)"); ax3.set_title("Pulse availability across visits")
        ax3.tick_params(axis="x", rotation=45)
    else:
        ax3.text(0.5, 0.5, "No pulses detected", ha="center", va="center"); ax3.set_axis_off()

    # Row 2: Age / Sex / Visits per year
    ax_age = fig.add_subplot(gs[2, 0])
    if not ages.empty:
        bins = max(5, min(20, int(np.ceil(np.sqrt(len(ages))))))
        ax_age.hist(ages.values, bins=bins, edgecolor="black")
        ax_age.set_title("Age at baseline"); ax_age.set_xlabel("Years")
    else:
        ax_age.text(0.5, 0.5, "No age data", ha="center", va="center"); ax_age.set_axis_off()

    ax_sex = fig.add_subplot(gs[2, 1])
    if not sex_counts.empty:
        labels = [str(int(k)) for k in sex_counts.index]
        ax_sex.pie(sex_counts.values, labels=labels, autopct="%1.0f%%", startangle=90)
        ax_sex.set_title("Sex distribution\n(code as in metadata)")
    else:
        ax_sex.text(0.5, 0.5, "No sex data", ha="center", va="center"); ax_sex.set_axis_off()

    ax_year = fig.add_subplot(gs[2, 2:])
    if not visits_df.empty and visits_df["date"].notna().any():
        years = pd.to_datetime(visits_df["date"]).dt.year.dropna().astype(int)
        counts = years.value_counts().sort_index()
        ax_year.bar(counts.index.astype(str), counts.values)
        ax_year.set_title("Visits per year"); ax_year.set_xlabel("Year"); ax_year.set_ylabel("Visits")
    else:
        ax_year.text(0.5, 0.5, "No dated visits", ha="center", va="center"); ax_year.set_axis_off()

    # Row 3: Heatmap (tall) + legend outside bottom
    axM = fig.add_subplot(gs[3, :])
    if codes.size > 0:
        im = axM.imshow(M, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=K+1)
        im.cmap.set_bad(color_novisit)  # no-visit cells
        axM.set_yticks(np.arange(len(row_labels))); axM.set_yticklabels(row_labels)
        axM.set_xticks(np.arange(len(col_labels))); axM.set_xticklabels(col_labels)
        axM.set_xlabel("Controls"); axM.set_ylabel("Patients (P-ID)")
        axM.set_title("Missing pulses per control")
        # Legend below the axis
        legend_handles = [Patch(facecolor=color_complete, edgecolor='black', label="complete")]
        for p in expected_pulses:
            legend_handles.append(Patch(facecolor=color_map_for_pulses[p], edgecolor='black',
                                        label=f"missing {p}"))
        legend_handles.append(Patch(facecolor=color_severe, edgecolor='black',
                                    label="Only two or less pulses available"))
        axM.legend(handles=legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.18), ncol=min(3, 2 + len(expected_pulses)))
    else:
        axM.text(0.5, 0.5, "No controls to display", ha="center", va="center"); axM.set_axis_off()

    fig.suptitle("MenGrowth 2025 • Database State Panel", y=0.985)
    plt.subplots_adjust(bottom=0.22)  # room for legend outside heatmap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single data panel for MenGrowth database state.")
    parser.add_argument("--root", type=Path, default=Path(
        "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/MenGrowth-2025"),
                        help="Root directory with MenGrowth-000NN folders.")
    parser.add_argument("--meta", type=Path, default=Path(
        "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/MenGrowth-2025/metadata_clean.json"),
                        help="Path to metadata_clean.json.")
    parser.add_argument("--map", type=Path, default=Path(
        "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/MenGrowth-2025/patient_id_map.json"),
                        help="Path to patient_id_map.json mapping Pxx -> 000NN.")
    parser.add_argument("--out", type=Path, default=Path(
        "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/dashboard/mengrowth_dashboard.png"),
                        help="Output image file (.png or .pdf).")
    parser.add_argument("--export_csv_dir", type=Path, default=Path(
        "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/dashboard"),
                        help="Optional directory to export CSV summaries.")
    parser.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG,INFO,...)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="[%(levelname)s] %(message)s")

    meta = load_json(args.meta)
    p2num, _ = build_maps(load_json(args.map))
    pulses_fs = scan_pulses(args.root)
    logging.info("Scanned %d patients with NIfTI files.", len(pulses_fs))

    _, patients_df, intervals_df, visits_df, pulse_cov = assemble_patient_summaries(
        meta=meta, p2num=p2num, pulses_fs=pulses_fs
    )

    if args.export_csv_dir:
        args.export_csv_dir.mkdir(parents=True, exist_ok=True)
        patients_df.to_csv(args.export_csv_dir / "patients_summary.csv", index=False)
        intervals_df.to_csv(args.export_csv_dir / "intervals.csv", index=False)
        visits_df.to_csv(args.export_csv_dir / "visits.csv", index=False)
        pulse_cov.to_csv(args.export_csv_dir / "pulse_coverage.csv", index=False)
        logging.info("CSV summaries exported to %s", str(args.export_csv_dir))

    draw_dashboard(
        patients_df=patients_df, intervals_df=intervals_df,
        visits_df=visits_df, pulse_cov=pulse_cov, out_path=args.out,
    )
    logging.info("Panel saved to %s", str(args.out))





if __name__ == "__main__":
    main()
