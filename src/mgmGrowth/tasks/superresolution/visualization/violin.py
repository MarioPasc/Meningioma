#!/usr/bin/env python3
"""
make_violin_grid.py  —  per-pulse SR-quality visualisation
---------------------------------------------------------

Creates one {FORMAT} per (pulse, resolution, metric) with
axes arranged for LaTeX montage:

row = resolution  (3 mm → 7 mm)
col = metric      (PSNR | SSIM | BC)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import scienceplots

from mgmGrowth.tasks.superresolution.visualization.octant_vis import FORMAT
plt.style.use(["science", "ieee", "grid"])

# ╭──────────────────── user-specific paths ──────────────────────────────╮
METRICS_PATH = pathlib.Path(
    "/home/mariopasc/Python/Results/Meningioma/super_resolution/metrics/metrics.npz"
)
OUT_ROOT = pathlib.Path(
    "/home/mariopasc/Python/Results/Meningioma/super_resolution/metrics/plots"
)
# ╰───────────────────────────────────────────────────────────────────────╯

# ─────────── reshape dataset ───────────
d = np.load(METRICS_PATH, allow_pickle=True)
metrics       = d["metrics"][..., 0]      # slice-mean statistic
pulses        = [p.decode() if isinstance(p, bytes) else str(p) for p in d["pulses"]]
resolutions   = d["resolutions_mm"]       # [3, 5, 7]
model_names   = [m.decode() if isinstance(m, bytes) else str(m) for m in d["models"]]
metric_names  = [m.decode() if isinstance(m, bytes) else str(m) for m in d["metric_names"]]
roi_labels    = [r.decode() if isinstance(r, bytes) else str(r) for r in d["roi_labels"]]

# ─────────── plotting constants ────────
ROI_COLORS  = ["#BBBBBB", "#EE6677", "#228833", "#4477AA"]
ROI_OFFSETS = np.linspace(-0.75, 0.75, len(ROI_COLORS))   # wider separation
GAP         = 4.0                                         # more space between models
VIOLIN_W    = 0.45                                        # slimmer violins

# ─────────── helper to hide / keep axes ───────────
def configure_axes(ax: plt.Axes, metric: str, res_mm: int) -> None:
    """Hide / display spines & ticks according to montage rules."""

    # ── 1. globally hide top & right spines + ticks ────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both",
                   top=False, labeltop=False,
                   right=False, labelright=False)

    # ── 2. metric-specific spine / tick handling (unchanged) ───────────
    if metric == "PSNR":
        if res_mm in (3, 5):                     # keep y, drop x
            ax.spines["bottom"].set_visible(False)
        else:                                    # 7-mm PSNR
            ax.set_xlabel("")
    else:  # SSIM or BC
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=False)

        if res_mm not in (7,):                   # 3-mm / 5-mm
            ax.spines["bottom"].set_visible(False)

    # ── 3. remove bottom ticks for all rows above the bottom row ───────
    if res_mm != 7:                              # 3-mm and 5-mm rows
        ax.tick_params(axis="x",
                       which="both",
                       bottom=False, labelbottom=False)


# ─────────── main loop ───────────
for pulse_idx, pulse in enumerate(pulses):
    pulse_data = metrics[:, pulse_idx, ...]     # (P, res, models, metrics, roi)

    for res_idx, res_mm in enumerate(resolutions):
        y_centres = np.arange(len(model_names)) * GAP

        for met_idx, met in enumerate(metric_names):

            fig, ax = plt.subplots(figsize=(8, 6))

            # draw all violins
            for mdl_idx, mdl_name in enumerate(model_names):
                base_y = y_centres[mdl_idx]

                for roi_idx, offset in enumerate(ROI_OFFSETS):
                    vals = pulse_data[:, res_idx, mdl_idx, met_idx, roi_idx]
                    vals = vals[~np.isnan(vals)]

                    if met.upper() == "PSNR":
                        vals = np.clip(vals, 0, None)      # clip < 0 dB

                    if vals.size == 0:
                        continue

                    y_pos = base_y + offset
                    vp = ax.violinplot(
                        vals,
                        positions=[y_pos],
                        vert=False,
                        widths=VIOLIN_W,
                        showmeans=True,
                        showextrema=False,
                    )
                    body = vp["bodies"][0]
                    body.set_facecolor(ROI_COLORS[roi_idx])
                    body.set_alpha(0.85)
                    body.set_edgecolor("black")
                    vp["cmeans"].set_color("black")

            # cosmetic limits & labels
            if met.upper() == "SSIM":
                ax.set_xlim(0.85, 1.0)
            # ax.set_xlabel("PSNR (dB)" if met.upper() == "PSNR" else met)

            # y-axis labels only for PSNR plots
            if met.upper() == "PSNR":
                ax.set_yticks(y_centres)
                ax.set_yticklabels(model_names, fontsize=32)
            else:
                ax.set_yticks([])

            ax.grid(axis="x", linestyle="--", alpha=0.3)
            ax.xaxis.set_tick_params(labelsize=32)
            # apply montage-axis logic
            configure_axes(ax, met.upper(), res_mm)

            plt.tight_layout()

            # save figure
            out_dir = OUT_ROOT / pulse / f"{res_mm}mm"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{pulse}_{res_mm}mm_{met}.{FORMAT}", dpi=300)
            plt.close(fig)
# ───────────────────────── legend graphic ────────────────────────────
# put this *after* the main plotting loop, before the script ends
from matplotlib.patches import Patch
# ── pretty display names for the legend ──────────────────────────────
ROI_NAME_MAP = {
    "all":      "Volume",
    "core":     "Enhancing Tumor",
    "edema":    "Edema",
    "surround": "Non-Enhancing Tumor",
}

def display_name(code: str) -> str:
    """Return the pretty label; fall back to the raw code if unknown."""
    return ROI_NAME_MAP.get(code, code)

legend_handles = [
    Patch(facecolor=ROI_COLORS[i],
        edgecolor="black",
        label=display_name(roi_labels[i]))      # ← mapped name
    for i in range(len(roi_labels))
]


fig_leg = plt.figure(figsize=(6, 0.8))          # wide & short banner
fig_leg.legend(handles=legend_handles,
            loc="center",
            ncol=len(legend_handles),
            frameon=False,
            handlelength=1.5,
            fontsize=32)


plt.axis("off")                                 # no axes at all
legend_path = OUT_ROOT / f"roi_legend.{FORMAT}"
fig_leg.savefig(legend_path, dpi=300,
                bbox_inches="tight", transparent=True)
plt.close(fig_leg)
print(f"Legend saved to {legend_path}")