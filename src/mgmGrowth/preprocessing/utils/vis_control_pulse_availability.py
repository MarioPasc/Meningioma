import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_pulses_grouped_by_patient(root_folder: Path):
    """
    Plot all pulses per patient with:
      - more horizontal separation between patients
      - smaller dots
      - dashed lines and dots in the same color per pulse
      - legend outside at lower center
    """
    # 1) collect all (pid, cid, pulse)
    rx = re.compile(r"MenGrowth-(\d+)-(\d{4})-([A-Za-z0-9\-]+)\.nii\.gz$")
    records = []
    for d in sorted(root_folder.iterdir()):
        if not (d.is_dir() and d.name.startswith("MenGrowth-")):
            continue
        for f in d.glob("*.nii.gz"):
            m = rx.match(f.name)
            if not m:
                continue
            pid, cid, pulse = int(m.group(1)), int(m.group(2)), m.group(3)
            records.append((pid, cid, pulse))
    if not records:
        return

    # 2) uniques
    patient_idxs = sorted({r[0] for r in records})
    pulses       = sorted({r[2] for r in records})

    # 3) layout: more space per patient, offsets per pulse
    spacing = 3.0   # widen gap between patient groups
    n       = len(pulses)
    width   = 0.8 / n  # wider span for pulses
    offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n)
    pulse_offset = {p: off for p, off in zip(pulses, offsets)}

    # pick one color per pulse from the default cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    pulse_color = {p: colors[i % len(colors)] for i, p in enumerate(pulses)}

    # 4) organize data by pulse → patient → [ctrl indices]
    data = {p: {} for p in pulses}
    for pid, cid, pulse in records:
        data[pulse].setdefault(pid, []).append(cid)

    # 5) plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    for pulse in pulses:
        c = pulse_color[pulse]
        for pid in patient_idxs:
            ctrls = sorted(data[pulse].get(pid, []))
            if not ctrls:
                continue
            x_base = pid * spacing + pulse_offset[pulse]
            x_vals = [x_base] * len(ctrls)
            # join with dashed line in same color
            ax.plot(x_vals, ctrls, linestyle="--", linewidth=1, color=c)
            # scatter in same color, smaller dots
            ax.scatter(x_vals, ctrls, color=c, s=30, label=pulse if pid==patient_idxs[0] else "")

    # 6) finalize
    ax.set_xticks([pid * spacing for pid in patient_idxs])
    ax.set_xticklabels(patient_idxs)
    ax.set_xlabel("Patient index")
    ax.set_ylabel("Control (visit) index")
    ax.set_title("Pulse availability per patient/control")
    ax.grid(True, linestyle="--", alpha=0.3)

    # legend once, outside bottom center
    ax.legend(
        title="Pulse",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=n
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()



if __name__ == "__main__":
    import sys
    root = Path("/home/mpascual/research/datasets/meningiomas/raw/MenGrowth-2025")
    plot_pulses_grouped_by_patient(root)
