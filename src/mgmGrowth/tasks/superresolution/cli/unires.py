#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
unires.py
===================

Batch driver for the UNIRES super-resolution algorithm on BraTS-style
low-resolution datasets (3 mm / 5 mm / 7 mm).

* For every *subject* directory inside ``--input-dir`` the script expects
  exactly **one** file ending in ``-t1c.nii.gz``, **one** ending in
  ``-t2w.nii.gz`` and **one** ending in ``-t2f.nii.gz``.
* It assembles the triplet in the order *T1c – T2w – T2f* and calls

      unires --device <device> --dir_out <out_dir>/<subject>  T1  T2  FLAIR

---------------------------------------------------------------------------
Usage (within an sbatch script)
---------------------------------------------------------------------------
$ python unires.py \
      --input-dir  /path/to/3mm \
      --output-dir /path/to/results/3mm \
      --device     cuda \
      --threads    8
---------------------------------------------------------------------------
Author  : Mario Pascual González
Created : 2025-07-02

We are using a separate logger from the module-logger since this is a standalone
script. This is because the UniRES package does not support 40-- series GTX cards, 
therefore we cannot add this script to our project dependencies. 

Check https://github.com/brudfors/UniRes/issues/24 for more details.
--------------------------------------------------------------------------- 
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


# --------------------------------------------------------------------------- #
#                               Data structures                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SubjectTriplet:
    """Container with the three pulse paths for one subject."""

    sid: str           # Subject identifier (= folder name)
    t1c: Path          # Path to *T1-contrast* image
    t2w: Path          # Path to T2-weighted image
    t2f: Path          # Path to FLAIR/T2-FLAIR image


# --------------------------------------------------------------------------- #
#                                Core helpers                                 #
# --------------------------------------------------------------------------- #
def discover_subjects(root: Path) -> List[SubjectTriplet]:
    """
    Scan *root* for subject folders and return the complete triplets.

    Any folder not containing the three expected files **is skipped** with a
    warning.

    Parameters
    ----------
    root : pathlib.Path
        Dataset resolution folder (e.g. '.../low_res/3mm').

    Returns
    -------
    list[SubjectTriplet]
        All discovered and complete subjects.
    """
    subjects: List[SubjectTriplet] = []

    for subj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        t1c  = list(subj_dir.glob("*-t1c.nii.gz"))
        t2w  = list(subj_dir.glob("*-t2w.nii.gz"))
        t2fl = list(subj_dir.glob("*-t2f.nii.gz"))

        if len(t1c) == len(t2w) == len(t2fl) == 1:
            subjects.append(
                SubjectTriplet(
                    sid=subj_dir.name,
                    t1c=t1c[0],
                    t2w=t2w[0],
                    t2f=t2fl[0],
                )
            )
        else:
            logging.warning(
                "Skipping %s (t1c=%d, t2w=%d, t2f=%d)",
                subj_dir.name, len(t1c), len(t2w), len(t2fl)
            )

    return subjects


def build_cmd(s: SubjectTriplet, out_root: Path, device: str) -> List[str]:
    """
    Assemble the `unires` command for one *subject*.

    The subject-specific output directory is created if necessary.
    """
    dest = out_root / s.sid
    dest.mkdir(parents=True, exist_ok=True)

    return [
        "unires",
        "--device", device,
        "--dir_out", str(dest),
        str(s.t1c), str(s.t2w), str(s.t2f)
    ]


def run_cmd(cmd: List[str], threads: int) -> None:
    """
    Execute *cmd* with proper thread affinity and detailed logging.

    If UNIRES returns a non-zero exit code, stdout/stderr are dumped to the
    log for post-mortem inspection.
    """
    env = {
        **os.environ,
        "OMP_NUM_THREADS":     str(threads),
        "NITORCH_NUM_THREADS": str(threads),
    }

    logging.info("►  %s", " ".join(cmd))
    completed = subprocess.run(cmd, env=env, text=True,
                               capture_output=True, check=False)

    if completed.returncode:
        logging.error(
            "UNIRES FAILED (rc=%d)\n----- stdout -----\n%s\n----- stderr -----\n%s",
            completed.returncode, completed.stdout, completed.stderr
        )
    else:
        logging.info("✓  finished successfully")


# --------------------------------------------------------------------------- #
#                                   CLI                                       #
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_unires_batch.py",
        description="Iterate over BraTS low-res subjects and run UNIRES."
    )
    p.add_argument("--input-dir",  type=Path, required=True,
                   help="Folder containing <sid>/ sub-folders.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Root where per-subject results are written.")
    p.add_argument("--device", default="cuda",
                   help="UNIRES --device flag (default: cuda).")
    p.add_argument("--threads", type=int, default=8,
                   help="CPUs to expose via OMP_NUM_THREADS (default: 8).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print UNIRES commands but do NOT execute them.")
    return p.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Input  : %s", args.input_dir)
    logging.info("Output : %s", args.output_dir)
    logging.info("Device : %s | Threads : %d", args.device, args.threads)

    subjects = discover_subjects(args.input_dir)
    logging.info("Discovered %d valid subject(s)", len(subjects))

    if not subjects:
        logging.error("Nothing to process – exiting.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for subj in subjects:
        cmd = build_cmd(subj, args.output_dir, args.device)

        if args.dry_run:
            print(" ".join(cmd))
        else:
            run_cmd(cmd, args.threads)


if __name__ == "__main__":
    main()
