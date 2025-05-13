# file: src/mgmGrowth/tasks/superresolution/tools/parallel.py
"""Tiny wrapper around ProcessPoolExecutor for patient-level parallelism."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

from src.mgmGrowth.tasks.superresolution import LOGGER

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    func: Callable[[T], R],
    items: Sequence[T],
    jobs: int,
    desc: str = "job",
) -> list[R]:
    """
    Run *func* on each element of *items* (patient folders) using *jobs* processes.

    *jobs*:
        1 → sequential (no overhead)  
        0 or <0 → os.cpu_count()  (use all cores)

    Returns a list with the function results (discard `None`).
    """
    if jobs == 1:
        return [func(x) for x in items]

    if jobs <= 0:
        jobs = None  # means "as many as CPUs" for ProcessPoolExecutor

    LOGGER.info("Running %d %s in parallel (%s cores)…", len(items), desc, jobs or "all")
    results: list[R] = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(func, item): item for item in items}
        for fut in as_completed(futs):
            item = futs[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:  # noqa: BLE001 – log & continue
                LOGGER.error("Worker failed on %s → %s", item, e, exc_info=False)
    return results
