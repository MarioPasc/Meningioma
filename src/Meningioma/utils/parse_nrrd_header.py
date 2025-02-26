#!/usr/bin/env python

"""
Simple nrrd header parse of np.ndarray object to json-serializable
objects. Used in various scripts such as nrrd_to_nifti.py or planner.py 
"""

import numpy as np


def numpy_converter(o):
    """Convert NumPy objects into JSON-serializable objects."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.float64, np.float32)):
        return float(o)
    return str(o)
