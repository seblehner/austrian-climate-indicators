"""Functions to handle setting attributes for output data."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from typing import Union

import xarray as xr

from utils import _get_logger


def set_attrs(
    climind: xr.DataArray,
    index: str,
    descr: str,
) -> Union[xr.DataArray, xr.Dataset]:
    """Helper function to set some metadata attributes.

    Args:
        climind (xr.DataArray): data for which metadata is assigned
        index (str): name of climate index
        descr (str): description of climate index
        input_xda (xr.DataArray): base variable xr.DataArray that was used to
            calculate the climate index
        input_xds (xr.Dataset): base variable xr.Dataset used to calculate the
            climate index
        extra_attrs (list[str]): allows to set some predefined extra attributes

    Returns:
        xr.DataArray, xr.Dataset: output data with assigned attributes
    """
    log = _get_logger()
    log.debug("set generic attributes")
    climind.attrs["description"] = descr
    climind = climind.to_dataset(name=index)
    general_attrs = {
        "title": "Climate Indicators for Austria",
        "institution": "GeoSphere Austria, Vienna, Austria",
        "contact": "GeoSphere Austria (sebastian.lehner@geosphere.at)",
        "Conventions": "CF-1.7",
        "creation_date": datetime.now().strftime(format="%Y-%m-%d %H:%M:%S"),
    }
    for key, val in general_attrs.items():
        climind.attrs[key] = val
    return climind
