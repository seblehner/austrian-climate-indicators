#!/usr/bin/env -S uv run

"""Main file to start computation of climate indices"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

import numpy as np
import xarray as xr

from climind_attrs import set_attrs
from climind_calc import calc_climate_indices
from climind_io import (
    _get_percentile_threshold_xda,
    get_outfile,
    open_input_data,
    write_climate_indices,
)
from utils import (
    _get_logger,
    get_climate_indices_df_iterator,
    get_config,
    get_input_vars_and_paths,
    get_inputs_from_climind_df,
    logger,
)


def wrapper_calc_climate_indices(
    index: str,
    group: str,
    descr: str,
    params: list[str],
    in_dir_list: list[Path],
    year: str,
    out_dir: Path,
    epsg: str,
    aggperiod: str,
    extra_kwargs: str,
    xclim_func: str,
) -> None:
    """Convenience wrapper for subsequent function calls to calculate climate indices

    Args:
        index (str): abbreviated name of climate index
        group (str): category group of climate index
        descr (str): description of climate index
        input_vars (list[str]): abbreviated input variables
        input_paths (list[Path]): paths to input variables
        year (str): year for which climate index is calculated
        out_dirpath (Path): output Path
        epsg (str): EPSG number as string for CRS information
        aggperiod (str): resampling frequency of climate indicator
        extra_kwargs (str): additional kwargs for climate indicator calculation
        xclim_func (str): name of xclim implementation within the xclim.indicator module

    Returns:
        None: None
    """
    log = _get_logger()
    # check it outfile exists already
    outfile = get_outfile(
        indexname=index,
        indexgroup=group,
        out_dir=out_dir,
        suffix=year,
        aggperiod=aggperiod,
    )
    if outfile.exists():
        log.debug(f"{outfile = } exists already")
        return None
    log.debug(f"Calculating {index = } for {year = }")

    # load input data
    input_xds = open_input_data(
        input_vars=params, input_paths=in_dir_list, years=int(year), aggperiod=aggperiod
    )

    # extract data for climate indices calculation, but keep the original
    # xr.Dataset to handle extra variables like lat/lon/dem if existing
    input_xda = []
    inherent_mask = 0
    for var_, xds_ in zip(params, input_xds):
        xda_iter = xds_[var_]
        try:
            mask_ = xds_["mask"]
            xda_iter = xda_iter.where(mask_ == 1, np.nan)
            inherent_mask = 1
            mask_notnull = xda_iter.notnull().isel(time=0).drop_vars("time")
        except KeyError:
            log.debug("no mask exists in data")
            if xda_iter.isnull().any():
                log.debug("found nan inside data, treating as mask")
                inherent_mask = 1
                mask_notnull = xda_iter.notnull().isel(time=0).drop_vars("time")

        input_xda.append(xda_iter)

    thresholds_xda, percentile_kwarg_str = _get_percentile_threshold_xda(
        params=params,
        in_dir_list=in_dir_list,
        extra_kwargs=extra_kwargs,
        aggperiod=aggperiod,
        out_dir=out_dir,
    )
    # append thresholds xda to input list for climate indicator calc
    if isinstance(thresholds_xda, xr.DataArray):
        input_xda.append(thresholds_xda)
        # redefine extra_kwargs without unwanted kwargs
        extra_kwargs = {
            key_: val_
            for key_, val_ in extra_kwargs.items()
            if key_ not in ["ref_period", percentile_kwarg_str]
        }

    # calculate climate indices
    climind = calc_climate_indices(
        xda=input_xda,
        index=index,
        aggperiod=aggperiod,
        extra_kwargs=extra_kwargs,
        xclim_func=xclim_func,
    )

    # set various attributes
    climind = set_attrs(
        climind=climind,
        index=index,
        descr=descr,
    )

    # apply data inherent mask
    if inherent_mask == 1:
        climind = climind.where(mask_notnull, np.nan)

    # write data
    write_climate_indices(
        climind=climind,
        outfile=outfile,
        epsg=epsg,
    )
    return None


if __name__ == "__main__":
    config = get_config(conf_file="config.toml")
    log = logger()

    # get some variables from config file
    year_start = config.GENERAL.YEAR_START
    year_end = config.GENERAL.YEAR_END
    out_dir = Path(config.PATHS.OUT)
    epsg = config.CRS.EPSG

    # construct iterator for climate indices to be calculated
    inddf_iter = get_climate_indices_df_iterator(config)
    clim_ind_inputs = get_inputs_from_climind_df(inddf_iter)
    # loop over climate indices dataframe to get all info for calculation
    for idx, (
        index,
        group,
        input_vars,
        descr,
        aggperiod,
        extra_kwargs,
        xclim_func,
        unit,
    ) in enumerate(clim_ind_inputs):
        log.info(f"Progress {idx + 1}/{len(clim_ind_inputs)} {index} {aggperiod}")
        input_vars, input_paths = get_input_vars_and_paths(input_vars, config.PATHS)
        for year in range(year_start, year_end + 1):
            log.debug(f"Process {year = }")
            wrapper_calc_climate_indices(
                index=index,
                group=group,
                descr=descr,
                params=input_vars,
                in_dir_list=input_paths,
                year=year,
                out_dir=out_dir,
                epsg=epsg,
                aggperiod=aggperiod,
                extra_kwargs=extra_kwargs,
                xclim_func=xclim_func,
            )
