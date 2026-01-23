#!/usr/bin/env -S uv run

"""Evaluation script that calculates various aggregations and quantities from climate indiators."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

import xarray as xr
from scipy.stats import mannwhitneyu

from utils import (
    get_climate_indices_df_iterator,
    get_config,
    get_inputs_from_climind_df,
    logger,
)


def get_eval_outfile_path(str_: str) -> Path:
    """get path to output file

    Args:
        str_ (str): suffix for output data

    Returns:
        Path: path to output file
    """
    outfile_ = Path(
        str(outpath_ind).replace("indicators", f"indicators_{str_}"),
        f"{index}_{aggperiod}_{str_}.nc",
    )
    outfile_.parent.mkdir(exist_ok=True, parents=True)
    return outfile_


def MWU(xda1: xr.DataArray, xda2: xr.DataArray, dim: str) -> xr.DataArray:
    """calculates a Mann Whitney U test between the two input data

    Args:
        xda1 (xr.DataArray): sample 1
        xda2 (xr.DataArray): sample 2
        dim (str): core dimension along which the data is calculated

    Raises:
        ValueError: raises an error if the provided dimensions are not supported

    Returns:
        xr.DataArray: p values from the MWU test
    """
    xda2["time"] = xda1.time
    if dim == "time":
        return xr.apply_ufunc(
            mannwhitneyu,
            xda1.load(),
            xda2.load(),
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[], []],
            vectorize=True,
            kwargs=dict(alternative="two-sided", nan_policy="omit"),
        )
    elif dim == "season":
        return (
            x.sel(season=["DJF", "MAM", "JJA", "SON"])
            for x in xr.apply_ufunc(
                mannwhitneyu,
                xda1.load().groupby("time.season"),
                xda2.load().groupby("time.season"),
                input_core_dims=[["time"], ["time"]],
                output_core_dims=[[], []],
                vectorize=True,
                kwargs=dict(alternative="two-sided", nan_policy="omit"),
            )
        )
    else:
        raise ValueError(f"Incompatible {dim = }")


if __name__ == "__main__":
    config = get_config(conf_file="config.toml")
    log = logger()

    # get some variables from config file
    year_start = config.GENERAL.YEAR_START
    year_end = config.GENERAL.YEAR_END
    out_dir = Path(config.PATHS.OUT)
    epsg = config.CRS.EPSG

    years_past = (1961, 1990)
    years_now = (1991, 2020)

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

        outpath_ind = Path(out_dir, group, index, aggperiod)

        try:
            datapath = "/".join([str(outpath_ind), "*.nc"])
            xda = xr.open_mfdataset(
                datapath,
                engine="h5netcdf",
                decode_timedelta=False,
            )[index]
        except OSError as err:
            log.info(f"Could not open dataset for {datapath} with error: {err}")

        ## calc areamean per aggperiod
        ameanfile = get_eval_outfile_path("areamean")
        if ameanfile.exists():
            log.info("Exists already: areamean")
        else:
            log.info("Calculating areamean")
            areamean = xda.mean(dim=["x", "y"])
            areamean.attrs["units"] = unit
            areamean.to_netcdf(ameanfile, engine="h5netcdf")

        ## calc climatological mean for 1961--1990 and 1991 -- 2020
        climpastfile = get_eval_outfile_path(f"clim_{years_past[0]}_{years_past[1]}")
        climnowfile = get_eval_outfile_path(f"clim_{years_now[0]}_{years_now[1]}")
        clim_past_sel = xda.sel(time=slice(str(years_past[0]), str(years_past[1])))
        clim_now_sel = xda.sel(time=slice(str(years_now[0]), str(years_now[1])))

        if climpastfile.exists():
            log.info(
                f"Exists already: climatological field {str(years_past[0]), str(years_past[1])}"
            )
        else:
            log.info(
                f"Calculating climatological field {str(years_past[0]), str(years_past[1])}"
            )
            if aggperiod in [
                "yea",
                "hydrological_year",
                "summer_halfyear",
                "winter_halfyear",
            ]:
                clim_past = clim_past_sel.mean(dim="time")
            elif aggperiod == "sea":
                clim_past = (
                    clim_past_sel.groupby("time.season")
                    .mean()
                    .sel(season=["DJF", "MAM", "JJA", "SON"])
                )
            else:
                raise ValueError(f"Incompatible {aggperiod = }")
            clim_past.attrs["units"] = unit
            clim_past.compute().to_netcdf(climpastfile, engine="h5netcdf")

        if climnowfile.exists():
            log.info(
                f"Exists already: climatological field {str(years_now[0]), str(years_now[1])}"
            )
        else:
            log.info(
                f"Calculating climatological field {str(years_now[0]), str(years_now[1])}"
            )
            if aggperiod in [
                "yea",
                "hydrological_year",
                "summer_halfyear",
                "winter_halfyear",
            ]:
                clim_now = clim_now_sel.mean(dim="time")
            elif aggperiod == "sea":
                clim_now = (
                    clim_now_sel.groupby("time.season")
                    .mean()
                    .sel(season=["DJF", "MAM", "JJA", "SON"])
                )
            else:
                raise ValueError(f"Incompatible {aggperiod = }")
            clim_now.attrs["units"] = unit
            clim_now.compute().to_netcdf(climnowfile, engine="h5netcdf")

        ## statistical significance test between clim mean
        pvalsfile = get_eval_outfile_path("significance")
        if pvalsfile.exists():
            log.info("Exists already: statistical significance")
        else:
            log.info("Calculating statistical significance")
            aggdim = (
                "time"
                if aggperiod
                in ["yea", "hydrological_year", "summer_halfyear", "winter_halfyear"]
                else "season"
            )
            _, pvals = MWU(clim_past_sel, clim_now_sel, dim=aggdim)
            pvals.to_netcdf(pvalsfile, engine="h5netcdf")
