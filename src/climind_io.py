"""I/O functions used to handle data loading and writing."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from typing import Union

import rioxarray
import xarray as xr

from climind_calc import calculate_aggperiod_percentile, calculate_doy_percentile
from utils import _get_logger


def open_data(
    path_to_data: str,
    year_start: int = None,
    year_end: int = None,
    year_period: str = None,
) -> xr.Dataset:
    """Load data needed to calculate the specified climate indices
    from a given path. Note that all .nc data contained in the folder is loaded

    Args:
        path_to_data (str): Path to the data in the form of the parent folder where data
            resides in
        year_start (int, optional): Starting year for subselection of data. Defaults to None.
        year_end (int, optional): Ending year for subselection of data. When None, uses the
            same value as year_start. Defaults to None.
        year_period (str, optional): Controls if shifted temporal selection is needed. Can be
            'hydrological_year', or 'winter_halfyear', 'summer_halfyear' otherwise defaults to calendar years.
            Defaults to None.

    Returns:
        xr.Dataset: Opened and temporally subselected data
    """
    log = _get_logger()
    log.debug(
        f"Open data from {path_to_data}\n\tfor {year_start = } {year_end = }"
        f" {year_period = }"
    )
    path_ = Path(path_to_data, "*.nc")
    if not year_end:
        year_end = year_start
    if (
        year_period == "sea"
    ):  # seasonal data is structured to add DEC of last year to current year
        time_slicer = slice(f"{year_start - 1}-12-01", f"{year_end}-11-30")
    elif year_period in ["hydrological_year", "winter_halfyear"]:
        time_slicer = slice(f"{year_start - 1}-10-01", f"{year_end}-09-30")
    else:
        time_slicer = slice(f"{year_start}-01-01", f"{year_end}-12-31")
    return xr.open_mfdataset(
        str(path_), compat="override", coords="minimal", engine="h5netcdf"
    ).sel(time=time_slicer)


def open_input_data(
    input_vars: list[str],
    input_paths: list[str],
    years: Union[int, tuple[int, int]],
    aggperiod: str,
) -> list[xr.Dataset]:
    """Convenience wrapper around open_data to open multiple xr.Dataset and return within
    a list.

    Returns:
        list[xr.Dataset]: opened data within a list
    """
    log = _get_logger()
    log.debug(f"{input_vars = }")
    xds_list = []
    for var_, path_ in zip(input_vars, input_paths):
        xds_iter = open_data(
            path_to_data=path_, year_start=years, year_period=aggperiod
        )
        if var_ == "RR":
            xds_iter = xds_iter.rename_vars({"RRhr": "RR"})
            # SPARTACUS units are kg m-2 per default, which are not compatible default units
            # because the time is missing. Therefor set manually to "kg m-2 day-1".
            log.warning(f"setting units for {var_} to 'kg m-2 day-1'")
            xds_iter[var_].attrs["standard_name"] = "precipitation_flux"
            xds_iter[var_].attrs["units"] = "kg m-2 day-1"
        xds_list.append(xds_iter)
    return xds_list


def write_climate_indices(
    climind: xr.Dataset,
    outfile: Path,
    epsg: str,
    encoding: dict = {},
    compression: bool = False,
):
    """function to save climate indices data to disk as netcdf file.

    Args:
        climind (xr.Dataset): data to be saved
        outfile (Path): path to output file
        epsg (str): EPSG number as str to write CRS information
        encoding (dict, optional): Additional encoding used for xr.to_netcdf(). Defaults to {}.
        compression (bool, optional): boolean input if compression is to be used, or not. Defaults to False.
    """
    log = _get_logger()
    try:
        # check if lcc data exist
        log.debug(f"checking for lcc attr: {climind.lambert_conformal_conic}")
        encoding = encoding | {"lambert_conformal_conic": {"dtype": str}}
    except AttributeError:
        log.debug("no lcc attr found, continuing without expanding encoding")

    ## Note: saving data with compression yields errors with gdalsrsinfo,
    ## hence no compression is used at the moment.
    # add compression
    if compression:
        comp = dict(zlib=True, complevel=5)
        if isinstance(climind, xr.Dataset):
            encoding = encoding | {var: comp for var in climind.data_vars}
        elif isinstance(climind, xr.DataArray):
            encoding = encoding | {climind.name: comp}

    # add crs information
    climind = climind.rio.write_crs(int(epsg))

    log.debug(f"Saving file to {outfile} with {encoding = }")
    climind.to_netcdf(outfile, encoding=encoding, engine="h5netcdf")
    return None


def get_outfile(
    indexname: str,
    indexgroup: str,
    out_dir: Path,
    suffix: str = "",
    aggperiod: str = "yea",
) -> Path:
    """Helper function to construct the output filename.

    Args:
        indexname (str): Start of Filename, name of the climate index
        indexgroup (str): group category of the climate index
        out_dirpath (Path): path to parent dir of output file
        suffix (str, optional): Suffix to be added to the filename, in front of the file type.
            Defaults to "".
        aggperiod (str, optional): resampling frequency. annual/seasonal <> yea/sea.
            Defaults to "yea".

    Returns:
        Path: path to output file
    """
    if suffix != "":
        suffix = f"_{suffix}"
    if indexname in ["ET0_Qcold", "ET0_Qdry", "ET0_Qwarm", "ET0_Qwet"]:
        # ET0 quarterly calcs need input data based on sea, but the resulting climind
        # is yea, so manually override the out_file here
        out_file = Path(
            out_dir, indexgroup, indexname, "yea", f"{indexname}{suffix}.nc"
        )
    else:
        out_file = Path(
            out_dir, indexgroup, indexname, aggperiod, f"{indexname}{suffix}.nc"
        )
    out_file.parent.mkdir(exist_ok=True, parents=True)
    return out_file


def get_infile(varname: str, in_dir: Path, suffix: str = "") -> Path:
    """Helper function to construct the input filename.

    Args:
        varname (str): input variable name
        inpath (Path): path to parent dir of input file
        suffix (str, optional): Suffix to be added to the filename, in front of the file type.
            Defaults to "".

    Returns:
        Path: path to output file
    """
    # SPARTACUS default
    infile = Path(in_dir, f"{varname}{suffix}.nc")
    return infile


def _get_percentile_threshold_xda(
    params: list[str],
    in_dir_list: list[str],
    extra_kwargs: dict,
    aggperiod: str,
    out_dir: Path,
) -> tuple[Union[None, dict[xr.DataArray]], str]:
    """get percentile threshold xr.DataArray either from loading the precalculated data
    or by triggering calculating it

    Args:
        params (list[str]): input variable
        in_dir_list (list[str]): paths to input variable
        extra_kwargs (dict): kwargs for xclim and for determining threshold information
        aggperiod (str): resampling frequency
        out_dir (Path): output path to save thresholds to

    Raises:
        NotImplementedError: will get raised if multiple params are supplied. only
            implemented for univariate params

    Returns:
        Union[None, xr.DataArray]: Either None if no threshold is needed, or the
            threshold as xr.DataArray
        str: returns the percentile input kwarg as str
    """
    log = _get_logger()
    # check if percentile based thresholds ("thresh_per") are needed for calculation
    # if yes, then the extra if block is executed and the threshold calculated/loaded
    # if the kwarg contains doy percentiles ("tasmin_per", "tasmax_per"), then extra
    # data similarly be loaded on a doy basis
    for kwarg_key, kwarg_val in extra_kwargs.items():
        if kwarg_key in ["thresh_per", "tasmax_per", "tasmin_per", "pr_per"]:
            if len(params) > 1:
                raise NotImplementedError(
                    f"Percentile threshold calculation only implemented for univariate input, but got {len(params) = }"
                )
            else:
                param = params[0]
            # grab some variables and construct filepath
            _years = extra_kwargs.get("ref_period")
            ref_period = [int(year_) for year_ in _years.split("-")]
            pctl = int(kwarg_val.lstrip("q"))
            _thresh_path = Path(out_dir, "thresholds")
            _fname = f"thresholds_{param}_TFREQ_q{pctl}_{_years}.nc"
            if kwarg_key == "thresh_per":
                _fname = _fname.replace("TFREQ", aggperiod)
            else:
                _fname = _fname.replace("TFREQ", "doy")
            thresh_file = Path(_thresh_path, _fname)
            log.debug(
                f"Using {ref_period = } for {kwarg_key} percentile q{pctl} thresholds"
            )
            # load or calculate thresholds
            if thresh_file.exists():
                thresholds = xr.open_dataarray(thresh_file, engine="h5netcdf")
                log.debug("load percentile thresholds")
            else:
                log.debug("calculate percentile thresholds")
                thresh_file.parent.mkdir(exist_ok=True, parents=True)
                thresh_inputs_list = [
                    open_input_data(
                        input_vars=params,
                        input_paths=in_dir_list,
                        years=year_iter,
                        aggperiod=aggperiod,
                    )
                    for year_iter in range(ref_period[0], ref_period[1] + 1)
                ]
                # flatten input list
                # WARNING: this assumes univariate inputs. multivariate input will throw an error at xr.concat
                thresh_inputs_flat = [x for xs in thresh_inputs_list for x in xs]
                thres_input = xr.concat(thresh_inputs_flat, dim="time")[param]
                if kwarg_key == "thresh_per":
                    thresholds = calculate_aggperiod_percentile(
                        thres_input=thres_input,
                        aggperiod=aggperiod,
                        pctl=pctl,
                    )
                    if aggperiod == "sea":
                        thresholds = thresholds.sel(season=["DJF", "MAM", "JJA", "SON"])
                else:
                    thresholds = calculate_doy_percentile(
                        xda_daily=thres_input, kwarg_key=kwarg_key, pctl=pctl
                    )
                log.debug(f"save percentile thresholds to: {thresh_file}")
                thresholds.to_netcdf(thresh_file, engine="h5netcdf")
            return thresholds, kwarg_key
    return None, None
