"""Various utility functions used for the calculation of climate indices."""

import json
import logging
import math
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import pandas as pd
import tomli
import xarray as xr


def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log KeyboardInterrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    errorlog = _get_logger()
    errorlog.exception(
        "Uncaught exception: ", exc_info=(exc_type, exc_value, exc_traceback)
    )


def logger(outfile: str = None) -> logging.getLogger:
    """Defines a basic logger. By default, logs everything to a file and prints INFO additionally to stdout.

    Args:
        outfile (str): filename for logging output

    Returns:
        logging object
    """
    _ROOT_DIR_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    logger = logging.getLogger("climate_indices")
    logger.setLevel(logging.DEBUG)
    # logging to file
    if not outfile:
        outfile = f"{_ROOT_DIR_}/logs/climind.out"
        outfile_err = f"{_ROOT_DIR_}/logs/climind.err"
        Path(outfile).parent.mkdir(exist_ok=True, parents=True)
    filelogger = RotatingFileHandler(filename=outfile, maxBytes=10e5, backupCount=3)
    filelogger.setFormatter(
        logging.Formatter(
            "%(levelname)s: %(asctime)s> %(message)s", datefmt="%Y-%m-%d %H:%M:%Sz%z"
        )
    )
    filelogger_err = logging.FileHandler(filename=outfile_err, mode="w")
    filelogger_err.setFormatter(
        logging.Formatter(
            "%(levelname)s: %(asctime)s> %(message)s", datefmt="%Y-%m-%d %H:%M:%Sz%z"
        )
    )
    filelogger_err.setLevel(logging.ERROR)

    logger.addHandler(filelogger)
    logger.addHandler(filelogger_err)

    # logging to console
    consolelogger = logging.StreamHandler()
    consolelogger.setLevel(logging.INFO)
    consolelogger.setFormatter(
        logging.Formatter("%(asctime)s> %(message)s", datefmt="%H:%M")
    )
    logger.addHandler(consolelogger)

    logger.debug("initialise logging")
    sys.excepthook = log_uncaught_exceptions
    return logger


def _get_logger() -> logging.getLogger:
    """Convenience function to get the logging object defined from main.py

    Returns:
        logging.getLogger: Logger to log messages
    """
    log = logging.getLogger("climate_indices")
    return log


def _dict_to_recursive_namespace(dict_: dict) -> SimpleNamespace:
    """Convenience function to transform a dictionary into a recursive
    SimpleNamespace to allow for '.' syntax to access attributes.

    Args:
        dict_ (dict): dictionary which is to be transformed

    Returns:
        namespace (SimpleNamespace): dictionary transformed into a SimpleNamespace
    """
    return json.loads(
        json.dumps(dict_), object_hook=lambda item: SimpleNamespace(**item)
    )


def get_config(conf_file: str = "config.toml", out_type: str = "namespace") -> dict:
    """Load config params from toml config file.

    Args:
        conf_file (str, optional): Path to config file. Defaults to "config.toml".
        out_type (str, optional): Output type for config, either 'dict', or 'namespace'.
            'dict' returns the config as nested dictionary, while 'namespace' returns
            a nested SimpleNamespace, allowing for '.' syntax to acces attributes.
            Defaults to "namespace".

    Returns:
        dict: config dict
    """
    with open(conf_file, "rb") as file:
        config = tomli.load(file)
        if out_type == "namespace":
            config = _dict_to_recursive_namespace(dict_=config)
    return config


def get_input_vars_and_paths(
    input_variables: Union[str, list[str]], dict_paths: dict
) -> tuple[list[str], list[str]]:
    """Takes input variables and paths and handles correct type within lists.

    Args:
        input_variables (Union[str, list[str]]): input variables as str or within a list
        dict_paths (Union[dict, SimpleNamespace]): paths within a dict, or SimpleNamespace

    Returns:
        tuple[list[str], list[str]]: Returns the input variables and paths wrapped within
            a list
    """
    # if string, needs to be wrapped within a list
    if isinstance(input_variables, str):
        input_variables = [input_variables]
    if isinstance(dict_paths, SimpleNamespace):
        input_paths = [getattr(dict_paths, var_) for var_ in input_variables]
    else:
        input_paths = [dict_paths[var_] for var_ in input_variables]
    return input_variables, input_paths


def get_climate_indices_df_iterator(config: SimpleNamespace) -> pd.DataFrame:
    """Construct iterator for climate indices to be calculated.

    Args:
        config (SimpleNamespace): configuration SimpleNamespace containing the
            information which climate indices should be calculated

    Returns:
        inddf_iter (pd.DataFrame): returns the climate indices to be calculate in a
            queueried pd.DataFrame
    """
    _ROOT_DIR_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log = _get_logger()

    inddf = pd.read_csv(f"{_ROOT_DIR_}/doc/indices.csv")
    climind_list = config.CLIMATE_INDICES.IND_LIST
    if climind_list == "all":
        log.debug("Calculating all climate indices from doc/indices.csv")
        inddf_iter = inddf
    else:
        log.debug(f"Calculating selected list of climate indices: {climind_list}")
        inddf_iter = inddf.query(f"index in {climind_list}")
    return inddf_iter


def _try_kwarg_str_to_int(val_: str) -> Union[str, int]:
    """transforms input str to int if possible, otherwise return str

    Args:
        val_ (str): input arg

    Returns:
        Union[str, int]: transformed input
    """
    log = _get_logger()
    try:
        val_ = int(val_)
        log.debug(f"transformed kwarg value to int: {val_}")
    except ValueError:
        val_ = val_
        log.debug(f"no int transformation possible for kwarg value: {val_}")
    return val_


def _transform_kwargs_to_dict(kwargs_iter: str) -> dict:
    """Convenience function to transform str kwargs from indices.csv to dict.
    Multiple key-value pairs are separated by "+" and each pair separated by "=".

    Args:
        kwargs_iter (str): kwargs as str

    Raises:
        ValueError: Error is raised if kwargs is neither of type str, nor float

    Returns:
        dict: kwargs as dict
    """
    log = _get_logger()
    if isinstance(kwargs_iter, str):
        kwargs = {
            kwarg.split("=")[0]: _try_kwarg_str_to_int(
                kwarg.split("=")[1].replace("'", "").replace('"', "")
            )
            for kwarg in kwargs_iter.split("+")
        }
    elif isinstance(kwargs_iter, float):
        if math.isnan(kwargs_iter):
            log.debug(f"found nan {kwargs_iter = }, returning empty dict")
            kwargs = {}
    else:
        raise ValueError(f"Incompatible {kwargs_iter = } found. Please check input.")
    return kwargs


def get_inputs_from_climind_df(
    climind_df: pd.DataFrame,
) -> list[tuple[str, str, list[str], str, str, str, str, str]]:
    """Extracts the relevant input variables for the calculation of climate indicators
    from a pd.DataFrame based on the doc/indices.csv file.

    Args:
        climind_df (pd.DataFrame): DataFrame containing the climate indicators to be
            calculated

    Returns:
        tuple[list[str], list[str], list[list[str]], list[str], list[str], list[str], list[str], list[str]]: input variables to calculate
            climate indicators
    """
    name_list = climind_df["index"].to_list()
    group_list = climind_df["group"].to_list()
    # params are always wrapped within a list for each climate indicator
    # multiple input params need to be separated by a "+" character and are
    # split into a list here
    params_list = [param.split("+") for param in climind_df["parameter"].to_list()]
    descr_list = climind_df["description"].to_list()
    aggperiod_list = climind_df["aggperiod"].to_list()
    kwargs_list = climind_df["kwargs"].to_list()
    xclim_func_list = climind_df["xclim_func"].to_list()
    units_list = climind_df["units"].to_list()

    # aggperiod_list can contain information for yearly and seasonal calculation
    # this entails calling subsequent routines separately for each resampling freq
    # therefore, we have to expand the output list to separate those as individual entries
    climind_params = []
    for (
        name,
        group,
        params,
        descr,
        aggperiod,
        kwargs_iter,
        xclim,
        unit,
    ) in zip(
        name_list,
        group_list,
        params_list,
        descr_list,
        aggperiod_list,
        kwargs_list,
        xclim_func_list,
        units_list,
    ):
        # kwargs contain extra keyword arguments needed to calculate some climate
        # indicators. they are given as strings, which means we need to transform
        # them into proper a proper dictionary
        kwargs = _transform_kwargs_to_dict(kwargs_iter=kwargs_iter)
        for aggperiod_iter in aggperiod.split("+"):
            climind_params.append(
                (name, group, params, descr, aggperiod_iter, kwargs, xclim, unit)
            )
    return climind_params


def _sanity_check_varname(xda: xr.DataArray) -> xr.DataArray:
    """Check and rename xr.DataArray name to convention

    Args:
        xda (xr.DataArray): xr.DataArray to be checked and renamed

    Returns:
        xr.DataArray: renamed xr.DataArray
    """
    dict_varname_to_convention = {
        "TN": "tasmin",
        "TX": "tasmax",
        "TG": "tas",
        "RR": "pr",
        "snow_depth": "snd",
        "snow_depth_fresh": "snd",
    }
    if xda.name in [*dict_varname_to_convention]:
        xda = xda.rename(dict_varname_to_convention[xda.name])
    return xda
