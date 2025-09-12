#!/usr/bin/env -S uv run

"""Technical validation script to check number of expected valid vs existing files and validity in terms of NaNs and filesize"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

import numpy as np
import xarray as xr

from climind_io import get_outfile
from utils import (
    get_climate_indices_df_iterator,
    get_config,
    get_input_vars_and_paths,
    get_inputs_from_climind_df,
    logger,
)

if __name__ == "__main__":
    config = get_config(conf_file="config.toml")

    log = logger()
    log.info("Starting technical validation")

    # get some variables from config file
    year_start = config.GENERAL.YEAR_START
    year_end = config.GENERAL.YEAR_END
    out_dir = Path(config.PATHS.OUT)
    epsg = config.CRS.EPSG

    # construct iterator for climate indices to be calculated
    inddf_iter = get_climate_indices_df_iterator(config)
    clim_ind_inputs = get_inputs_from_climind_df(inddf_iter)
    # loop over climate indices dataframe to get all info for calculation
    collected_errors = []
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
        log.info(
            f"Progress {idx + 1}/{len(clim_ind_inputs)} {group} {index} {aggperiod}"
        )
        input_vars, input_paths = get_input_vars_and_paths(input_vars, config.PATHS)
        log.info("Checking file validity > 100 KiloBytes")
        expected_files_count = 1 + year_end - year_start
        log.info(f"Number of valid files expected: {expected_files_count}")
        outfile = get_outfile(
            indexname=index,
            indexgroup=group,
            out_dir=out_dir,
            suffix=year_start,
            aggperiod=aggperiod,
        )

        valid_file_count = 0
        invalid_files = []
        for file_ in outfile.parent.glob("*.nc"):
            file_size_bytes = file_.stat().st_size
            if file_size_bytes > 1024 * 100:
                valid_file_count += 1
            else:
                invalid_files.append(file_.name)
        log.info(f"Number of valid files found: {valid_file_count}")
        if expected_files_count != valid_file_count:
            collected_errors.append(
                f"\t{group} {index} {aggperiod}:\texpected {expected_files_count - valid_file_count} more files"
            )
        if invalid_files != []:
            log.info("Invalid files:")
            for ifile in invalid_files:
                log.info(f"\t{ifile}")

        # check for files with only nans
        nan_files = []
        nan_file_counter = 0
        for file_ in outfile.parent.glob("*.nc"):
            xda = xr.open_dataset(file_)[index]
            data = xda.values.ravel()
            data_notnan = data[~np.isnan(data)]
            if len(data_notnan) == 0:
                nan_files.append(file_.name)
                nan_file_counter += 1
        if nan_file_counter > 0:
            collected_errors.append(
                f"\t{group} {index} {aggperiod}:\tFound {nan_file_counter} files containing only NaN values"
            )
        if nan_files != []:
            log.info("NaN files:")
            for ifile in nan_files:
                log.info(f"\t{ifile}")

    log.info("---------------------")
    log.info("TECHNICAL VALIDATION SUMMARY")
    if len(collected_errors) != 0:
        for error in collected_errors:
            log.info(error)
    else:
        log.info(
            "All expected files exist, contain valid values and meet filesize threshold!"
        )
