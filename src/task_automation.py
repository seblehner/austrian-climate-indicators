"""Automation of tasks via pytask. Runs all climate indices based on config.toml."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from itertools import product
from pathlib import Path
from typing import Annotated, NamedTuple

from pytask import DataCatalog, Product, task

from climind_io import get_outfile
from main import wrapper_calc_climate_indices
from utils import (
    get_climate_indices_df_iterator,
    get_config,
    get_inputs_from_climind_df,
    logger,
)

config = get_config(conf_file="config.toml")
log = logger()
clim_ind_df = get_climate_indices_df_iterator(config)
data_catalog = DataCatalog()


class ClimateIndices(NamedTuple):
    name: str
    group: str
    year: str
    params: list[str]
    in_dirs: list[str]
    descr: str
    out_file: Path
    epsg: str
    aggperiod: str
    extra_kwargs: str
    xclim_func: str
    unit: str

    @property
    def task_name(self) -> str:
        return f"{self.name}-{self.aggperiod}-{self.year}"


# get all needed input variables for clim ind calculation
outpath = Path(config.PATHS.OUT)
epsg = str(config.CRS.EPSG)
years = list(range(config.GENERAL.YEAR_START, config.GENERAL.YEAR_END + 1))
climinds_inout = get_inputs_from_climind_df(clim_ind_df)

# construct list of all clim inds to be calculated
all_climinds = [
    ClimateIndices(
        name,
        group,
        year,
        param,
        [getattr(config.PATHS, input_var) for input_var in param],
        descr,
        get_outfile(
            indexname=name,
            indexgroup=group,
            out_dir=outpath,
            suffix=year,
            aggperiod=aggperiod,
        ),
        epsg,
        aggperiod,
        extra_kwargs,
        xclim_func,
        unit,
    )
    for (
        name,
        group,
        param,
        descr,
        aggperiod,
        extra_kwargs,
        xclim_func,
        unit,
    ), year in product(climinds_inout, years)
]

for run in all_climinds:
    log.debug(f"Executing task: {run = }")

    @task(kwargs=run)
    def task_calc_climate_indices(
        name: str,
        group: str,
        year: int,
        params: list[str],
        in_dirs: list[str],
        descr: str,
        out_file: Annotated[Path, Product],
        epsg: str,
        aggperiod: str,
        extra_kwargs: str,
        xclim_func: str,
    ) -> Annotated[Path, data_catalog[run.task_name]]:
        wrapper_calc_climate_indices(
            index=name,
            group=group,
            descr=descr,
            params=params,
            in_dir_list=in_dirs,
            year=year,
            out_dir=out_file.parent.parent.parent.parent,
            epsg=epsg,
            aggperiod=aggperiod,
            extra_kwargs=extra_kwargs,
            xclim_func=xclim_func,
        )
