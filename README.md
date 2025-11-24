# Climate indicators for Austria since 1961 at 1 km resolution

<!-- badges: start -->
<!-- [![DOI paper](https://zenodo.org/badge/DOI/xyz/)]() -->
[![DOI data](https://zenodo.org/badge/DOI/10.5281/zenodo.16928609.svg)](https://doi.org/10.5281/zenodo.16928609)
[![Python code style: black](https://img.shields.io/badge/codestyle-black-black)](https://github.com/psf/black)
<!-- badges: end -->

This repository supplements the manuscript by
Sebastian Lehner <sup>[![](https://info.orcid.org/wp-content/uploads/2020/12/orcid_16x16.gif)](https://orcid.org/0000-0002-7562-8172)</sup>
and
Matthias Schlögl
<sup>[![](https://info.orcid.org/wp-content/uploads/2020/12/orcid_16x16.gif)](https://orcid.org/0000-0002-4357-523X)</sup>:
**Climate indicators for Austria since 1961 at 1 km resolution**.

This repository provides a toolkit to compute climate indicators for applications in climate impact research.
The implementation leverages [xclim](https://xclim.readthedocs.io/en/stable/) and self-implemented algorithms for index computation.
`xclim` is a library on top of `xarray` for the calculation of climate indicators.

Furthermore, this repository contains default configs for the following meteorological variables and their corresponding data source:

- Temperature: https://doi.org/10.60669/m6w8-s545
- Precipitation: https://doi.org/10.60669/m6w8-s545
- Evapotranspiration: https://doi.org/10.60669/f6ed-2p24
- Sunshine duration: https://doi.org/10.60669/m6w8-s545
- Snow and runoff: https://doi.org/10.60669/fsxx-6977﻿

The routines are flexible and the source of the data can be configured to any data readable with `xarray`, by changing the configuration parameters in `config.toml`. Note though, that for different datasets, some adaptations to deal with differently structured data likely has to be done.


## Usage

This project has been set up with [uv](https://docs.astral.sh/uv). After cloning this repository, start with `uv sync` to install the dependencies into a local `.venv` environment (if local `quota` is an issue, you can also symlink `.venv` to a different location (e.g. your `/perm`, or `/scratch`); `uv` will still work correctly). Then, `uv run src/main.py`, or better, `uv run pytask` (see Automation below) can be used to execute the calculation of climate indicators. Note that all the configuration is done within `config.toml`.

In order to quickly verify if the calculation of indices was succesful, use `uv run src/check_files.py`. Thereby, based on the configuration within `config.toml`, the number of existing files in the output path is checked against the expected number depending on the years specified. As validity criteria, a filesize of 100 KiloBytes is used. Note that no checks on the data itself is being done.


## Automation

This repository includes automation by using [pytask](https://pytask-dev.readthedocs.io/en/stable/), which is installed automatically by the given environment via `uv`. Simply run `uv run pytask` to execute all automated tasks. Some `pytask` configuration can be found and adapted within the `pyproject.toml` file. For parallel processing, either update the config file, or use `uv run pytask -n NUM_WORKERS`. Note that `pytask` by default captures all `stdout` and `stderr`, which means that it will not properly be logged within the logfiles. To prevent this and log all output correctly to the logfile, use the `-s` option: e.g. `uv run pytask -s`.


## Climate indicators

Climate indicators can be calculated by using the defined abbrevitation from the `index` column in [doc/indices.csv](doc/indices.csv), within the `config.toml` file in the `[CLIMATE_INDICES]` section to calculate only a subset of climate indicators. Otherwise, all climate indicators listed in `doc/indices.csv` are calculated. 

The list of available climate indicators contains [Climdex](https://www.climdex.org/learn/indices/), [BIOCLIM](https://www.worldclim.org/data/bioclim.html) and various other indicators (full list in [doc/indices.csv](doc/indices.csv)).


## Aggregation, Evaluation and Visualisation

Aggregation and evaluation can be run by `uv run src/climind_eval.py`. This script also takes the configuration from the `config.toml` file for which indicators it should be executed. There are 3 steps that are executed: 1) calculation of spatial averages, 2) calculation of climatological means (for the two periods 1961 to 1990 and 1991 to 2020), and 3) a two-tailed Mann-Whitney U-test for statistical significant changes between the two climatological periods. The calculated data is saved into the output directory specified in `config.toml`.

Visualisation intakes the calculated climate indicators and/or the aggregation/evaluation data depending on the plots. There following visualisations are implemented and can be explored in this [showcase document](doc/visualisation_showcase.md): 1) time series of spatial averaged anomalies for annual/seasonal aggregations, 2) spatial maps of the climatologies and their difference, 3) stampplots and anomaly stampplots of the climate indicators, 4) grouped significant changes plots, and 5) grouped warming stripes. The groupings for 4) and 5) are based on the base input parameter for each indicator (see the linked showcase above for more details).
