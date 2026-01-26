# High-resolution climate indicators for gridded climate data in Austria

<!-- badges -->
<p align="center">
    <a href="https://seblehner.github.io/austrian-climate-indicators">
        <img alt="GitHub Pages" src="https://img.shields.io/badge/GitHub-Pages-blue"></a>
    <a href="https://doi.org/10.5281/zenodo.16928609">
        <img alt="Zenodo data doi" src="https://img.shields.io/badge/DOI-10.5281/zenodo.16928609-blue"></a>
    <a href="https://github.com/psf/black">
        <img alt="Python code style: black" src="https://img.shields.io/badge/codestyle-black-000000.svg"></a>
</p>
<!-- badges -->

This repository supplements the manuscript by
Sebastian Lehner <sup>[![](https://info.orcid.org/wp-content/uploads/2020/12/orcid_16x16.gif)](https://orcid.org/0000-0002-7562-8172)</sup>
and
Matthias Schlögl
<sup>[![](https://info.orcid.org/wp-content/uploads/2020/12/orcid_16x16.gif)](https://orcid.org/0000-0002-4357-523X)</sup>:
**Climate indicators for Austria since 1961 at 1 km resolution**.


> [!NOTE]
> In a nutshell:
> - A comprehensive toolkit to compute, aggregate, and visualize climate indicators for climate impact applications.
> - Built on [`xclim`](https://xclim.readthedocs.io/en/stable/) and [`xarray`](https://docs.xarray.dev/en/stable/index.html) for index computation, with task automation via [`pytask`](https://pytask-dev.readthedocs.io/en/stable/).
> - Works out of the box with defaults for gridded climate data sets from Austria (SPARTACUS, WINFORE and SNOWGRID-CL), but can be configured for any dataset readable by `xarray`.


## Input data sets

Default configs for the following meteorological variables and their corresponding data source are available:

- Temperature: https://doi.org/10.60669/m6w8-s545
- Precipitation: https://doi.org/10.60669/m6w8-s545
- Evapotranspiration: https://doi.org/10.60669/f6ed-2p24
- Sunshine duration: https://doi.org/10.60669/m6w8-s545
- Snow and runoff: https://doi.org/10.60669/fsxx-6977

Different datasets are untested and may require adaptations due to a different input data structure (e.g., variable naming, calendar, units, chunking).


## Quick start

> [!IMPORTANT]
> This project leverages [uv](https://docs.astral.sh/uv) for package and project management.

1. Clone the repository
2. Install dependencies using uv (creates a local `.venv`):
   ```bash
   uv sync
   ```
3. Adapt the configuration in `config.toml`.
4. Run the workflow:
   ```bash
   uv run pytask
   ```
   or run individual steps manually.


### Output validation

> [!TIP]
> Run the validation task with
> `uv run src/check_files.py`

This performs the following checks:
- Compare the actual output file counts in the output path versus the expected number depending on the years specified in the `config.toml`.
- Test whether the minimum output file size exceeds a threshold of 100 KB.
- Check for files that only contain NaN values.
- Note: This does not validate data values in any other ways except for the blank NaN check.


### Aggregation and evaluation

> [!TIP]
> Run the aggregation and evaluation task:
> `uv run src/climind_eval.py`

This computes:
1. Spatial averages;
2. Climatological means for 1961–1990 and 1991–2020;
3. Two-tailed Mann–Whitney U-tests for significant changes between the two climatological fields.

Outputs are stored in the directory specified in `config.toml`.


## Automation

> [!IMPORTANT]
> This project leverages [pytask](https://pytask-dev.readthedocs.io/en/stable/) for workflow management.

- `pytask` is included in the environment and installed via `uv`.
- Simply run `uv run pytask` to execute all automated tasks.
- The `pytask` configuration can be adapted through the `pyproject.toml` file.
- For parallel processing, either update the config file, or use `uv run pytask -n NUM_WORKERS`.
- By default, `pytask` captures all `stdout` and `stderr`, which means that it will not properly be logged within the logfiles. To stream logs to console and log files (recommended), use the `-s` option, i.e. `uv run pytask -s`.

## Configuration

> [!NOTE]
> All configuration settings are controlled via `config.toml`.

Typical settings:
- Input dataset location and variable names
- Years or time range
- Output directory
- Specification of indicators to compute

To compute only selected indicators, use the abbreviations (as listed in the `index` column in [doc/indices.csv](doc/indices.csv)) in the `[CLIMATE_INDICES]` section of `config.toml`. Default is `IND_LIST = "all"`, which computes all indicators listed in `doc/indices.csv`.


## Climate indicators

- The list of all currently implemented indicators is available as [doc/indices.csv](doc/indices.csv)
- The list comprises [Climdex](https://www.climdex.org/learn/indices/), [BIOCLIM](https://www.worldclim.org/data/bioclim.html) and additional indices from [`xclim`](https://xclim.readthedocs.io/en/stable/), as well as custom routines.


## Visualisation

Visualizations use either the computed indicators or the aggregated/evaluated outputs. The following visualizations are implemented and can be explored in the [showcase document](doc/visualisation_showcase.md):

1. Time series of spatially averaged anomalies (annual/seasonal)
2. Spatial maps of climatologies and their differences
3. Stamp plots and anomaly stamp plots
4. Grouped significant changes plots
5. Grouped warming stripes
6. Year-of-minimum/maximum per grid cell and histograms over time

Groupings for plots (4) and (5) are based on each indicator’s input parameters (details in the showcase).
