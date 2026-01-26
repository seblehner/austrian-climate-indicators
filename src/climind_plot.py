#!/usr/bin/env -S uv run
"""Plotting routines to visualise climate indicators
1. "warming stripes" 2d vis indicator areamean vs year
3. spatial significant change % of ensemble (ensemble = indicator category; e.g. temperature)
4. stampplots/spatial plots, mean year plots, ...
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from typing import Union

import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import ticker

from utils import (
    get_climate_indices_df_iterator,
    get_config,
    get_inputs_from_climind_df,
    logger,
)


def configure_font():
    # Configure default plotting font to 'Source Sans 3' (preferred) with a
    # graceful fallback. If the font is available in the system font list it is
    # prepended to the sans-serif rcParam so matplotlib will use it for plots.
    # If the font is not found a warning is emitted but execution continues and
    # matplotlib will fall back to the next available sans-serif font.
    # source: https://fonts.adobe.com/fonts/source-sans-3
    _climind_preferred_font = "Source Sans 3"
    # Note: the following 2 lines manually add for user-installed fonts
    # in ~/.local/share/fonts in case they are not found in the system
    for fontfile in Path(Path.home(), ".local/share/fonts/").glob("*.ttf"):
        font_manager.fontManager.addfont(str(fontfile))
    _available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if _climind_preferred_font in _available_fonts:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [
            _climind_preferred_font,
        ] + matplotlib.rcParams.get("font.sans-serif", [])
    else:
        # still prepend so if the font gets installed later it will be preferred
        matplotlib.rcParams["font.sans-serif"] = [
            _climind_preferred_font,
        ] + matplotlib.rcParams.get("font.sans-serif", [])

        log.warning(
            f"Font '{_climind_preferred_font}' not found in system fonts. Plots will use a fallback sans-serif font until it's installed."
        )
    return None


# projection for EPSG:3416
cartopy_proj = {
    "projection": cartopy.crs.LambertConformal(
        central_longitude=13.3333333,
        central_latitude=47.5,
        false_easting=400000,
        false_northing=400000,
        standard_parallels=(49, 46),
    )
}


def get_eval_outfile_path_group(
    str_: str, out_dir: Path, group: str, index: str, aggperiod: str
) -> Path:
    """returns output file for given kwargs

    Args:
        str_ (str): suffix for output data
        out_dir (Path): parent path to data
        group (str): category group for climate indicators
        index (str): abbreviated name of climate indicator
        aggperiod (str): aggregation period of climate indicator

    Returns:
        Path: Path object to output file
    """

    outpath_ind = Path(out_dir, group, index, aggperiod)
    outfile_ = Path(
        str(outpath_ind).replace("indicators", f"indicators_{str_}"),
        f"{index}_{aggperiod}_{str_}.nc",
    )
    outfile_.parent.mkdir(exist_ok=True, parents=True)
    return outfile_


def ax_styling(ax_: matplotlib.axes.Axes, xlims: tuple[float]) -> None:
    """some ax styling for the given matplotlib axis

    Args:
        ax_ (matplotlib.axes.Axes): matplitlib axis object that is modified
        xlims (tuple[float]): limits for x-axis
    """
    ax_.grid(lw=0.3)
    ax_.set_xlim(xlims)
    return None


def save_plot(fig_: matplotlib.figure.Figure, savepath: str) -> None:
    """saves current plot with tight layout

    Args:
        fig_ (matplotlib.figure.Figure): figure handle of matplotlib plot
        savepath (str): output path to save plot to
    """
    fig_.tight_layout()
    plt.savefig(savepath)
    return None


def calc_anoms_refs(
    xda: xr.DataArray, years_past: tuple[int], years_now: tuple[int]
) -> tuple[xr.DataArray, float, float]:
    """calculated anomalies based on reference

    Args:
        xda (xr.DataArray): data for which anomalies are calculated
        years_past (tuple[int]): years for the reference period
        years_now (tuple[int]): years for the second period as additional information

    Returns:
        tuple[xr.DataArray, float, float]: anomaly data and reference values
    """
    ref_past = xda.sel(time=slice(str(years_past[0]), str(years_past[1]))).mean().values
    ref_now = xda.sel(time=slice(str(years_now[0]), str(years_now[1]))).mean().values
    anom = xda - ref_past
    return anom, ref_past, ref_now


def get_xlims(xvals: np.array) -> tuple[float, float]:
    """derive xlims for plotting based on the xvalues of the data

    Args:
        xvals (np.array): x-values of the data

    Returns:
        tuple[float, float]: lower and upper xlim values
    """
    dxvals = xvals[1] - xvals[0]
    return xvals.min() - dxvals, xvals.max() + dxvals


def plot_lines_with_fill(
    xda: xr.DataArray,
    xvals: np.array,
    ax: matplotlib.axes.Axes,
    color_above: str = "C3",
    color_below: str = "C0",
) -> None:
    """plot lines with area fills between values and 0

    Args:
        xda (xr.DataArray): data to be plotted
        xvals (np.array): x-values for the corresponding data
        ax (matplotlib.axes.Axes): axis for plot
    """
    xda.plot(ax=ax, color="k", lw=1)

    # area fills
    plotdata = xda.values
    ax.fill_between(
        xvals,
        0,
        plotdata,
        where=(plotdata > 0),
        color=color_above,
        interpolate=True,
        alpha=0.5,
    )
    ax.fill_between(
        xvals,
        0,
        plotdata,
        where=(plotdata < 0),
        color=color_below,
        interpolate=True,
        alpha=0.5,
    )
    return None


def plot_ref_hlines(
    xrefs: list[tuple[np.timedelta64]],
    yrefs: list[tuple[np.timedelta64]],
    labels: str,
    colors: str,
    ax: matplotlib.axes.Axes,
    ls: str,
) -> None:
    """plot horizontal reference lines for climatological values

    Args:
        xrefs (list[tuple[np.timedelta64]]): x values for reference
        yrefs (list[tuple[np.timedelta64]]): y values for reference
        labels (str): label description for the reference
        colors (str): colour string
        ax (matplotlib.axes.Axes): ax to plot data on
        ls (str): linestyle for horizontal reference lines
    """
    for xref, yref, label, color, ls_ in zip(xrefs, yrefs, labels, colors, ls):
        ax.hlines(
            yref,
            xref[0],
            xref[1],
            color=color,
            lw=4,
            ls=ls_,
            label=label,
            capstyle="round",
        )
    return None


def plot_anomaly_timeseries_year(
    xda: xr.DataArray,
    years_past: tuple[int],
    years_now: tuple[int],
    savefile: str = "test.png",
    color_kwargs: dict = {"color_above": "C3", "color_below": "C0"},
) -> None:
    """plotting routine for anomaly timeseries of annual data

    Args:
        xda (xr.DataArray): annual data to be plotted
        years_past (tuple[int]): start and end year for past reference
        years_now (tuple[int]): start and end year for second, more recent reference
        savefile (str, optional): output file for saved plot. Defaults to "test.png".
    """
    unit_ = xda.attrs["units"]

    anom, past, now = calc_anoms_refs(xda, years_past, years_now)
    xvals = anom.time.values

    fig, ax = plt.subplots(figsize=(7, 4))

    plot_lines_with_fill(anom, xvals, ax, **color_kwargs)

    # ref lines
    time1 = xda.sel(time=slice(str(years_past[0]), str(years_past[1]))).time.values
    time2 = xda.sel(time=slice(str(years_now[0]), str(years_now[1]))).time.values
    xref1 = time1.min(), time1.max()
    xref2 = time2.min(), time2.max()
    plot_ref_hlines(
        xrefs=[xref1, xref2],
        yrefs=[0, now - past],
        labels=[f"1961–1990: {past:.2f} {unit_}", f"1991–2020: {now:.2f} {unit_}"],
        colors=["k", "k"],
        ls=["-", ":"],
        ax=ax,
    )

    # labeling
    plt.legend(title="Annual mean", loc="upper left")
    plt.title(f"{xda.name} annual anomalies areamean")
    ax.set_ylabel(f"{xda.name} [{unit_}]")
    ax.set_xlabel("")

    # styling
    ax_styling(ax, get_xlims(xvals))

    # save plot
    save_plot(fig, savepath=savefile)
    plt.close()
    return None


def plot_anomaly_timeseries_season(
    xda: xr.DataArray,
    years_past: tuple[int],
    years_now: tuple[int],
    savefile: str = "test.png",
    color_kwargs: dict = {"color_above": "C3", "color_below": "C0"},
) -> None:
    """plotting routine for anomaly timeseries of seasonal data

    Args:
        xda (xr.DataArray): seasonal data to be plotted
        years_past (tuple[int]): start and end year for past reference
        years_now (tuple[int]): start and end year for second, more recent reference
        savefile (str, optional): output file for saved plot. Defaults to "test.png".
    """
    unit_ = xda.attrs["units"]
    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(14, 8))
    for ii, ((season, xdaseas), axit) in enumerate(
        zip(xda.groupby("time.season"), ax.flat)
    ):
        anom, past, now = calc_anoms_refs(xdaseas, years_past, years_now)
        xvals = anom.time.values

        plot_lines_with_fill(anom, xvals, axit, **color_kwargs)

        # ref lines
        time1 = xdaseas.sel(
            time=slice(str(years_past[0]), str(years_past[1]))
        ).time.values
        time2 = xdaseas.sel(
            time=slice(str(years_now[0]), str(years_now[1]))
        ).time.values
        xref1 = time1.min(), time1.max()
        xref2 = time2.min(), time2.max()
        plot_ref_hlines(
            xrefs=[xref1, xref2],
            yrefs=[0, now - past],
            labels=[
                f"1961–1990: {past:.2f} {unit_}",
                f"1991–2020: {now:.2f} {unit_}",
            ],
            colors=["k", "k"],
            ls=["-", ":"],
            ax=axit,
        )

        # labeling
        axit.legend(title=f"{season} mean", loc="upper left")
        plt.suptitle(f"{xda.name} seasonal anomalies areamean (reference: 1961–1990)")
        axit.set_title("")
        if ii % 2 == 0:
            axit.set_ylabel(f"{xda.name} [{unit_}]")
        else:
            axit.set_ylabel("")
        axit.set_xlabel("")

        # styling
        ax_styling(axit, get_xlims(xvals))

    # save plot
    save_plot(fig, savepath=savefile)
    plt.close()
    return None


def add_cartopy_styling(
    ax_: matplotlib.axes.Axes, gridlines: bool = True, drawlabel: bool = True
) -> matplotlib.lines.Line2D:
    """add cartopy specific styling for spatial map plots

    Args:
        ax_ (matplotlib.axes.Axes): axis to modify
        gridlines (bool, optional): draw gridlines. Defaults to True.
        drawlabel (bool, optional): draw labels. Defaults to True.

    Returns:
        matplotlib.lines.Line2D: gridlines object
    """
    gl = None
    ax_.add_feature(cartopy.feature.LAND)
    ax_.add_feature(cartopy.feature.OCEAN)
    ax_.add_feature(cartopy.feature.COASTLINE)
    ax_.add_feature(cartopy.feature.BORDERS, linestyle="-")
    if gridlines:
        gl = ax_.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=drawlabel,
            linewidth=1,
            color="gray",
            alpha=0.5,
            linestyle="--",
            x_inline=False,
            rotate_labels=False,
        )
    ax_.set_extent([9.3, 17.3, 46, 49.2])
    return gl


def plot_spatial_map_comparison(
    xda_ref1: xr.DataArray,
    xda_ref2: xr.DataArray,
    xda_anom: xr.DataArray,
    epsg: int = None,
    savefile: str = "test.png",
    titlestr: str = "",
    vmin: float = None,
    vmax: float = None,
    cmap_div_anom: str = "RdBu_r",
    cmap_div_clim: str = "PuOr_r",
) -> None:
    """plot spatial maps comparison. this creates 3 plots in one row, where
    the first two plots are climatologial fields and the third is the difference
    from the second to the first.

    Args:
        xda_ref1 (xr.DataArray): past spatial data for the first plot
        xda_ref2 (xr.DataArray): recent spatial data for the second plot
        xda_anom (xr.DataArray): difference spatial data for the third plot
        epsg (int, optional): epsg code for the data. Defaults to None.
        savefile (str, optional): path to save plot to. Defaults to "test.png".
        titlestr (str, optional): title string for the plot. Defaults to "".
        vmin (float, optional): vmin for the colorbar. Defaults to None.
        vmax (float, optional): vmax for the colorbar. Defaults to None.
    """
    fig, axes = plt.subplots(
        ncols=3,
        figsize=(15, 4.5),
        sharey=True,
        sharex=True,
        layout="constrained",
        subplot_kw=cartopy_proj,
    )
    plots = []
    for idx, (ax, xdaiter, subplotnum) in enumerate(
        zip(iter(axes), [xda_ref1, xda_ref2, xda_anom], ["a)", "b)", "c)"])
    ):
        gl = add_cartopy_styling(ax)

        if idx == 0:
            gl.right_labels = False
        elif idx == 1:
            gl.left_labels = False
            gl.right_labels = False
        elif idx == 2:
            gl.left_labels = False
        # gl.top_labels = False
        if epsg:
            datplot = xdaiter.rio.write_crs(f"EPSG:{epsg}").rio.reproject("EPSG:3416")
        else:
            datplot = xdaiter

        if vmin < 0 and vmax > 0:
            cmap = cmap_div_clim
        else:
            cmap = "viridis"
        if idx == 2:
            p = datplot.plot(
                ax=ax,
                cbar_kwargs={
                    "orientation": "horizontal",
                    "label": f"{datplot.name} anomaly [{datplot.attrs['units']}]",
                },
                cmap=cmap_div_anom,
                center=0,
            )
        else:
            p = datplot.plot(
                ax=ax,
                add_colorbar=False,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        plots.append(p)
        if idx == 0:
            ax.set_title("Past: 1961–1990")
        elif idx == 1:
            ax.set_title("Recent: 1991–2020")
        elif idx == 2:
            ax.set_title("Anomaly: Recent - Past")

        ax.text(
            0.013,
            0.905,
            subplotnum,
            transform=ax.transAxes,
            fontsize="x-large",
            fontweight="bold",
            va="bottom",
            bbox={"facecolor": "w"},
        )

    fig.colorbar(
        plots[0],
        ax=axes.ravel().tolist()[:2],
        orientation="horizontal",
        label=f"{datplot.name} [{datplot.attrs['units']}]",
        shrink=0.48,
    )
    plt.suptitle(titlestr)
    plt.savefig(savefile, bbox_inches="tight")
    plt.close()
    return None


def plot_spatial_map(
    xda: xr.DataArray,
    epsg: int = None,
    savefile: str = "test.png",
    titlestr: str = "",
    vmin: float = None,
    vmax: float = None,
    cmap_div: str = "RdBu_r",
) -> None:
    """plot spatial map

    Args:
        xda (xr.DataArray): spatial data to be plotted
        savefile (str, optional): path to save plot to. Defaults to "test.png".
        titlestr (str, optional): title string for the plot. Defaults to "".
        vmin (float, optional): vmin for the colorbar. Defaults to None.
        vmax (float, optional): vmax for the colorbar. Defaults to None.
    """
    ax = plt.axes(**cartopy_proj)
    add_cartopy_styling(ax)

    if epsg:
        datplot = xda.rio.write_crs(f"EPSG:{epsg}").rio.reproject("EPSG:3416")
    else:
        datplot = xda

    if vmin is None:
        vmin = float(xda.min().values)
    if vmax is None:
        vmax = float(xda.max().values)

    if vmin < 0 and vmax > 0:
        cmap = cmap_div
    else:
        cmap = "viridis"

    datplot.plot(
        ax=ax,
        cbar_kwargs={
            "orientation": "horizontal",
            "label": f"{xda.name} [{xda.attrs['units']}]",
        },
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    plt.title(titlestr)
    plt.savefig(savefile, bbox_inches="tight")
    plt.close()
    return None


def stampplot_time(
    xda: xr.DataArray,
    vmin: float = None,
    vmax: float = None,
    savename: str = "test.png",
    title_suffix: str = "",
    cmap_div: str = "RdBu_r",
) -> None:
    """creates a stampplot over the time dimension for the input data

    Args:
        xda (xr.DataArray): data to be plotted
        vmin (float, optional): vmin for the colorbar. Defaults to None.
        vmax (float, optional): vmax for the colorbar. Defaults to None.
        savename (str, optional): path to save plot to. Defaults to "test.png".
        title_suffix (str, optional): title string for the plot. Defaults to "".
    """
    fontsize = 14
    dim_len = len(xda.time)
    log.debug(f"Creating a stampplot for {dim_len = }")
    fig, axes = plt.subplots(
        ncols=10,
        nrows=7,
        figsize=(25, 10),
        sharey=True,
        sharex=True,
        layout="constrained",
        subplot_kw=cartopy_proj,
    )
    fg_color = "k"
    bg_color = "w"
    axes = axes.ravel()
    axit = iter(axes)
    # first ax is used for general information and colorbar
    axinit = next(axit)
    for timestep in xda.time:
        xdaiter = xda.sel(time=timestep)
        ax = next(axit)
        if vmin < 0 and vmax > 0:
            cmap = cmap_div
        else:
            cmap = "viridis"
        add_cartopy_styling(ax, gridlines=False)
        p = xdaiter.plot(
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            add_colorbar=False,
        )
        ax.annotate(
            str(timestep.dt.year.values),
            (0, 1),
            xytext=(4, -4),
            xycoords="axes fraction",
            textcoords="offset points",
            color=fg_color,
            bbox=dict(edgecolor=fg_color, facecolor=bg_color),
            ha="left",
            va="top",
            fontsize=fontsize,
        )
        ax.set_aspect("equal")
        ax.set_title("")
    axinit.set_visible(False)
    cbar_ax = fig.add_axes([0.01, 0.93, 0.08, 0.02])
    cb = fig.colorbar(
        p,
        cax=cbar_ax,
        orientation="horizontal",
    )
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label(f"[{xda.attrs['units']}]", size=fontsize)
    cb.ax.set_title(f"{xda.name}:{title_suffix}", fontsize=fontsize + 2)
    cb.ax.xaxis.set_tick_params(color=fg_color)
    cb.outline.set_edgecolor(fg_color)
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color=fg_color)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cb.locator = tick_locator
    cb.update_ticks()
    for ax in axes[dim_len + 1 :]:
        ax.set_visible(False)
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels("")
        ax.set_yticklabels("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["bottom"].set_color(fg_color)
        ax.spines["top"].set_color(fg_color)
        ax.spines["right"].set_color(fg_color)
        ax.spines["left"].set_color(fg_color)
        ax.tick_params(axis="x", colors=fg_color, which="both")
        ax.tick_params(axis="y", colors=fg_color, which="both")
        ax.yaxis.label.set_color(fg_color)
        ax.xaxis.label.set_color(fg_color)
    fig.set_facecolor(bg_color)
    plt.savefig(savename)
    plt.close()
    return None


def get_vmin_vmax(
    arr: Union[list, xr.DataArray], type_: str = "99p"
) -> tuple[float, float]:
    """derive vmin and vmax from input array. either as min/max values,
    or as the 1st and 99th percentile of the data.

    Args:
        arr (Union[list, xr.DataArray]): data for which vmin/vmax is calculated
        type_ (str, optional): which method to use ("99p", or "minmax"). Defaults to "99p".

    Raises:
        ValueError: raises an error if neither of the two allowed type_ is specified

    Returns:
        tuple[float, float]: derived vmin, vmax
    """
    if isinstance(arr, list):
        arr = np.array([x.values.ravel() for x in arr]).ravel()
    else:
        arr = arr.values.ravel()

    if type_ == "minmax":
        vmin, vmax = arr.nanmin(), arr.nanmax()
    elif type_ == "99p":
        vmin, vmax = np.nanquantile(arr, 0.01), np.nanquantile(arr, 0.99)
    else:
        raise ValueError(f"Invalid {type_ = }. Allowed are 'minmax', '99p'.")

    if vmin < 0 and vmax > 0:
        vmin = -max(abs(vmin), vmax)
        vmax = max(abs(vmin), vmax)
    return vmin, vmax


def standardize_data(
    xda_: xr.DataArray, ref_years: tuple[int] = (1961, 1990)
) -> xr.DataArray:
    """standardize the input data by subtracting the mean and dividing
    through the standard deviation of the reference time period.

    Args:
        xda_ (xr.DataArray): data that is standardized
        ref_years (tuple[int]): reference years used to subselect the data. Defaults to (1961, 1990).

    Returns:
        xr.DataArray: standardized output data
    """
    ref = xda_.sel(time=slice(f"{ref_years[0]}", f"{ref_years[1]}"))
    ref_mean = ref.mean(dim="time")
    ref_std = ref.std(dim="time")
    xda_norm = (xda_ - ref_mean) / ref_std
    return xda_norm


def plot_warming_stripes(
    grouped_xda: xr.DataArray,
    title_str: str,
    savename: str = "test.png",
    cmap: str = "RdBu_r",
) -> None:
    """plotting routines for warming stripes

    Args:
        grouped_xda (xr.DataArray): data to be plotted
        title_str (str): title string for plot
        savename (str, optional): path to save plot to. Defaults to "test.png".
    """
    labels = grouped_xda.group.values
    grouped_xda = grouped_xda.assign_coords(dict(group=range(len(labels))))
    ylen = len(labels)
    fig, ax = plt.subplots(figsize=(9, ylen / 2 - 1), layout="constrained")
    grouped_xda.plot(
        ax=ax,
        cmap=cmap,
        cbar_kwargs={
            "label": "Standardized anomaly",
            "aspect": 20 * (ylen / 13),
        },
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels=labels)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.title(f"{title_str} standardized anomalies (ref. 1961–1990)")
    plt.savefig(savename, bbox_inches="tight")
    plt.close()
    return None


if __name__ == "__main__":
    config = get_config(conf_file="config.toml")
    log = logger()
    log.info("Starting plotting")

    gen_anom_ts = config.PLOT.ANOMALY_TIMESERIES
    gen_clim_map = config.PLOT.CLIMATOLOGY_MAP
    gen_stampplots = config.PLOT.STAMP_PLOTS
    gen_stampplots_anomaly = config.PLOT.STAMP_PLOTS_ANOMALY
    gen_group_significance = config.PLOT.GROUP_SIGNIFICANCE
    gen_warming_stripes = config.PLOT.WARMING_STRIPES

    # set custom font
    configure_font()

    # get some variables from config file
    year_start = config.GENERAL.YEAR_START
    year_end = config.GENERAL.YEAR_END
    out_dir = Path(config.PATHS.OUT)
    plot_dir = str(out_dir).replace("indicators", "indicators_plots")

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
        if not any(
            [
                gen_anom_ts,
                gen_clim_map,
                gen_stampplots,
                gen_stampplots_anomaly,
            ]
        ):
            break
        log.info(f"Progress {idx + 1}/{len(clim_ind_inputs)} {index} {aggperiod}")

        outfile = Path(
            plot_dir,
            group,
            index,
            aggperiod,
            f"{index}_{aggperiod}_PLTTYPE.png",
        )
        outfile.parent.mkdir(exist_ok=True, parents=True)

        # anomaly plots, timeseries
        if gen_anom_ts:
            log.info("Plot anomaly timeseries")
            pngfile = str(outfile).replace("PLTTYPE", "tsanom")
            amean = xr.open_dataarray(
                get_eval_outfile_path_group(
                    str_="areamean",
                    out_dir=out_dir,
                    group=group,
                    index=index,
                    aggperiod=aggperiod,
                )
            )
            if group in ["precipitation", "snow", "runoff"]:
                color_kwargs = {"color_above": "C0", "color_below": "C3"}
            else:
                color_kwargs = {"color_above": "C3", "color_below": "C0"}

            if aggperiod in [
                "yea",
                "hydrological_year",
                "summer_halfyear",
                "winter_halfyear",
            ]:
                plot_anomaly_timeseries_year(
                    amean,
                    years_past,
                    years_now,
                    savefile=pngfile,
                    color_kwargs=color_kwargs,
                )
            elif aggperiod == "sea":
                plot_anomaly_timeseries_season(
                    amean,
                    years_past,
                    years_now,
                    savefile=pngfile,
                    color_kwargs=color_kwargs,
                )

        # plot spatial climatological maps
        if gen_clim_map:
            log.info("Plot climatological map")
            pngfile_past = str(outfile).replace("PLTTYPE", "clim_1961_1990")
            pngfile_now = str(outfile).replace("PLTTYPE", "clim_1991_2020")
            pngfile_anomaly = str(outfile).replace("PLTTYPE", "clim_anomaly")
            pngfile_comparison = str(outfile).replace("PLTTYPE", "clim_comparison")

            cmean_past = xr.open_dataset(
                get_eval_outfile_path_group(
                    str_=f"clim_{years_past[0]}_{years_past[1]}",
                    out_dir=out_dir,
                    group=group,
                    index=index,
                    aggperiod=aggperiod,
                )
            )[index]
            cmean_now = xr.open_dataset(
                get_eval_outfile_path_group(
                    str_=f"clim_{years_now[0]}_{years_now[1]}",
                    out_dir=out_dir,
                    group=group,
                    index=index,
                    aggperiod=aggperiod,
                )
            )[index]

            # get global vmin, vmax
            vmin, vmax = get_vmin_vmax(arr=[cmean_now, cmean_past], type_="99p")
            vmin_anom, vmax_anom = get_vmin_vmax(
                arr=cmean_now - cmean_past, type_="99p"
            )

            if group in ["precipitation", "snow", "runoff"]:
                cmap_div = "RdBu"
                cmap_div_clim = "PuOr"
            else:
                cmap_div = "RdBu_r"
                cmap_div_clim = "PuOr_r"

            if aggperiod in [
                "yea",
                "hydrological_year",
                "summer_halfyear",
                "winter_halfyear",
            ]:
                plot_spatial_map(
                    cmean_past,
                    titlestr=f"{index}: annual mean {years_past[0]}–{years_past[1]}",
                    savefile=pngfile_past,
                    vmin=vmin,
                    vmax=vmax,
                    cmap_div=cmap_div,
                )
                plot_spatial_map(
                    cmean_now,
                    titlestr=f"{index}: annual mean {years_now[0]}–{years_now[1]}",
                    savefile=pngfile_now,
                    vmin=vmin,
                    vmax=vmax,
                    cmap_div=cmap_div,
                )
                anom = cmean_now - cmean_past
                anom.attrs["units"] = unit
                plot_spatial_map(
                    anom,
                    titlestr=f"{index}: annual mean anomaly {years_now[0]}–{years_now[1]} - {years_past[0]}–{years_past[1]}",
                    savefile=pngfile_anomaly,
                    vmin=vmin_anom,
                    vmax=vmax_anom,
                    cmap_div=cmap_div,
                )
                plot_spatial_map_comparison(
                    cmean_past,
                    cmean_now,
                    anom,
                    titlestr=f"{index}: annual mean climatologies and difference",
                    savefile=pngfile_comparison,
                    vmin=vmin,
                    vmax=vmax,
                    cmap_div_anom=cmap_div,
                    cmap_div_clim=cmap_div_clim,
                )
            elif aggperiod == "sea":
                for xdaiter_past in cmean_past:
                    season_str = xdaiter_past.season.values
                    xdaiter_now = cmean_now.sel(season=season_str)
                    plot_spatial_map(
                        xdaiter_past,
                        titlestr=f"{index}: seasonal {season_str} mean {years_past[0]}–{years_past[1]}",
                        savefile=pngfile_past.replace(".png", f"_{season_str}.png"),
                        vmin=vmin,
                        vmax=vmax,
                        cmap_div=cmap_div,
                    )
                    plot_spatial_map(
                        xdaiter_now,
                        titlestr=f"{index}: seasonal {season_str} mean {years_now[0]}–{years_now[1]}",
                        savefile=pngfile_now.replace(".png", f"_{season_str}.png"),
                        vmin=vmin,
                        vmax=vmax,
                        cmap_div=cmap_div,
                    )
                    anom = xdaiter_now - xdaiter_past
                    anom.attrs["units"] = unit
                    vmin_anom, vmax_anom = get_vmin_vmax(arr=anom, type_="99p")
                    plot_spatial_map(
                        anom,
                        titlestr=f"{index}: seasonal {season_str} mean anomaly {years_now[0]}–{years_now[1]} - {years_past[0]}–{years_past[1]}",
                        savefile=pngfile_anomaly.replace(".png", f"_{season_str}.png"),
                        vmin=vmin_anom,
                        vmax=vmax_anom,
                        cmap_div=cmap_div,
                    )
                    plot_spatial_map_comparison(
                        xdaiter_past,
                        xdaiter_now,
                        anom,
                        titlestr=f"{index}: seasonal {season_str} mean climatologies and difference",
                        savefile=pngfile_comparison.replace(
                            ".png", f"_{season_str}.png"
                        ),
                        vmin=vmin,
                        vmax=vmax,
                        cmap_div_anom=cmap_div,
                        cmap_div_clim=cmap_div_clim,
                    )

        # spatial stampplots
        if gen_stampplots:
            log.info("Plot stampplots")
            pngfile = str(outfile).replace("PLTTYPE", "stampplot")

            outpath_ind = Path(out_dir, group, index, aggperiod)
            xda = xr.open_mfdataset(
                "/".join([str(outpath_ind), "*.nc"]),
                engine="h5netcdf",
                decode_timedelta=False,
            )[index]
            vmin, vmax = get_vmin_vmax(arr=xda, type_="99p")
            try:
                xda.attrs["units"]
            except KeyError:
                xda.attrs["units"] = unit

            if group in ["precipitation", "snow", "runoff"]:
                cmap_div = "RdBu"
            else:
                cmap_div = "RdBu_r"

            if aggperiod in [
                "yea",
                "hydrological_year",
                "summer_halfyear",
                "winter_halfyear",
            ]:
                stampplot_time(
                    xda=xda,
                    vmin=vmin,
                    vmax=vmax,
                    title_suffix=" annual",
                    savename=pngfile,
                    cmap_div=cmap_div,
                )
            elif aggperiod == "sea":
                for season_str, xdaiter in xda.groupby("time.season"):
                    stampplot_time(
                        xda=xdaiter,
                        vmin=vmin,
                        vmax=vmax,
                        title_suffix=f" {season_str}",
                        savename=pngfile.replace(".png", f"_{season_str}.png"),
                        cmap_div=cmap_div,
                    )

        # spatial stampplots of anomalies
        if gen_stampplots_anomaly:
            log.info("Plot stampplots anomalies")
            pngfile = str(outfile).replace("PLTTYPE", "stampplot_anomaly")

            outpath_ind = Path(out_dir, group, index, aggperiod)
            xda = xr.open_mfdataset(
                "/".join([str(outpath_ind), "*.nc"]),
                engine="h5netcdf",
                decode_timedelta=False,
            )[index]

            if group in ["precipitation", "snow", "runoff"]:
                cmap_div = "RdBu"
            else:
                cmap_div = "RdBu_r"

            if aggperiod in [
                "yea",
                "hydrological_year",
                "summer_halfyear",
                "winter_halfyear",
            ]:
                # calculate anomalies
                xda = xda - xda.sel(
                    time=slice(str(years_past[0]), str(years_past[1]))
                ).mean(dim="time")
                xda.attrs["units"] = unit
                vmin, vmax = get_vmin_vmax(arr=xda, type_="99p")
                stampplot_time(
                    xda=xda,
                    vmin=vmin,
                    vmax=vmax,
                    title_suffix=" annual\nanomalies 1961-1990",
                    savename=pngfile,
                    cmap_div=cmap_div,
                )
            elif aggperiod == "sea":
                for season_str, xdaiter in xda.groupby("time.season"):
                    # calculate anomalies
                    xdaiter = xdaiter - xdaiter.sel(
                        time=slice(str(years_past[0]), str(years_past[1]))
                    ).mean(dim="time")
                    xdaiter.attrs["units"] = unit
                    vmin, vmax = get_vmin_vmax(arr=xda, type_="99p")
                    stampplot_time(
                        xda=xdaiter,
                        vmin=vmin,
                        vmax=vmax,
                        title_suffix=f" {season_str}\nanomalies 1961-1990",
                        savename=pngfile.replace(".png", f"_{season_str}.png"),
                        cmap_div=cmap_div,
                    )

    if any([gen_group_significance, gen_warming_stripes]):
        ## create plots for groupings
        df_groups = pd.DataFrame(
            clim_ind_inputs,
            columns=[
                "index",
                "group",
                "parameter",
                "description",
                "aggperiod",
                "kwargs",
                "xclim_func",
                "units",
            ],
        )
        ## modify the groups according to the following
        # radiation are only 2 indicators, put them into temperature because
        #     they are highly correlated
        # runoff is added to snow, similarly because of how they are related
        # mixed indicators are moved into either temperature,
        #     precipitation, or humidity, depending on the indicator
        df_groups["group_plots"] = df_groups["group"]
        df_groups["aggperiod_plots"] = df_groups["aggperiod"]
        df_groups.loc[df_groups["group_plots"].isin(["radiation"]), "group_plots"] = (
            "temperature"
        )
        df_groups.loc[df_groups["group_plots"].isin(["runoff"]), "group_plots"] = "snow"
        df_groups.loc[
            df_groups["aggperiod_plots"].isin(["summer_halfyear"]), "aggperiod_plots"
        ] = "yea"
        df_groups.loc[
            df_groups["aggperiod_plots"].isin(["winter_halfyear"]), "aggperiod_plots"
        ] = "yea"
        df_groups.loc[
            df_groups["aggperiod_plots"].isin(["hydrological_year"]), "aggperiod_plots"
        ] = "yea"
        df_groups.loc[df_groups["index"].isin(["BIO08", "BIO09"]), "group_plots"] = (
            "temperature"
        )
        df_groups.loc[df_groups["index"].isin(["BIO18", "BIO19"]), "group_plots"] = (
            "precipitation"
        )
        df_groups.loc[df_groups["index"].isin(["SHMI"]), "group_plots"] = "temperature"
        df_groups.loc[
            df_groups["index"].isin(["ET0_Qcold", "ET0_Qdry", "ET0_Qwarm", "ET0_Qwet"]),
            "group_plots",
        ] = "humidity"

        for (grouping, aggp), df_iter in df_groups.groupby(
            ["group_plots", "aggperiod_plots"]
        ):
            log.info(
                f"Iterating with {grouping} n = {df_iter.shape[0]} with aggperiod {aggp}"
            )

            # significance plots
            if gen_group_significance:
                log.info("Plot grouped significance")
                outfile = Path(
                    plot_dir,
                    "grouped_plots",
                    "significance",
                    f"{grouping}_{aggp}_PLTTYPE.png",
                )
                outfile.parent.mkdir(exist_ok=True, parents=True)
                pngfile = str(outfile).replace("PLTTYPE", "significance")

                xda_group = xr.concat(
                    [
                        xr.open_dataset(
                            get_eval_outfile_path_group(
                                str_="significance",
                                out_dir=out_dir,
                                group=group_,
                                index=index,
                                aggperiod=aggperiod_,
                            )
                        )[index]
                        for index, group_, aggperiod_ in zip(
                            df_iter["index"], df_iter["group"], df_iter["aggperiod"]
                        )
                    ],
                    dim="group",
                    coords="minimal",
                    compat="override",
                )
                group_len = df_iter.shape[0]
                xda_group_sig = (
                    xr.where(xda_group < 0.05, 1, 0).sum(dim="group") / group_len * 100
                )
                xda_group_sig = xda_group_sig.where(
                    xda_group.isel(group=0).notnull(), np.nan
                )
                xda_group_sig.attrs["units"] = "%"
                xda_group_sig.name = "Portion of group"

                if aggp in [
                    "yea",
                    "hydrological_year",
                    "summer_halfyear",
                    "winter_halfyear",
                ]:
                    plot_spatial_map(
                        xda_group_sig,
                        titlestr=f"Portion of group (n={group_len}) with significant changes\nfrom 1961–1990 to 1991–2020 for {grouping} and {aggp}",
                        savefile=pngfile,
                        vmin=0,
                        vmax=100,
                    )
                elif aggp == "sea":
                    for xda_group_sig_iter in xda_group_sig:
                        xda_group_sig_iter.attrs["units"] = "%"
                        season_str = xda_group_sig_iter.season.values
                        plot_spatial_map(
                            xda_group_sig_iter,
                            titlestr=f"Portion of group (n={group_len}) with significant changes\nfrom 1961–1990 to 1991–2020 for {grouping} and {season_str}",
                            savefile=pngfile.replace(".png", f"_{season_str}.png"),
                            vmin=0,
                            vmax=100,
                        )

            # warming stripes
            if gen_warming_stripes:
                # grouped plots by indicator category
                log.info("Plot warming stripes")
                outfile = Path(
                    plot_dir,
                    "grouped_plots",
                    "warming_stripes",
                    f"{grouping}_{aggp}_PLTTYPE.png",
                )
                outfile.parent.mkdir(exist_ok=True, parents=True)
                pngfile = str(outfile).replace("PLTTYPE", "warming_stripes")

                xda_group = xr.concat(
                    [
                        xr.open_dataarray(
                            get_eval_outfile_path_group(
                                str_="areamean",
                                out_dir=out_dir,
                                group=group_,
                                index=index,
                                aggperiod=aggperiod_,
                            )
                        ).expand_dims({"group": [index]})
                        for index, group_, aggperiod_ in zip(
                            df_iter["index"], df_iter["group"], df_iter["aggperiod"]
                        )
                    ],
                    dim="group",
                    coords="minimal",
                    compat="override",
                    join="override",
                )
                group_len = df_iter.shape[0]
                if grouping in ["precipitation", "snow"]:
                    cmap = "RdBu"
                else:
                    cmap = "RdBu_r"
                xda_group.name = f"{grouping}_{aggp}"

                if aggp == "yea":
                    xda_iter = standardize_data(xda_group)
                    plot_warming_stripes(
                        xda_iter,
                        title_str=f"{grouping}: annual",
                        savename=pngfile,
                        cmap=cmap,
                    )
                elif aggp == "sea":
                    for season, xda_group_iter in xda_group.groupby("time.season"):
                        xda_group_iter_norm = standardize_data(xda_group_iter)
                        plot_warming_stripes(
                            xda_group_iter_norm,
                            title_str=f"{grouping}: seasonal {season}",
                            savename=pngfile.replace(".png", f"_{season}.png"),
                            cmap=cmap,
                        )
