"""Collection of functions handdling everything directly relation to climate indices computation."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import pandas as pd
import xarray as xr
import xclim as xc

# from climind_io import open_input_data
from utils import _get_logger, _sanity_check_varname

# if possible, xclim implementations are used to calculate climate indices
# indices for which no implementation exists are listed in the
# "_NO_XCIM_IMPLEMENTATION_" variable and have to implemented within the
# "set_impl()" function
_NO_XCLIM_IMPLEMENTATION_ = [
    "KYS",
    "spring_backlash",
    "SBI",
    "SAmean",
    "SRmean",
    "Rsum30min",
    "Rsum30max",
    "RR90pct",
    "RR95pct",
    "SPEI30mean",
    "SPEI90mean",
    "SPEI365mean",
    "ET0mean",
    "ET0_Qcold",
    "ET0_Qdry",
    "ET0_Qwarm",
    "ET0_Qwet",
    "ET0_seasonality",
    "SPEI90_th-2",
    "SPEI90_th-1",
    "SPEI90_th+1",
    "SPEI90_th+2",
    "SHMI",
    "HS_Q95",
    "HSfr_Q95",
    "SWE_Q95",
    "SWEfr_Q95",
    "NSD_72h",
    "runoff_Q95",
    "snowmelt_Q95",
    "continentality",
    "API07_Q95",
    "API14_Q95",
    "API28_Q95",
    "PCI_Q95",
]
_XCLIM_NUM_DAYS_TO_PERCENT = [
    "TN10p",
    "TX10p",
    "TN90p",
    "TX90p",
]
_XCLIM_THRESHOLD_SEASON_TENSOR = [
    "R75p",
    "R90p",
    "R95p",
    "R99p",
    "HSF",
    "CSF",
]
_XCLIM_FRACTION_TO_PERCENT = [
    "R90pTOT",
    "R95pTOT",
    "R99pTOT",
]


def _aggperiod_to_freq(aggperiod: str) -> str:
    """Extracts resampling requency for aggregated climate indices

    Args:
        aggperiod (str): aggperiod extracted from indices.csv

    Raises:
        ValueError: if an incompatible aggperiod is supplied

    Returns:
        str: resampling frequency compatible with xclim. See also:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """
    if aggperiod == "yea":
        freq = "YS"
    elif aggperiod == "sea":
        freq = "QE-FEB"
    elif aggperiod == "hydrological_year":
        freq = "YE-SEP"
    elif aggperiod == "summer_halfyear":
        freq = "YE-SEP"
    elif aggperiod == "winter_halfyear":
        freq = "YE-MAR"
    else:
        raise ValueError(
            f"Incompatible aggregation period: '{aggperiod}'. Allowed are 'year', "
            f"'sea', 'hydrological_year', 'summer_halfyear', 'winter_halfyear'"
        )
    return freq


def postprocess_climind(climind: xr.DataArray, index: str, freq: str) -> xr.DataArray:
    """Various postprocessing steps that some climate indicators need to match their
    definitions

    Args:
        climind (xr.DataArray): output climate indicator xr.DataArray
        index (str): name of index
        freq (str): resampling frequency

    Returns:
        xr.DataArray: postprocessed climate indicator
    """

    def select_same_seas(_xda: xr.DataArray) -> xr.DataArray:
        """helper function to select only overlapping time+season dims for xr.DataArray
        that has both a time and a season dim due to a tensor product.

        Args:
            _xda (xr.DataArray): input data with temporal time and seas dims

        Returns:
            xr.DataArray: output data with only time dim
        """
        return xr.concat(
            [
                _xda.where(_xda.time.dt.season.isin(seas), drop=True).sel(season=seas)
                for seas in _xda.season
            ],
            dim="time",
        )

    if index in _XCLIM_NUM_DAYS_TO_PERCENT:
        # output is days, but index should be percentage => needs normalisation and x100
        if freq == "YS":
            days_in_year = int(climind.time.dt.days_in_year)
            climind = climind.groupby("time.year").map(lambda x: x / days_in_year)
            climind = climind * 100
        elif freq == "QE-FEB":
            # when xclim is invoked with threshold data for multiple seasons
            # xclim will calculate every combination (like a tensor product)
            # but we are only in the combination of the same seasons, because
            # for example, the result for MAM with the threshold from DJF is not
            # needed. therefor this small functions selects only the relevant seasons
            climind = select_same_seas(_xda=climind)
            _year = int(climind.time.dt.year[0])
            _months = pd.date_range(f"{_year - 1}-12", f"{_year}-12", freq="1ME")
            _months_vals = _months.days_in_month.values
            _xdamon = xr.DataArray(data=_months_vals, coords={"time": _months})
            days_in_season = _xdamon.groupby("time.season").sum()
            climind = climind.groupby("time.season") / days_in_season
            climind = climind * 100
        climind.attrs["units"] = "%"
    if index in _XCLIM_THRESHOLD_SEASON_TENSOR:
        if freq == "QE-FEB":
            # when xclim is invoked with threshold data for multiple seasons
            # xclim will calculate every combination (like a tensor product)
            # but we are only in the combination of the same seasons, because
            # for example, the result for MAM with the threshold from DJF is not
            # needed. therefor this small functions selects only the relevant seasons
            climind = select_same_seas(_xda=climind)
    if index in _XCLIM_FRACTION_TO_PERCENT:
        # output is fraction, but index should be percentage => x100
        climind = climind * 100
        climind.attrs["units"] = "%"
    elif index == "EID":
        # EID is calculated using a differently named function which is why some
        # attributes need to be overwritten
        climind.attrs["standard_name"] = (
            "number_of_days_with_air_temperature_below_threshold"
        )
        climind.attrs["long_name"] = (
            "Number of days with daily minimum below -7 degc and daily maximum"
            " temperatures below 0 degc"
        )
        climind.attrs["history"] = (
            "xclim tx_tn_days_above has been used, but the input data was corrected"
            " to account for the methods requirement. {(tasmin*(-1))-14} & {tasmax*(-1)}"
        )
    elif index in ["RR_summerhalf", "RR_winterhalf"]:
        # due to the way xclim calculates and aggregates data and halfyear
        # using YE-month as frequency, xclim adds a second time step containing the
        # leftover months of the year. these need to be dropped
        climind = climind.isel(time=[0])
    return climind


def postprocess_units(climind: xr.DataArray) -> xr.DataArray:
    """Transforms units for temperature based climate indicators from K to degC.

    Args:
        climind (xr.DataArray): climate indicator xr.DataArray

    Returns:
        xr.DataArray: modified climate indicator xr.DataArray
    """
    log = _get_logger()
    if climind.attrs["units"] == "K":
        if climind.min().values >= 0 and climind.max().values >= 120:
            log.debug(f"Transform units from K to degC for {climind.name}")
            climind = xc.core.units.convert_units_to(climind, "degC")
        else:
            log.debug(
                f"No unit transformation needed for {climind.name}, because temperature is"
                " not an absolute value in Kelvin, but presumably already a difference and"
                " hence equal to degC."
            )
    return climind


def preprocess_climind(xda_dict: dict[xr.DataArray], index: str) -> dict[xr.DataArray]:
    """preprocess input data in specific ways for some climate indicators

    Args:
        xda_dict (dict[xr.DataArray]): input data wrapper in a dict
        index (str): name of climate indicator

    Returns:
        dict[xr.DataArray]: preprocessed input data wrapped in a dict
    """
    if index == "CE":
        assert len(xda_dict) == 1, f"only one input expected, but got {len(xda_dict)}"
        xda_dict_new = {}
        xda_dict_new["tas"] = xda_dict["tasmax"]
        return xda_dict_new
    elif index == "EID":
        # extreme ice days are calculated by using xclim's tx_tn_days_above
        # function. therefore the values have to multiplied by (-1)
        xda_dict_new = {}
        for xda_name, xda_iter in xda_dict.items():
            assert xda_iter.attrs["units"] == "degree_Celsius", (
                f"units need to be 'degree_Celsius', but are {xda_iter.attrs['units']}"
            )
            if xda_name == "tasmax":
                xda_dict_new[xda_name] = (xda_iter * (-1)).assign_attrs(xda_iter.attrs)
            elif xda_name == "tasmin":
                # tasmin needs a constant subtracted, because xclim has sanity checks
                # such that the threshold of tasmin must be lower than the threshold of
                # tasmax.
                xda_dict_new[xda_name] = ((xda_iter * (-1)) - 14).assign_attrs(
                    xda_iter.attrs
                )
        return xda_dict_new
    elif index in ["RR_summerhalf", "RR_winterhalf"]:
        # halfyear calculations are being done by providing an additional
        # month kwarg to the xclim function which is added here to the xda_dict
        if index == "RR_summerhalf":
            months = list(range(4, 10))
        elif index == "RR_winterhalf":
            months = list(range(10, 13)) + list(range(1, 4))
        xda_dict["month"] = months
        return xda_dict
    elif index in ["BIO13", "BIO14"]:
        # BIO indicators calculated via xclim's anuclim module are required to be
        # resampled to the intended period before. BIOCLIM indicators are based
        # on monthly periods.
        xda_iter = xda_dict["pr"]
        xda_iter = xda_iter.resample(time="1MS").sum().assign_attrs(xda_iter.attrs)
        xda_iter.attrs["units"] = "kg m-2 month-1"
        xda_dict["pr"] = xda_iter
        return xda_dict
    elif index == "SDII":
        # SDII requires specific units: "mm day-1"
        xda_iter = xda_dict["pr"]
        xda_iter.attrs["units"] = "mm day-1"
        xda_dict["pr"] = xda_iter
        return xda_dict
    else:
        return xda_dict


def _xclim_mapping(
    xda: list[xr.DataArray],
    index: str,
    extra_kwargs: dict,
    xclim_func: str,
    freq: str = "YS",
) -> xr.DataArray:
    """Call xclim to calculate climate indicator, while also doing some sanity checks.

    Args:
        xda (list[xr.DataArray]): input data wrapped in a list
        index (str): name of climate indicator
        extra_kwargs (dict): any extra kwargs for xclim that are just passed through to the
            xclim call
        xclim_func (str): name of xclim method in the form of "domain.indicatorname"
        freq (str, optional): output resampling frequency. Defaults to "YS".

    Returns:
        xr.DataArray: calculated climate indicator
    """
    # check correct varnames for xclim
    xda = [_sanity_check_varname(xda_iter) for xda_iter in xda]
    # wrap input data in a dictionary to use xclims default
    # way of providing input dataarrays
    xda = {xda_iter.name: xda_iter for xda_iter in xda}
    # sanity control for correct xclim module domain
    xclim_module_levels = xclim_func.split(".")
    assert xclim_module_levels[0] in ["indices", "indicators"], (
        f"invalid xclim module realm: {xclim_module_levels[0]}"
    )
    # call xclim with specified method
    if len(xclim_module_levels) == 2:
        xclim_func_obj = getattr(
            getattr(xc, xclim_module_levels[0]), xclim_module_levels[1]
        )
    elif len(xclim_module_levels) == 3:
        xclim_func_obj = getattr(
            getattr(getattr(xc, xclim_module_levels[0]), xclim_module_levels[1]),
            xclim_module_levels[2],
        )
    xda = preprocess_climind(xda_dict=xda, index=index)
    climind = xclim_func_obj(**xda, freq=freq, **extra_kwargs)
    climind = postprocess_climind(climind=climind, index=index, freq=freq)
    return climind


def self_impl(
    xda: xr.DataArray, index: str, freq: str, extra_kwargs: dict = None
) -> xr.DataArray:
    """Function to handle self implemented climate indices.

    Args:
        xda (xr.DataArray): input variables for climate indices calculates
        index (str): name of climate index
        freq (str): resampling frequency
        extra_kwargs (str): additional kwargs for climate indicator calculation. Defaults to None

    Returns:
        xr.DataArray: calculated climate indices
    """

    def _get_xda_by_name_from_list(xda_: xr.DataArray, name_: str):
        return [xda_it for xda_it in xda_ if xda_it.name == name_][0]

    def _calc_quarterly_et0(et0_, time_):
        et0__mean = et0_.resample(time="QE-FEB").mean().compute()
        et0__mean = et0__mean.where(et0__mean.time == time_, drop=True)
        et0__mean["time"] = pd.date_range(
            f"{int(et0__mean.time.dt.year)}", freq="YS", periods=1
        )
        return et0__mean

    if index in [
        "SAmean",
        "SRmean",
        "SPEI30mean",
        "SPEI90mean",
        "SPEI365mean",
        "ET0mean",
    ]:
        climind = xda[0].resample(time=freq).mean()
    elif index in ["Rsum30min"]:
        climind = xda[0].rolling(time=30).sum().resample(time=freq).min()
    elif index in ["Rsum30max"]:
        climind = xda[0].rolling(time=30).sum().resample(time=freq).max()
    elif index in ["RR90pct", "RR95pct"]:
        tmp = xda[0].where(xda[0] >= 1, other=np.nan)  # only wet days
        climind = tmp.resample(time=freq).quantile(**extra_kwargs)
        climind.attrs["units"] = xda[0].attrs["units"]
    elif index == "ET0_Qcold":
        ET0 = _get_xda_by_name_from_list(xda, "ET0")
        TG = _get_xda_by_name_from_list(xda, "TG")
        coldest_seas = TG.sum(dim=["x", "y"]).resample(time="QE-FEB").mean().compute()
        cold_time = coldest_seas.where(
            coldest_seas == coldest_seas.min(dim="time"), drop=True
        ).time
        climind = _calc_quarterly_et0(ET0, cold_time)
    elif index == "ET0_Qdry":
        ET0 = _get_xda_by_name_from_list(xda, "ET0")
        RR = _get_xda_by_name_from_list(xda, "RR")
        driest_seas = RR.sum(dim=["x", "y"]).resample(time="QE-FEB").mean().compute()
        dry_time = driest_seas.where(
            driest_seas == driest_seas.min(dim="time"), drop=True
        ).time
        climind = _calc_quarterly_et0(ET0, dry_time)
    elif index == "ET0_Qwarm":
        ET0 = _get_xda_by_name_from_list(xda, "ET0")
        TG = _get_xda_by_name_from_list(xda, "TG")
        warmest_seas = TG.sum(dim=["x", "y"]).resample(time="QE-FEB").mean().compute()
        warm_time = warmest_seas.where(
            warmest_seas == warmest_seas.max(dim="time"), drop=True
        ).time
        climind = _calc_quarterly_et0(ET0, warm_time)
    elif index == "ET0_Qwet":
        ET0 = _get_xda_by_name_from_list(xda, "ET0")
        RR = _get_xda_by_name_from_list(xda, "RR")
        wet_seas = RR.sum(dim=["x", "y"]).resample(time="QE-FEB").mean().compute()
        wet_time = wet_seas.where(wet_seas == wet_seas.max(dim="time"), drop=True).time
        climind = _calc_quarterly_et0(ET0, wet_time)
    elif index == "ET0_seasonality":
        climind = xda[0].resample(time="1ME").std().resample(time=freq).mean()
    elif index == "SPEI90_th-2":
        climind = xda[0].where(xda[0] < -2, np.nan).resample(time=freq).count()
        climind = climind.where(_get_xy_nanmask_(xda[0]), np.nan)
        climind.attrs["units"] = "days"
    elif index == "SPEI90_th-1":
        climind = xda[0].where(xda[0] < -1, np.nan).resample(time=freq).count()
        climind = climind.where(_get_xy_nanmask_(xda[0]), np.nan)
        climind.attrs["units"] = "days"
    elif index == "SPEI90_th+1":
        climind = xda[0].where(xda[0] > +1, np.nan).resample(time=freq).count()
        climind = climind.where(_get_xy_nanmask_(xda[0]), np.nan)
        climind.attrs["units"] = "days"
    elif index == "SPEI90_th+2":
        climind = xda[0].where(xda[0] > +2, np.nan).resample(time=freq).count()
        climind = climind.where(_get_xy_nanmask_(xda[0]), np.nan)
        climind.attrs["units"] = "days"
    elif index == "SHMI":
        TG = _get_xda_by_name_from_list(xda, "TG")
        RR = _get_xda_by_name_from_list(xda, "RR")
        warmest_month = TG.resample(time="ME").mean().compute().max(dim="time")
        summer_rr = RR.where(RR.time.dt.month.isin([6, 7, 8]), np.nan).mean(dim="time")
        climind = warmest_month / summer_rr
        climind = climind.expand_dims(
            dim={
                "time": pd.date_range(
                    f"{int(RR.time.dt.year[-1])}", freq="YS", periods=1
                )
            },
            axis=0,
        )
        climind.attrs["units"] = "degC mm-1"
    elif index in [
        "HS_Q95",
        "HSfr_Q95",
        "SWE_Q95",
        "SWEfr_Q95",
        "runoff_Q95",
        "snowmelt_Q95",
    ]:
        climind = xda[0].resample(time=freq).quantile(0.95)
        climind.attrs["units"] = xda[0].attrs["units"]
    elif index == "NSD_72h":
        climind = xda[0].rolling(time=3).sum().resample(time=freq).max()
    elif index == "continentality":
        TG_mean = xda[0].resample(time="ME").mean().compute()
        warmest_month = TG_mean.resample(time="YS").max(dim="time")
        coldest_month = TG_mean.resample(time="YS").min(dim="time")
        climind = warmest_month - coldest_month
        climind.attrs["units"] = "degC"
    elif index == "KYS":
        climind = _calc_kysely_(xrda=xda[0], freq=freq)
    elif index == "spring_backlash":
        TG = _get_xda_by_name_from_list(xda, "TG")
        TN = _get_xda_by_name_from_list(xda, "TN")
        gss = xc.indicators.atmos.growing_season_start(tas=TG).compute().squeeze()
        TN_past_gss = TN.where(TN.time.dt.dayofyear > gss, np.nan)
        TN_inside = TN_past_gss.where(TN_past_gss.time.dt.dayofyear < 181, np.nan)
        backlash_events = TN_inside.where(TN_inside < -2, np.nan)
        backlash_events = backlash_events.where(backlash_events.isnull(), 1)
        climind = backlash_events.resample(time="YS").sum()
        climind = climind.where(_get_xy_nanmask_(xrda=TG), np.nan)
    elif index == "SBI":
        TG = _get_xda_by_name_from_list(xda, "TG")
        TN = _get_xda_by_name_from_list(xda, "TN")
        gss = xc.indicators.atmos.growing_season_start(tas=TG).compute().squeeze()
        TN_past_gss = TN.where(TN.time.dt.dayofyear > gss, np.nan)
        TN_inside = TN_past_gss.where(TN_past_gss.time.dt.dayofyear < 181, np.nan)
        # we want to find the last backlash event. for this we set values above -2 to 0
        # then multiply with -1 and use cumsum => the argmax output will return the first
        # occurrence of the max value, which will then be the last backlash event
        backlash_events = (TN_inside.where(TN_inside < -2, 0) * (-1)).cumsum(dim="time")
        last_backlash_event = backlash_events.argmax(dim="time")
        # set TG value to NaN outside range of Gss to last backlash event
        TGdoy = TG.time.dt.dayofyear
        TG_inside = TG.where((last_backlash_event > TGdoy) & (TGdoy > gss), 0)
        climind = xc.indicators.atmos.growing_degree_days(
            tas=TG_inside, thresh="5 degC"
        )
    elif index in ["API07_Q95", "API14_Q95", "API28_Q95"]:
        climind = _calc_api_(xda[0], freq=freq, **extra_kwargs)
    elif index == "PCI_Q95":
        climind = _calc_pci_(xda[0], freq=freq)
    return climind


def calc_climate_indices(
    xda: list[xr.DataArray],
    index: str,
    aggperiod: str,
    extra_kwargs: dict,
    xclim_func: str,
) -> xr.DataArray:
    """Main function to handle calculation of climate indices. Calls either xclim to calculate
    climate indices, or if no xclim implementation exists, the self implementated version.

    Args:
        xda (list[xr.DataArray]): input data to calculate climate indices from wrapped in a list
        index (str): name of climate index
        aggperiod (str): resampling frequency for climate indicator (annual/seasonal <> 'yea'/'sea')
        extra_kwargs (dict): additional kwargs for climate indicator calculation
        xclim_func (str): name of method within xclim.indicator module for climate indicator calculation

    Returns:
        xr.DataArray: calculated climate index
    """
    log = _get_logger()
    freq = _aggperiod_to_freq(aggperiod=aggperiod)
    # make sure data is chunked correctly in the time dimension, which is the only dim where
    # any operation takes place
    xda = _ensure_temporal_chunking(xda=xda)

    if index in _NO_XCLIM_IMPLEMENTATION_:
        log.debug(
            f"no implementation in xclim for {index = }, calculate via self implementation"
        )
        climind = self_impl(xda=xda, index=index, freq=freq, extra_kwargs=extra_kwargs)
    else:
        # call xclim for calculation
        log.debug(f"calculate via xclim: {index}")
        climind = _xclim_mapping(
            xda=xda,
            index=index,
            freq=freq,
            extra_kwargs=extra_kwargs,
            xclim_func=xclim_func,
        )
    climind = postprocess_units(climind=climind)
    return climind


def _ensure_temporal_chunking(xda: list[xr.DataArray]) -> list[xr.DataArray]:
    """ensure that xarray data is chunked properly in the temporal dim.
    temporal dim name  should be either of time, dayofyear, or season

    Args:
        xda (list[xr.DataArray]): input xr.DataArray wrapped in a list

    Returns:
        list[xr.DataArray]: rechunked xr.DataArray wrapped in a list
    """
    xda_new = []
    for xda_ in xda:
        if any(tdim in xda_.dims for tdim in ["time", "dayofyear", "season"]):
            for tdim in ["time", "dayofyear", "season"]:
                if tdim in xda_.dims:
                    dim = tdim
            xda_tmp = xda_.chunk({dim: -1})
        else:
            xda_tmp = xda_
        xda_new.append(xda_tmp)
    return xda_new


def calculate_aggperiod_percentile(
    thres_input: xr.DataArray, aggperiod: str, pctl: int
) -> xr.DataArray:
    """calculate percentile threshold from xr.DataArray needed for certain xclim indicators

    Args:
        thres_inputs (xr.DataArray): input data for which a percentile is calculated
        aggperiod (str): resampling frequency
        pctl (int): percentile [0-100]

    Returns:
        xr.DataArray: calculated percentile threshold
    """
    thres_input = thres_input.chunk(dict(time=-1))
    if aggperiod == "yea":
        thresholds = thres_input.quantile(pctl / 100, dim="time")
    elif aggperiod == "sea":
        thresholds = thres_input.groupby("time.season").quantile(pctl / 100)
    # Note: The name of the dataarray is set to "thresh", because it is intended to
    # be used for xclim functions that take a "thresh" argument, where instead of a
    # static threshold, a gridpoint/temporally based threshold is provided with this data
    thresholds.name = "thresh"
    thresholds.attrs["units"] = thres_input.attrs["units"]
    thresholds.attrs["aggperiod"] = aggperiod
    thresholds.attrs["percentile"] = pctl
    thresholds.attrs["reference_period"] = (
        f"{int(thres_input.time.dt.year.min())}-{int(thres_input.time.dt.year.max())}"
    )
    return thresholds


def calculate_doy_percentile(
    xda_daily: xr.DataArray, kwarg_key: str, pctl: int, window: int = 5
) -> xr.DataArray:
    """calculate doy+window based percentile using xclim core funcs

    Args:
        xda_daily (xr.DataArray): daily input data
        kwarg_key (str): name of input kwarg for xclim func of the form "<param>_per"
        pctl (int): percentile [0-100]
        window (int, optional): window for doy percentile calculation. Defaults to 5.

    Returns:
        xr.DataArray: doy based percentiles
    """
    # Note: the percentile for this method needs to be between [0-100]
    # and not [0-1] like for the other methods => no division by 100
    assert pctl >= 1, f"supplied percentile needs to be between [0, 100], but is {pctl}"
    xda_per = xc.core.calendar.percentile_doy(
        xda_daily.chunk(dict(time=-1)), window=window, per=pctl
    )
    xda_per.name = kwarg_key
    xda_per.attrs["units"] = xda_daily.attrs["units"]
    xda_per.attrs["doy_window"] = window
    xda_per.attrs["percentile"] = pctl
    xda_per.attrs["reference_period"] = (
        f"{int(xda_daily.time.dt.year.min())}-{int(xda_daily.time.dt.year.max())}"
    )
    return xda_per.squeeze()


def _calc_kysely_(xrda: xr.DataArray, freq: str) -> xr.DataArray:
    """Kysely days, kysely
    Annual number of days inside a heatwave. The start of the heatwave is defined
    as 3 consecutive days with TX > 30°C and lasts as long as TX does not go below 25°C
    and as long as the TX average the period stays above 30°C.
    """
    xrda, nanmask = _check_input_and_get_nanmask_(xrda=xrda, input_data_cat="T")

    # calculate index
    tmp = xrda.where(xrda > 30, other=np.nan)
    tmp.load()
    xrda.load()
    # first guess of kysely episodes: at least 3 consecutive days above 30 degrees
    kys_days = ~tmp.isnull().values
    # define integer indices
    tx_s = tmp.assign_coords(
        ix=("x", range(len(tmp.x))),
        iy=("y", range(len(tmp.y))),
        it=("time", range(len(tmp.time))),
    )
    # stack arrays and drop all irrelevant gridpoints
    tx_s = tx_s.stack(xy=("x", "y")).T
    tmp2 = xrda.stack(xy=("x", "y")).T
    tx_s = tx_s.dropna("xy", how="all")
    tmp2 = tmp2.sel(xy=tx_s.xy)
    # drop days that are never part of first guess
    tx_s = tx_s.dropna("time", how="all")
    # get updated index arrays
    ixs, iys, its = tx_s.ix.values, tx_s.iy.values, tx_s.it.values
    # get reduced first guess
    kys_days_s = ~tx_s.isnull().values
    # convert to numpy arrays for better performance
    tmp3 = tmp2.values
    tx_s = tx_s.values
    # loop over gridpoints
    for ix, iy, datxy_org, kys_days_xy in zip(ixs, iys, tmp3, kys_days_s):
        # all time indices that are part of first guess
        # -> can be start of Kysely episode
        t30 = its[kys_days_xy]
        end = -1
        for t in t30:
            # day already part of previous episode?
            if t <= end:
                continue
            tm = 0
            for k, d in enumerate(datxy_org[t:]):
                # update episode mean
                tm = (k * tm + d) / (k + 1)
                if d > 30:
                    # already part of first guess
                    continue
                elif (tm > 30) and (d > 25):
                    kys_days[t + k, iy, ix] = True
                else:
                    # end of episode
                    end = t + k
                    break
    # numpy array into xr.DataArray
    da_out = xr.DataArray(
        data=kys_days,
        dims=["time", "y", "x"],
        coords={
            "time": xrda.time.values,
            "y": xrda.y.values,
            "x": xrda.x.values,
        },
    )
    # aggregate to yearly counts
    da_out = da_out.resample(time=freq).sum()

    # set original nan gridcells back to nan
    fin = da_out.where(nanmask, np.nan)
    fin.attrs["units"] = "days"
    return fin


def _check_input_and_get_nanmask_(xrda: xr.DataArray, input_data_cat: str = None):
    """wrapper function to sanity check input data and get xy nanmask"""
    # sanity check that input is of correct type
    xrda = _sanity_check_one_dataarray_(xrda=xrda)
    if input_data_cat in ["temperature", "t", "T", "temp", "Temp"]:
        # check that temperatures are in deg C and transform if not
        xrda = _check_transform_temperature_degc_(xrda=xrda)
    # get mask for nan values in space (out of bounds gridcells)
    nanmask = _get_xy_nanmask_(xrda=xrda)
    return xrda, nanmask


def _check_transform_temperature_degc_(xrda: xr.DataArray) -> xr.DataArray:
    """check if input temperature field are in degree Celsius
    and if not, transform them (assuming the original unit is Kelvin)
    """
    if xrda.isel(time=0).mean() > 150:
        return xrda - 273.15
    else:
        return xrda


def _get_xy_nanmask_(xrda: xr.DataArray) -> xr.DataArray:
    """Get (x,y) nanmask. Valid values = True, NaN values = False.
    Assumes the dimensions "x" and "y" are the spatial dimensions
    from which the nanmask is extracted. All other dimensions are
    reduced to the first element.
    """
    dim_slicer = {dim: 0 for dim in xrda.dims if dim not in ["x", "y"]}
    return xrda.isel(dim_slicer).notnull()


def _sanity_check_one_dataarray_(xrda: xr.DataArray) -> xr.DataArray:
    """convenience function to check if input is a single xr.DataArray, or
    a single xr.DataArray wrapped inside a list.
    """
    if isinstance(xrda, list):
        assert len(xrda) == 1, f"Expected len=1, got {len(xrda)}"
        return xrda[0]
    elif isinstance(xrda, xr.DataArray):
        return xrda
    else:
        raise TypeError(f"Expected a xr.DataArray, got {type(xrda)}")


def _calc_api_(
    xrda: xr.DataArray,
    k_timeperiod: int = 7,
    p_decay: float = 0.935,
    quantile: str = 0.95,
    freq: str = "YS",
) -> xr.DataArray:
    """API - Antecedent Precipitation Index
    weighted summation of daily precipitation amounts:
    using k_timeperiod = 7 days and a decay constant of = 0.935

    k_timeperiod specifies the time window which is considered for the calculation
    p_decay is a decay parameter usually between 0.85 and 0.98

    References:
    (i) https://doi.org/10.1016/j.crm.2021.100294
    (ii) https://doi.org/10.1016/j.jhydrol.2021.126027
    """
    # sanity check that input is of correct type
    xrda = _sanity_check_one_dataarray_(xrda=xrda)

    # get mask for nan values in space (out of bounds gridcells)
    nanmask = _get_xy_nanmask_(xrda=xrda)

    # construct a xr.DataArray containing the weights based on input params
    weights = xr.DataArray(
        list(reversed([p_decay ** (idx - 1) for idx in range(1, k_timeperiod + 1)])),
        dims="window_dim",
    )
    # apply the weights to input data and the given time window
    api = xrda.rolling(time=k_timeperiod).construct("window_dim").dot(weights)
    api.attrs["k"] = k_timeperiod
    api.attrs["p"] = p_decay

    # some data aggregation might be sought after, cause the output of API is daily
    # API resampling is given by aggregation method
    fin = api.resample(time=freq).quantile(quantile)
    fin.attrs["quantile"] = quantile

    # set original nan gridcells back to nan
    fin = fin.where(nanmask, np.nan)
    fin.attrs["units"] = "kg m-2 day-1"
    return fin


def _calc_pci_(rr_da: xr.DataArray, freq: str) -> xr.DataArray:
    """Calculate the prectipitation concentration index (PCI)"""
    total_sq = rr_da.resample(time=freq).sum() ** 2
    monthlies_sq_sum = (rr_da.resample(time="1ME").sum() ** 2).resample(time=freq).sum()
    pci = monthlies_sq_sum / total_sq * 100
    pci.name = "pci"
    pci.attrs["units"] = "1"
    return pci
