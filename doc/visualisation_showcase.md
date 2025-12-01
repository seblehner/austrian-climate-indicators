# Visualisation showcase for climate indicators

This repository provides visualisation routines for climate indicators, which are aggregations of meteorological data over long timescales (climatological scale) to analyze climate change.
This is a brief showcase of the available visualisation types using the average air temperature (`TGmean`) as an example.

<!-- toc -->


## Time series of spatial averages

Visualisation of spatial averages across the SPARTACUS domain as annual or seasonal time series (with one subplot for each season).
These plots display spatially averaged anomalies relative to the climatological period 1961-1990.
Key elements of the plots are:

- Use of red/blue shading for values above/below the reference line.
- Horizontal reference lines for climatological averages (solid: 1961-1990, dotted: 1991-2020).
- The legend displays absolute values.

![](./example_plots/TGmean_yea_tsanom.png)

![](./example_plots/TGmean_sea_tsanom.png)


## Spatial plots for climatologies

Climatologies are visualised in two different ways:

1. Comparison plots: Displaying past (1961-1990), recent (1991-2020) climatologies, as well as their differences (recent - past).
   The plots for the seasonal comparison are generated for each season separately.
2. Standalone plots: Showing individual fields for specific seasons (c.f. third plot).

![](./example_plots/TGmean_yea_clim_comparison.png)

![](./example_plots/TGmean_sea_clim_comparison_DJF.png)

![](./example_plots/TGmean_sea_clim_1991_2020_JJA.png)


## Stampplots for yearly fields

Stamp plots visualise annual/seasonal fields for each year, arranged by decade. The rows start at the beginning of a decade.

Two versions are available:

1. the absolute values of a quantity (top plot) and 
2. anomalies relative to the past reference period (1961-1990; bottom plot).

![](./example_plots/TGmean_yea_stampplot.png)

![](./example_plots/TGmean_yea_stampplot_anomaly.png)


## Grouped significant changes

This visualisation highlights indicators with significant changes between the past (1961-1990) and recent (1991-2020) reference periods,
based on a two-tailed Mann-Whitney _U_ test (using a significance level of $\alpha =  0.05$).
Indicators are grouped into categories, with the result visualizing the portion of the group with significant changes (i.e., proportion of indices exhibiting $p < 0.05$).

Some indicators are merged into larger groups to have a more robust assessement. (1) `runoff` indicators are grouped with `snow`, (2) `radiation` with `temperature` and (3) `mixed` indicators into either `precipitation`, `temperature`, or `humidity` depending on the indicator.

![](./example_plots/temperature_yea_significance.png)


## Warming stripes

Warming stripes visualise area means of annual/seasonal anomalies relative to 1961-1990.
Indicators are grouped into categories for concise overviews, similar to the grouped significant changes plots.

(1) `runoff` indicators are grouped with `snow`, (2) `radiation` with `temperature` and (3) `mixed` indicators into either `precipitation`, `temperature`, or `humidity` depending on the indicator.

<details>
<summary>Humidity</summary>

![](./example_plots/humidity_yea_warming_stripes.png)

</details>

<details>
<summary>Snow</summary>

![](./example_plots/snow_yea_warming_stripes.png)

</details>

<details>
<summary>Precipitation</summary>

![](./example_plots/precipitation_yea_warming_stripes.png)

</details>

<details>
<summary>Temperature</summary>

![](./example_plots/temperature_yea_warming_stripes.png)

</details>
