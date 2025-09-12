# Contributing

✨ Contributions are welcome, and they are greatly appreciated! 🙏


## Suggestions and discussion about indicators

Feel free to open an issue to discuss any of the existing implementations, or to suggest the addition of a new indicator.

Merge requests with changes, bug fixes or additions of new indicators are encouraged.


## Implementation of new indicators

New indicators require additions in a few different places depending on if there is an existing `xclim` implementation available or not.

First, new indicators need to be added into the `doc/indices.csv` table.
This is the main file based on which information is programmatically used to load input data, handle different resampling frequencies, link to existing `xclim` routines and more.

Second, if no corresponding `xclim` implementation is provided, the indicator needs to be implemented by hand.
This happens in the `src/climind_calc.py` file in mainly two places:
1. The variable `_NO_XCLIM_IMPLEMENTATION_` needs to extended by the new indicator by using the abbreviated indicator name (e.g. `SU` for Summer Days).
2. Inside the function `self_impl()` a new `elif` block needs to be added that contains the calculation for the new indicator.
If the calculations are long, it's best to write it inside an individual function and just call the function in `elif` block, for brevity.
