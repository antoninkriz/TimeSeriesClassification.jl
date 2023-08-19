# Time Series Classification in Julia

Fast and optimized time series classification (TSC) toolkit for Julia.


## Models supported

- [MiniRocket](https://arxiv.org/abs/2012.08791)
- k-Nearest Neighbors (kNN) with Dynamic Time Warping and support Lower Bounding functions
    - Dynamic Time Warping
        - [DTW](https://link.springer.com/article/10.1007/BF01074755) (without any search space limitations)
        - DTW with [Sakoe-Chiba band](https://www.irit.fr/~Julien.Pinquier/Docs/TP_MABS/res/dtw-sakoe-chiba78.pdf)
        - DTW with [Itakura parallelogram](https://www.ee.columbia.edu/~dpwe/papers/Itak75-lpcasr.pdf)
        - Your own function
    - Lower Bounding
        - None (without any lower bounding)
        - [LB\_Keogh](https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm)
        - Your own function


## Examples

Examples created with [Pluto.jl](https://plutojl.org/) can be found in the `/examples` folder.

Execute the following command and open an example notebook.

```jl
import Pluto; Pluto.run()
```

All tests are located in the `/tests` folder.


## Performance

Benchmarks were done on 113 datasets from the [UCR Time Series Archive](https://www.timeseriesclassification.com/index.php).

When compared to [sktime](https://www.sktime.net/) (a Python package for machine learning with time series, optimized using [Numba](https://numba.pydata.org/) and based on [scikit-learn](https://scikit-learn.org/stable/index.html)) the MiniRocket implementation in Julia was 8.5 times faster and KNN was 17.8 faster.

Tests were done on Intel Core i7-11700 8c/16t @ 2.5 GHz with DDR4 RAM of 32 GB @ 3600 MHz, CL17.

### Benchmarks

All benchmarks, plot generating tools and other scripts are located in the `/benchmarks` folder.


## Tests

Open Julia REPL and execute following commands to run tests.

```jl
] activate .
] test
```


## About this package

This package is a project created as a part of my bachelors thesis [Time Series Classification in Julia](https://dspace.cvut.cz/handle/10467/109353) (Czech language only).
