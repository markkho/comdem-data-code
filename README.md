# Communicative Demonstrations Code

## Data
Trial-by-trial data for Experiments 1a, 1b, 2a, and 2b are included in `./experiments/data`.

## Python Code
The model implementations in `./experiments` are in python and use two packages that are included in this repo (pyrlap and demoteaching). To set up the python environment to use those packages, follow these steps:

1. Set up an environment using, e.g., anaconda:
```
$ conda create -y -n comdem python=3.6 numpy scipy matplotlib seaborn pandas jupyter pytorch scikit-learn
```

2. Start environment
```
$ conda activate comdem
```

3. Set up libraries in environment
```
$ pip install -e ./lib/pyrlap
$ pip install -e ./lib/demoteaching
```

## WebPPL code
The code in `./dev-models` is implemented in WebPPL. For details on installing and using WebPPL see their documentation [here](https://webppl.readthedocs.io/en/master/).
