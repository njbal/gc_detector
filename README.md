# gc_detector

This is a Python-based globular cluster detection framework, which makes of several machine learning algorithms such as [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/), [convolutional neural networks (CNN)](https://www.tensorflow.org/tutorials/images/cnn) and [Gaussian mixture models (GMM)](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html). This repo is a supplement to the paper titled "Automated detection of globular clusters in _Gaia_ DR3: Proposal of 2 new moving group candidates" - [Baloyi et al. 2025](link/doi to be provided soon) . This work draws inspiration from the open cluster detection tool OCfinder by [Castro-Ginard et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...661A.118C/abstract) and related work by [Hunt and Reffert (2024)](https://ui.adsabs.harvard.edu/abs/2024A%26A...686A..42H/abstract).

From Gaia DR3's astrometric and photometric data, our framework can:
- Extract spatial overdensities via HDBSCAN;
- Classify overdensities as GCs or noise from their corresponing CMDs (details in the paper), by means of a CNN;
- Collect/tabulate the members of the recovered known GCs, newly found GC candidates or other known objects via GMM
- Perform a cluster significance test to determine/quantify the astrometric density contrast between members and the background;
- Calculate parallax-based distances, proper motion dispersions, and compute radial surface density profiles (to be added).   

![Framework depiction](Flow_chart.svg)
**Basic schematic of the framework**

This work makes use of several packages including, but not limited to:
- [`astropy`](https://docs.astropy.org/en/stable/index.html) 5.3.4
- [`corner`](https://corner.readthedocs.io/) 2.2.3
- [`emcee`](https://emcee.readthedocs.io/) 3.1.4
- [`hdbscan`](https://hdbscan.readthedocs.io/) 0.8.33
- [`matplotlib`](https://matplotlib.org/stable/index.html) 3.7.2
- [`numpy`](https://numpy.org/doc/stable/) 1.24.3
- [`opencv-python`](https://docs.opencv.org/4.x/index.html) 4.9.0.80
- [`pandas`](https://pandas.pydata.org/docs/) 2.0.3
- [`scikit-learn`](https://scikit-learn.org/stable/) 1.3.0
- [`scipy`](https://docs.scipy.org/doc/scipy/) 1.11.1
- [`tensorflow`](https://www.tensorflow.org/api_docs) 2.15.0

We also utilised the [`gaia_zeropoint`](https://gitlab.com/icc-ub/public/gaiadr3_zeropoint) parallax correction function by [Lindegren et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...649A...4L/abstract). We discuss in our paper how we dealt with this functions tendency for overcorrections.