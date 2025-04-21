# gc_detector

This is a Python-based globular cluster detection framework, which makes of several machine learning algorithms such as [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/), [convolutional neural networks(CNN)](https://www.tensorflow.org/tutorials/images/cnn) and [Gaussian mixture models (GMM)](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html). This repo is a supplement to the paper titled "Automated detection of globular clusters in _Gaia_ DR3: Proposal of 2 new moving group candidates" - [Baloyi et al. 2025](link/doi to be provided soon) . This work draws inspiration from the open cluster detection tool OCfinder by [Castro-Ginard et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...661A.118C/abstract) and related work by [Hunt and Reffert (2024)](https://ui.adsabs.harvard.edu/abs/2024A%26A...686A..42H/abstract).

From Gaia DR3's astrometric and photometric data, our  can:
- Extract spatial overdensities via HDBSCAN;
- Classify overdensities as GCs or noise from their corresponing CMDs (details in the paper), by means of a CNN;
- Collect/tabulate the members of the recovered known GCs, newly found GC candidates or other known objects via GMM
- Perform a cluster significance test to determine/quantify the astrometric density contrast between members and the background;
- Calculate distance, velocity dispersion (observed), and compute radial surface density profiles (to be added).   

![Flow_chart.svg](Schematic of the framework):

This work makes use of several packages including, but not limited to:
- `corner` 2.2.3
- `emcee` 3.1.4
- `hdbscan` 0.8.33
- `matplotlib` 3.7.2
- `numpy` 1.24.3
- `opencv-python` 4.9.0.80
- `pandas` 2.0.3
- `scikit-learn` 1.3.0
- `scipy` 1.11.1
- `tensorflow` 2.15.0
