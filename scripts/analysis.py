import sys
sys.dont_write_bytecode = True

from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})

import emcee
import corner
from matplotlib.backends.backend_pdf import PdfPages
from tables_and_vis import ensure_dir
import pandas as pd

# Define King profile:
def King(r, k, R_c, R_t, const):
    n = np.zeros_like(r)
    term1 = (1+(r/R_c)**2)**(-1/2)
        
    term2 = (1+(R_t/R_c)**2)**(-1/2)
    n = k*(term1 - term2)**2 + const
    return n

# Maximum likelihood estimation for radial profile:
def radial_mle(file_path, r_lower, r_upper, priors, p0):
    """Returns radial surface density profile and corresponding MCMC profiles.
    Parameters:
    
    file_path: path to the cluster membership list file
    r_lower: minimum radius, from cluster centre, at which the radial profile must begin.
    r_upper: maximum radius, from cluster centre, at which the radial profile ends
    priors: prior distribution (uniform), with specified bounds for each fit parameter 
    p0: initial guess"""
    
    T = ascii.read(file_path)
    prob_ind = T['gmm_mem_prob']>=0.9
    T = T[prob_ind]
    ra, dec = np.radians(T['ra']), np.radians(T['dec'])
    
    x = np.sin(ra-ra.mean())*np.cos(dec)
    y = np.cos(dec.mean())*np.sin(dec) - np.sin(dec.mean())*np.cos(dec)*np.cos(ra-ra.mean())
    R = (x**2 + y**2)**0.5*(180/np.pi*60) # R in arcmin!

    bws = [0.4e-2, 0.5e-2, 1e-1]
    N_array = np.array([])
    r_array = np.array([])

    r_lo, r_hi = 0.001, 20

    for bw0 in bws:
    #     bw0 = 1e-1 #  bin width
        r_bins = np.arange(r_lo, r_hi + bw0, bw0)
        for r in r_bins[:-1]:
            N = len(R[(R<=r+bw0)&(R>r)])/(np.pi*((r+bw0)**2 - r**2))
            N_array = np.append(N_array, N)
            r_array = np.append(r_array, r)


    N_array_nz = N_array[(N_array>0)] # Non-zero N(r) values
    r_array_nz = r_array[(N_array>0)] # corresponding r values

    bw1 = 5e-1

    r_bins1 = np.arange(r_lo, r_hi + bw1, bw1)

    N_new = np.array([])
    eN_new = np.array([])
    r_new = np.array([])
    for r_ in r_bins1[:-1]:
        final_ind = (r_array_nz<=r_+bw1)&(r_array_nz>r_)
        N_vals = N_array_nz[final_ind]

        if N_vals.size<=1:
            continue
        else: 
            N_mean = N_vals.mean()
            eN = np.sqrt(N_mean/N_vals.size)
            N_new = np.append(N_new, N_mean)
            eN_new = np.append(eN_new, eN)
            r_new = np.append(r_new, r_+0.5*bw1)

    # Log-Likelihood Function for MCMC
    def log_likelihood(theta, r, N_obs, eN_obs):
        k, r_c, r_t, eta = theta
        model = King(r, k, r_c, r_t, eta)
        return -0.5 * np.sum(((N_obs - model) / eN_obs) ** 2)

    # Define Priors (Uniform Prior)
    def log_prior(theta):
        k, r_c, r_t, eta = theta
        if priors[0,0] < k < priors[0,1] and priors[1,0] < r_c < priors[1,1] and priors[2,0] < r_t < priors[2,1] and priors[3,0] < eta < priors[3,1]:
            return 0.0  # Flat prior
        return -np.inf  # Log probability of -inf means rejected sample

    # Define Full Probability Function
    def log_probability(theta, r, N_obs, eN_obs):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, r, N_obs, eN_obs)

    # Initialize MCMC Sampler
    ndim, nwalkers = 4, 16  

    pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)  # Slightly perturb initial guess

    r_ind = (r_new>r_lower)&(r_new<r_upper)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(r_new[r_ind], N_new[r_ind], eN_new[r_ind]))


    # Run MCMC
    nsteps = 10000
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Extract Results
    samples = sampler.get_chain(discard=2000, thin=10, flat=True)
    
    return samples, r_new[r_ind], N_new[r_ind], eN_new[r_ind]


# corner plot function:
def corner_plot(file_path, samples, r_lower, r_upper):
    clus_name = file_path.split('/')[-1][:-5]
    
    best_fit = np.median(samples[0], axis=0)
    error_hi = np.percentile(samples[0], 84, axis=0) - best_fit # upper error
    error_lo = best_fit - np.percentile(samples[0], 16, axis=0) # lower error
    error_ab = (error_hi**2 + error_lo**2)**0.5 # total error

    # Plot Results
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.errorbar(samples[1], samples[2], yerr=samples[3], 
                 capsize=2, lw=0, elinewidth=1, 
                 ecolor='red', marker='s', 
                 color='black', label="Data", ms=2)
    rfit = np.linspace(r_lower, r_upper, 1000)
    ax1.plot(rfit, King(rfit, *best_fit), label="MCMC Fit", color="blue")
    
    # Compute model uncertainty using MCMC samples
    n_samples_to_draw = 1000
    sample_inds = np.random.choice(len(samples[0]), n_samples_to_draw, replace=False)
    model_curves = np.array([King(rfit, *samples[0][i]) for i in sample_inds])

    # Compute percentiles at each radius
    lower = np.percentile(model_curves, 16, axis=0)
    median = np.percentile(model_curves, 50, axis=0)
    upper = np.percentile(model_curves, 84, axis=0)
    
    results_dir = '../results/visualisations/radial_profile_fits/'
    ensure_dir(results_dir)

    # Plot shaded 1-sigma region
    ax1.fill_between(rfit, lower, upper, color='blue', alpha=0.3, label=r'$1\sigma$ band')
    
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$r \rm \ [arcmin]$', size=16)
    ax1.set_ylabel(r'$N_{\rm stars}/\rm arcmin^2$', size=16)
    ax1.legend()
    ax1.set_xlim(r_lower, r_upper)
    fig.suptitle(f'{clus_name}', size=24)
    fig.tight_layout()
    plt.savefig(f'{results_dir}{clus_name} - MCMC fit.png')
    plt.close()

    # Compute Concentration Parameter
    c = np.log10(np.abs(best_fit[2] / best_fit[1]))

    # calculate uncertainty of c:
    u_c = np.sqrt( (best_fit[1]/error_ab[1])**-2 + (best_fit[2]/error_ab[2])**-2 + (best_fit[3]/error_ab[3])**-2 ) 
    print(f"Best-fit parameters: k={best_fit[0]:.3f}, r_c={best_fit[1]:.3f}, r_t={best_fit[2]:.3f}, eta={best_fit[3]:.3f}")
    print(f"Concentration parameter: c = {c:.3f} +/- {u_c:.3f}")

    labels = [r'$k \rm \ [arcmin^{-2}]$', 
              r'$r_c \ \rm [arcmin]$', 
              r'$r_t \ \rm [arcmin]$', 
              r'$\eta \rm \ [arcmin^{-2}]$']
    
    fig_corner = corner.corner(samples[0], labels=labels, quantiles=[0.16,0.5,0.84], truths=best_fit,
                               show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12}, 
                               smooth=True, 
                               label_kwargs={'fontsize':14})

    fig_corner.suptitle(f'{clus_name}', size=24)
    plt.savefig(f'{results_dir}{clus_name} - corner.png')
    
    plt.close()
    
def plx_dist(file_path):
    """Returns median parallax-derived distance, and 1-sigma errors associated to 16th and 84th percentile distances"""
    
    # open file:
    T = ascii.read(file_path)
    
    #extract cluster name:
    clus_name = file_path.split('/')[-1][:-5]
    
    # select only high membership probability sources. In this case we use 90%
    prob_ind = T['gmm_mem_prob']>0.9

    T = T[prob_ind]

    plx = np.array(T['parallax'])
    eplx = np.array(T['e_parallax'])

    def log_likelihood(D, plx, eplx):
        model = 1 / D  # Model parallax = 1/distance
        residuals = (plx - model) / eplx
        return -0.5 * np.sum(np.log(2 * np.pi * eplx**2) + residuals**2)

    def log_prior(D):
        # Gamma distribution - similar to Bailer-Jones et al. (2021 
        D_s = 9.14 # scale distance, corresponding to median distance of all GCs - reference Baumgardt and Vasiliev (2021)
        if D>0:
            return -np.log(2*D_s) - 2*np.log(D_s/D) - D/D_s
        return -np.inf

    def log_probability(D,plx, eplx):
        lp = log_prior(D)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(D, plx, eplx)


    # Set up MCMC sampler:
    nwalkers = 10  # Number of MCMC walkers
    ndim = 1  # We are only fitting distance D
    nsteps = 10000  # Number of MCMC steps

    # Initial positions: Start walkers near an initial guess (e.g., 1/plx)
    initial_distances = 1 / (np.median(plx)) + 1e-4*np.random.randn(nwalkers)

    initial_positions = np.abs(initial_distances[:, np.newaxis])  # Ensure positive distances

    # Run MCMC:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(plx, eplx))
    sampler.run_mcmc(initial_positions, nsteps, progress=True)

    # Extract samples:
    flat_samples = sampler.get_chain(discard=2000, thin=10, flat=True)
    
    save_dir = '../results/visualisations/distances/parallax/'
    ensure_dir(save_dir)
    
    # Plot the posterior distribution:
    fig = corner.corner(flat_samples, labels=[r"$D \rm [kpc]$"], quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_fmt=".3f", title_kwargs={"fontsize":12}, label_kwargs={"fontsize": 16})
    fig.tight_layout()
    plt.savefig(f'{save_dir}{clus_name}.png')
    plt.close()

    # Report the best-fit distance:
    dist_median = np.median(flat_samples)
    dist_err_low = dist_median - np.percentile(flat_samples, 16)
    dist_err_high = np.percentile(flat_samples, 84) - dist_median
    
    return clus_name, dist_median, dist_err_low, dist_err_high

def pm_mle(file_path):
    """
    Computes the proper motion (PM) dispersion of a cluster.
    
    Parameters:
    file_path : str or os.path() like
                - Path to cluster membership list file
    
    Returns:
    tuple (clus_name, disp_med, disp_err_lo, disp_err_hi)
    - Cluster name, Median PM dispersion, and 1-sigma errors associated with 16th and 84 percentile values
    
    """
    T = ascii.read(file_path)
    clus_name = file_path.split('/')[-1][:-5]
    
    # select only high membership probability sources. In this case we use 90%
    prob_ind = T['gmm_mem_prob']>0.9

    T = T[prob_ind]
    data = np.array(T['pmra', 'pmdec', 'e_pmra', 'e_pmdec'].to_pandas())

    pm = np.sqrt(data[:,0]**2 + data[:,1]**2)
    e_pm = np.sqrt( ( (data[:,0]*data[:,2])**2 + (data[:,1]*data[:,3])**2 )/ (pm**2) )

    def log_likelihood(theta, mu, e_mu):
        """
        Log-likelihood function to estimate intrinsic proper motion dispersion.

        Parameters:
        theta : tuple (mu_bar, sigma_int)
            - mu_bar: Mean proper motion (not needed for dispersion estimation)
            - sigma_int: Intrinsic proper motion dispersion
        mu : np.array
            - Observed proper motions
        e_mu : np.array
            - Measurement uncertainties

        Returns:
        Log-likelihood value.
        """
        mu_bar, sigma_int = theta

        # Ensure dispersion is positive
        if sigma_int <= 0:
            return -np.inf  

        sigma_sys = 0.02
        # Observed dispersion model
        sigma_obs2 = sigma_int**2 + e_mu**2 + sigma_sys**2
        log_like = -0.5 * np.sum(np.log(2 * np.pi * sigma_obs2) + (mu - mu_bar)**2 / (2 * sigma_obs2))

        return log_like

    def run_emcee(mu, e_mu, n_walkers=10, n_steps=5000):
        """
        Runs MCMC to estimate intrinsic proper motion dispersion.

        Parameters:
        mu : np.array - Observed proper motions
        e_mu : np.array - Measurement uncertainties
        n_walkers : int - Number of walkers
        n_steps : int - Number of MCMC steps

        Returns:
        samples 
        """
        ndim = 2  # Parameters: mu_bar, sigma_int
        initial_guesses = np.array([np.mean(mu), np.std(mu)])  # Start with observed mean and std

        # Small random perturbations
        pos = initial_guesses + 1e-4 * np.random.randn(n_walkers, ndim)  

        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood, args=(mu, e_mu))
        sampler.run_mcmc(pos, n_steps, progress=True)
        samples = sampler.get_chain(discard=1000, thin=10, flat=True)  # Discard burn-in & thin samples
        
        return samples
    
    samples = run_emcee(pm, e_pm)
    
    results_dir = f'../results/visualisations/pm_disp_fits/'
    ensure_dir(results_dir)
    
    med_vals = np.median(samples, axis=0)
    
    fig = corner.corner(samples, labels = [r'$ \bar{\mu} \ \rm [mas \ yr^{-1}]$', r'$\sigma_\mu \ \rm [mas \ yr^{-1}]$'], 
                        label_kwargs = {'fontsize':14}, quantiles = [0.16, 0.5, 0.84], truths=med_vals, smooth=True,
                        show_titles=True, title_fmt='.3f')

    fig.suptitle(f'{clus_name}', x = 0.75 , y=0.8, size=24)
    plt.savefig(f'{results_dir}{clus_name}.png')
    plt.close()

    disp_med = np.percentile(samples, 50, axis = 0)[-1]
    disp_err_lo = disp_med - np.percentile(samples, 16, axis = 0)[-1]
    disp_err_hi = np.percentile(samples, 84, axis = 0)[-1] - disp_med
    
    return clus_name, disp_med, disp_err_lo, disp_err_hi