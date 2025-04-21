import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
plt.rcParams.update({"text.usetex": True})

# Function for opening data, applying filters and correcting for offsets:
from zero_point import zpt
zpt.load_tables() 

import threading
from hdbscan import HDBSCAN

def process_data(path_to_file):
    """Carries out data culling with zero-point correction, ensuring source_id alignment"""
    file = fits.open(f'{path_to_file}')
    data = file[1].data
    
    # Extract necessary fields
    plx_over_error = data['parallax'] / data['parallax_error']
    m_G = data['phot_g_mean_mag']
    bp_rp = data['bp_rp']
    ruwe = data['ruwe']
    source_id = data['source_id']
    astrometric_params_solved = data['astrometric_params_solved']
    phot_bp_rp_excess_factor = data['phot_bp_rp_excess_factor']
    
    # phot_bp_rp_excess_factor corrections from Riello+2021
    def compute_corrected_bp_rp_excess_factor(bp_rp_e_f, bp_rp):
        # Initialize an array to store the corrected values, keeping original order
        corrected_values = np.empty_like(bp_rp_e_f)

        # Apply corrections based on ranges of `bp_rp`
        x = bp_rp
        # Segment 1: bp_rp < 0.5
        mask1 = x < 0.5
        corrected_values[mask1] = bp_rp_e_f[mask1] - (1.154360 + 0.033772 * x[mask1] + 0.032277 * x[mask1]**2)

        # Segment 2: 0.5 <= bp_rp < 4
        mask2 = (x >= 0.5) & (x < 4)
        corrected_values[mask2] = bp_rp_e_f[mask2] - (1.162004 + 0.011464 * x[mask2] + 
                                                      0.049255 * x[mask2]**2 + 0.005879 * x[mask2]**3)
        # Segment 3: bp_rp > 4
        mask3 = x > 4
        corrected_values[mask3] = bp_rp_e_f[mask3] - (1.057572 + 0.140537 * x[mask3])
        return corrected_values
    
    # phot_bp_rp_excess_factor_corr = compute_corrected_bp_rp_excess_factor(phot_bp_rp_excess_factor, bp_rp) # C* in Gaia
    
    # 1-sigma scatter for a sample of well behaved isolated stellar sources with good quality Gaia photometry.
    sigma_C = 0.0059898 + 8.817481e-12*m_G**7.618399 
    
    
    
    # Apply the primary filter
    ind = (plx_over_error > -3) & (~np.isnan(bp_rp)) & (~np.isnan(m_G)) & (astrometric_params_solved > 3) & (ruwe < 1.15) 
    #& (~np.isnan(phot_bp_rp_excess_factor))
    
    # Parallelize data processing using threads
    phot_lock = threading.Lock()
    ast_lock = threading.Lock()
    
    def process_photometry():
        nonlocal phot_array
        with phot_lock:
            phot_array = np.vstack((source_id[ind], m_G[ind], bp_rp[ind])).T
    
    def process_astrometry():
        nonlocal ast_array
        with ast_lock:
            ast_array = np.vstack((data['l'][ind], 
                                   data['b'][ind], 
                                   data['parallax'][ind], 
                                   data['pmra'][ind], 
                                   data['pmdec'][ind], 
                                   data['parallax_error'][ind], 
                                   data['pmra_error'][ind], 
                                   data['pmdec_error'][ind],
                                   source_id[ind])).T

    phot_array = None
    ast_array = None
    
    # Create threads for parallel processing
    phot_thread = threading.Thread(target=process_photometry)
    ast_thread = threading.Thread(target=process_astrometry)
    
    # Start threads
    phot_thread.start()
    ast_thread.start()
    
    # Wait for threads to finish
    phot_thread.join()
    ast_thread.join()
    
    # Apply additional filters to corrected data
    m_G_filtered = m_G[ind]
    
    zpvals = zpt.get_zpt(m_G_filtered, data['nu_eff_used_in_astrometry'][ind], data['pseudocolour'][ind], data['ecl_lat'][ind], astrometric_params_solved[ind], _warnings=False)
    
    nan_ind = ~np.isnan(zpvals)
    plx_true = ast_array[:, 2][nan_ind] - zpvals[nan_ind]
    
    def pmra_corr(pmra):
        corrected_values = np.empty_like(pmra)
        mask1 = m_G_filtered<13
        mask2 = m_G_filtered>=13
        corrected_values[mask1] = pmra[mask1] - np.random.normal(loc = 0.01, scale = 0.02, size = pmra[mask1].size)
        corrected_values[mask2] = pmra[mask2] - 0
        
        return corrected_values
    
    # Adjust for pm overcorrection
    pmra_true = pmra_corr(ast_array[:,3])[nan_ind]
    
    # Adjust for parallax over-correction!
    plx_true -= np.random.normal(loc = 0.01, scale = 0.003, size = plx_true.size)
    
    # Apply final corrections
    ast_array = np.hstack(( ast_array[:, :2][nan_ind], np.array([plx_true]).T, np.array([pmra_true]).T, ast_array[:, 4:][nan_ind] ))
    
#     phot_bp_rp_excess_factor_corr = compute_corrected_bp_rp_excess_factor(phot_bp_rp_excess_factor[ind][nan_ind], bp_rp[ind][nan_ind])
    
#     N = 5
#     sigma_C = 0.0059898 + 8.817481e-12*m_G[ind][nan_ind]**7.618399 

    # Final filtering and source_id alignment
    final_ind = (data['astrometric_excess_noise_sig'][ind][nan_ind] <= 2) & \
                (data['ipd_gof_harmonic_amplitude'][ind][nan_ind] <= np.exp(0.18 * (m_G_filtered[nan_ind] - 33))) & \
                (data['visibility_periods_used'][ind][nan_ind] >= 10) & \
                (data['ipd_frac_multi_peak'][ind][nan_ind] < 2)
#     & np.abs(phot_bp_rp_excess_factor_corr) < N*sigma_C)
    
    ast_array = ast_array[final_ind]
    phot_array = phot_array[nan_ind][final_ind]
    
    file.close()
    
    # Return the filtered data arrays, ensuring source_ids align
    return ast_array, phot_array

def partition_data(ast_data, phot_data, L_desired):
    """Partitions a given rectangular region of data into subdivisions each of the provided length L"""
    l_min, b_min = np.round(ast_data[:,0].min()), np.round(ast_data[:,1].min())
    l_max, b_max = np.round(ast_data[:,0].max()), np.round(ast_data[:,1].max())
    l_array = np.arange(l_min,l_max+L_desired,L_desired)
    b_array = np.arange(b_min,b_max+L_desired,L_desired)
    ast_data_part = {}
    phot_data_part = {}
    for i in range(l_array.size-1):
        for j in range(b_array.size-1):
            ind = (ast_data[:,0]>l_array[i])&(ast_data[:,0]<l_array[i+1])&(ast_data[:,1]>b_array[j])&(ast_data[:,1]<b_array[j+1])
            ast_data_part[f'{i}{j}'] = ast_data[ind,:]
            phot_data_part[f'{i}{j}'] = phot_data[ind,:]
    return ast_data_part, phot_data_part

def rescale(data):
    '''rescales data such that it has mean = 0 and stdev = var = 1'''
    rescaled = []
    for x in data.T:
        x_resc = (x - np.mean(x))/np.std(x)
        rescaled.append(x_resc)
    return np.array(rescaled).T
    

def find_good_labels(all_labels, all_pers, all_radii):
    """finds unique labels associated with possible clusters based on cluster persistences and angular radii"""
    all_good_labels = {} # dict of all good label indices
    for region in all_labels: #for a given region in the partitioned astrometric array (space)
        good_labels = {} # dict that stores all good cluster labels for a given
        for m in all_radii[region]:
            radii = all_radii[region][m]
            pers  = all_pers[region][m]
            g_ind = np.where((radii<0.20)&(pers>0.05))[0] 
            good_labels[m] = g_ind
        all_good_labels[region] = good_labels
    return all_good_labels

def cluster_cleaner(all_labels, all_good_labels):
    """Returns cleaned labels based on the good labels we previously found. 
    If a label is not 'good', it is relabelled to -1 """
    all_cleaned_labels = {}
    for region in all_labels:
        cleaned_labels = {}
        for m in all_labels[region]:
#             unique_labels = set(all_labels[region][m])-{-1}
            good_labels = all_good_labels[region][m]
            labels = all_labels[region][m]
            new_labels = []
            if good_labels.size==0:
                new_labels = (np.ones(labels.shape)*-1).astype(int)
            else:
                for l in labels:
                    if l in good_labels:
                        new_labels.append(l)
                    else:
                        new_labels.append(-1)
                new_labels = np.array(new_labels)
            cleaned_labels[m] = new_labels
        all_cleaned_labels[region] = cleaned_labels
    return all_cleaned_labels

def compute_angular_radius(xy_array): 
    """Returns cluster radius containing 50% of members"""
    x_clus, y_clus = xy_array[:,0], xy_array[:,1]
    distances = np.sqrt( (x_clus - x_clus.mean())**2  + (y_clus - y_clus.mean())**2 )
    r50 = np.percentile(distances, 50)
    return r50

def clusterer(path_to_file, partition_size):
    """Runs HDBSCAN for multiple mcls on a specified file/sky area 
    that is subdivided according to a partition_size provided by the user """
    fname = path_to_file.split('/')[-1][:-15]
    disc_loc = path_to_file.split('/')[-2]
    ast_array, phot_array = process_data(path_to_file)
        
    print('Filename:', fname)
#     open and process data
    ast_array_p, phot_array_p = partition_data(ast_array, phot_array, partition_size)
                                                  
    all_data_resc  = {}
#     rescale data:
                                               
    for region in ast_array_p:
        all_data_resc[region] = rescale(ast_array_p[region][:,:5])
        
    # Running HDBSCAN on each subdivision

    mcls = np.arange(6, 16, 1) # minclustersize values to feed to HDBSCAN: 5 to 15
    t0_h = time.time()
    all_labels_hdb = {}
    all_prob = {}
    all_pers = {}
    all_radii = {}
    for region in ast_array_p:
#         print(f'Region {region}:')
        hdb = {}
        labels = {}
        prob = {}
        pers = {}
        radii = {}
#         start = time.time()
        for m in mcls:
            hdb[m] = HDBSCAN(min_cluster_size=int(m)).fit(all_data_resc[region])
            labels[m]=hdb[m].labels_
            prob[m] = hdb[m].probabilities_
            pers[m] = hdb[m].cluster_persistence_
            radius_list = []
            n_clusters = len(set(labels[m])) - (1 if -1 in labels[m] else 0)
            unique_labels = set(labels[m]) - {-1}
            for label in unique_labels:
                radius_list.append(compute_angular_radius(ast_array_p[region][labels[m]==label,0:2]))
            radii[m] = np.array(radius_list)
    #         print(f'For min_cluster_size = {m}:')
    #         print('Number of clusters', n_clusters)
    #         print_cluster_table(data=ast_array_p[region][:,:-1], labels=labels[m], prob[m], 0.5)
    #         print("")
#         end = time.time()
#         duration = end-start
#         print(f'Runtime: {duration}s') 
        all_labels_hdb[region] = labels
        all_prob[region] = prob
        all_pers[region] = pers
        all_radii[region] = radii
    t1_h = time.time()
    print(f'Total runtime of algorithm (seconds):{t1_h-t0_h}')
    all_good_labels = find_good_labels(all_labels_hdb, all_pers, all_radii)
    all_cleaned_labels = cluster_cleaner(all_labels_hdb,all_good_labels)
    hdb_save_dir = f'../data/clustering/clus_info'

    labels_name = fname+f'_labels_{partition_size}.npy'
    pers_name = fname+f'_pers_{partition_size}.npy'
#     prob_name = fname+f'_prob_{partition_size}.npy'
    np.save(f'{hdb_save_dir}/labels/{labels_name}', all_cleaned_labels, allow_pickle=True)
    np.save(f'{hdb_save_dir}/pers/{pers_name}', all_pers, allow_pickle=True)
#     np.save(f'{hdb_save_dir}/prob/{prob_name}', all_prob, allow_pickle=True)
    print('Clusterer done!\n')