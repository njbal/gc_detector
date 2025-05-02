import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO (1), WARNING (2), and ERROR (3) messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: turn off oneDNN custom ops if needed

import sys
sys.dont_write_bytecode = True

import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
plt.rcParams.update({"text.usetex": True})
# # Function for opening data, applying filters and correcting for offsets:
from zero_point import zpt
zpt.load_tables() 
import threading
from hdbscan import HDBSCAN
from matplotlib.ticker import AutoMinorLocator
from sklearn.mixture import GaussianMixture

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

import glob
def delete_files_in_directory(directory_path):
    try:
        files = glob.glob(os.path.join(directory_path, '*'))
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
#         print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")

from clusterer import process_data, partition_data, compute_angular_radius, rescale
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model, load_model
import cv2

from sklearn.neighbors import NearestNeighbors
from scipy.stats import mannwhitneyu

def ax_config(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(labelsize=12, direction='in', which='major', length = 5)
    ax.tick_params(which='minor', direction='in', length = 2)
    ax.tick_params(top=True, right=True, which='both', direction='in')
    ax.set_xlabel(r'$\rm BP - RP \ [mag]$ ', size=20)
    ax.set_ylabel(r'$G \ \rm [mag]$', size=20)
    ax.invert_yaxis()

def gen_cmd(path_to_file, partition_sizes):
    """Runs HDBSCAN for multiple mcls on a specified file/sky area 
    that is subdivided according to a partition_size provided by the user """
    fname = path_to_file.split('/')[-1][:-15]
    disc_loc = path_to_file.split('/')[-2]
    ast_array, phot_array = process_data(path_to_file)
        
    print('Filename:', fname)
#     open and process data
    
    save_dir = '../data/cnn_classification/cmd/'
    for L in partition_sizes:
        ast_array_p, phot_array_p = partition_data(ast_array, phot_array, L)
        
        labels_path = f'../data/clustering/clus_info/labels/{fname}_labels_{L}.npy'
        all_labels = np.load(labels_path, allow_pickle=True)[()]
        
        pers_path = f'../data/clustering/clus_info/pers/{fname}_pers_{L}.npy'
        all_pers = np.load(pers_path, allow_pickle=True)[()]
#         prob_path = f'../data/clustering/clus_info/prob/{fname}_prob_{L}.npy'

        for sd in ast_array_p: # sd - subdivision
            for m in all_labels[sd]: # m - minclustersize used
                labels = all_labels[sd][m]
                unique_labels = set(labels)-{-1}
                
                if len(unique_labels)==0:
                    continue
                else:
                    # below code checks if overdensity satisifies R_50<0.2 deg and persistence>0.1
                    # If so, CMD is generated. If not, overdensity is ignored.
                    for label in unique_labels:
                        if all_pers[sd][m][label]>=0.1:
                            lb_array = ast_array_p[sd][labels==label, 0:2]
                            rad = compute_angular_radius(lb_array)
#                             print('radius =', rad)
                            if rad<0.2:
#                                 print('yes')
                                spec_g, spec_bp_rp = phot_array_p[sd][labels==label, 1:].T
                                fig, ax = plt.subplots(figsize=(6,6))
                                ax.scatter(spec_bp_rp, spec_g, s=1, color='black')
                                np.random.seed(42)
                                xmin = spec_bp_rp.min() + np.round(np.random.normal(-0.3,0.3,1)[-1], 2)
                                xmax = spec_bp_rp.max() + np.round(np.random.normal(0.3,0.3,1)[-1], 2)

                                ymin = spec_g.min() + np.round(np.random.normal(-0.3,0.2,1)[-1], 2)

                                ax.set_xlim(xmin, xmax)
                                ax.set_ylim(ymin,21)
                                ax_config(ax)
                                plt.axis('off')
                                fig.tight_layout()
                                save_name = fname + f'_{L}_{sd}_{m}_{label}.png' 
                                plt.savefig(f'{save_dir}{save_name}')
                                plt.close() 
                            else:
                                continue
                        else:
                            continue
                            
                            
def cnn(path_to_cmds, cnn_model, class_threshold=0.5):
    """Carries out CNN classification of CMDs. 
    Creates a able consisting of unique cluster identifiers and corresponding classification probabilities."""

    steps = np.arange(0, len(os.listdir(path_to_cmds)), 1000)
    steps = np.append(steps, len(os.listdir(path_to_cmds)))

    imsize=128
    all_labels = np.array([])
    all_ids = np.array([])
    all_pos_id = np.array([])
    all_pos_labels = np.array([])
    for i in range(steps.size-1):
        low  = steps[i]
        high = steps[i+1]
        X_hdb = []
        n_images = high-low
        for j, filename in enumerate(os.listdir(path_to_cmds)):
            if low<=j<high:
                img = cv2.imread(os.path.join(path_to_cmds, filename), cv2.IMREAD_GRAYSCALE)
                newimg = cv2.resize(img,(imsize,imsize))
                X_hdb.append(newimg)
            else:
                continue
        X_hdb = np.array(X_hdb).reshape(n_images, imsize, imsize, 1)/255
        new_model = tf.keras.models.load_model(cnn_model)
        labels = new_model.predict(X_hdb)
        labels = labels.flatten()

        all_ids = np.append(all_ids, np.array(os.listdir(path_to_cmds)[low:high], dtype=str))
        all_labels = np.append(all_labels, labels)
        positive_labels = labels[labels>=class_threshold]
        all_pos_labels = np.append(all_pos_labels, positive_labels)
        positive_id = np.array(os.listdir(path_to_cmds)[low:high], dtype=str)[(labels>class_threshold)]  
        all_pos_id = np.append(all_pos_id, positive_id)

    np.save('../data/cnn_classification/npy_files/pos_ids.npy', all_pos_id, allow_pickle=True)
    clus_sig_table = Table(data = np.vstack((all_pos_id, all_pos_labels)).T, 
                           names=['Cluster ID', 'p_class'], 
                           dtype = [str,'float64'])
    clus_sig_table.write('../data/cnn_classification/prelim_tables/p_class-table.ecsv', format='ascii.ecsv', overwrite=True)
        
    n_between = all_ids[(all_labels>=class_threshold)&(all_labels<=0.99)].size
    n_over = all_ids[all_labels>0.99].size
    n_under = all_ids[all_labels<class_threshold].size
        
    print('Number of overdensities with class_threshold < class_prob < 0.99:', n_between)

    print('Number of overdensities with class_prob > 0.99:                  ', n_over)

    print('Number of overdensities with class_prob < class_threshold:       ', n_under)
        
    print('Number of overdensities total:                                   ', all_ids.size)
        

def unify_repetitions(path_to_pos_ids):
    """Unifies all repetitions of a given overdensity. Returns a dictionary with keys given by distinct cluster IDs and items
    given by arrays of Gaia source IDs of the corresponding members."""
    # collect cluster info such as coordinates
    pos_ids = np.load(f'{path_to_pos_ids}pos_ids.npy', allow_pickle=True)

    id_infos = {}

    unique_files = set()
    for x in pos_ids:
        spl_id = x.split('_')
        file = '{0}_{1}_{2}_{3}'.format(*spl_id[:4])
        unique_files.add(file)

    for f in unique_files:
        infos = []
        for x in pos_ids:
            spl_id = x.split('_')
            if f in x:
                label = spl_id[-1][:-4]
                spec_info = [*spl_id[-4:-1], label]
                infos.append(spec_info)
            else:
                continue
        id_infos[f] = np.array(infos)

    # extract lb positions and corresp source ids:
    lb_dict = {}
    source_id_dict = {}

    for f, infos in id_infos.items():
        disc_position = 'below' if '-' in f else 'above'

        data_path = f'../data/clustering/{disc_position}_disc/{f}-result.fits.gz'
        ast_array, phot_array = process_data(data_path)

        for info in infos:
            L = int(info[0])
            ast_array_p, phot_array_p = partition_data(ast_array, phot_array, L)
            labels_path = f'../data/clustering/clus_info/labels/{f}_labels_{L}.npy'
            all_labels = np.load(labels_path, allow_pickle=True)[()]
            sd = info[1] # subdivision/subregion
            m = info[2] # mcls
            p = info[3] # label
    #         print(f'{f}_{info[0]}_{sd}_{m}_{p}')
            labels = all_labels[sd][int(m)]
            lb_mean = np.mean(ast_array_p[sd][labels==int(p)][:,0:2], axis=0)
            lb_dict[f'{f}_{info[0]}_{sd}_{m}_{p}'] = lb_mean
            source_id_dict[f'{f}_{info[0]}_{sd}_{m}_{p}'] = ast_array_p[f'{sd:02}'][labels==int(p)][:,-1]
                
    # apply criterion again to find repetitions
    def clus_distance_calc(coord1, coord2):
        """Calculate the distance between two points (l, b) in degrees."""
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

    def group_duplicates(clusters, threshold=0.6):
        """
        Groups clusters that are considered duplicates within a threshold distance.

        Parameters:
            clusters (dict): Dictionary of cluster labels and their coordinates.
            threshold (float): Distance threshold for considering duplicates.

        Returns:
            dict: A dictionary where keys are unique clusters and values are lists of duplicates.
        """
        grouped_clusters = {}
        seen = set()

        for label1, coord1 in clusters.items():
            if label1 in seen:
                continue

            # Initialize a group for this cluster
            group = [label1]
            seen.add(label1)

            for label2, coord2 in clusters.items():
                if label2 != label1 and label2 not in seen:
                    distance = clus_distance_calc(coord1, coord2)
                    if distance < threshold:
                        group.append(label2)
                        seen.add(label2)

            # Use the first label in the group as the key
            grouped_clusters[label1] = group

        return grouped_clusters

    # Group duplicates
    grouped_overdensities = group_duplicates(lb_dict)
    print(f"Number of distinct overdensities: {len(grouped_overdensities)}")

    final_source_id_dict = {}

    for label, group in grouped_overdensities.items():
        source_ids = np.array([])
        for clus in group:
            source_ids = np.append(source_ids, source_id_dict[clus])
        source_ids_set = set(source_ids)
        final_source_id_dict[label] = np.array(list(source_ids_set))

    np.save(f'{path_to_pos_ids}final_source_ids_dict.npy', final_source_id_dict)

def gmm_membership(spec_clus, source_ids_dict, clus_mem_lists_path):
    """Returns cluster memnbership lists of a given overdensity with a cluster_id of spec_clus"""

    spl = spec_clus.split('_')
    print(spec_clus)
    file = '{}_{}_{}_{}'.format(*spl[:4])
    disc_position = 'above' if '-' not in file else 'below'
    file_path = f'../data/clustering/{disc_position}_disc/{file}-result.fits.gz'
    ast_array, phot_array = process_data(file_path)

    clus_source_ids = source_ids_dict[spec_clus]

    clus_id_inds = np.intersect1d(ast_array[:,-1], clus_source_ids, return_indices=True)
    clus_ast_array, clus_phot_array = ast_array[clus_id_inds[1]], phot_array[clus_id_inds[1]]
    
    gmm_data = clus_ast_array[:,0:5]
    GMM = GaussianMixture(n_components=2, random_state=42, covariance_type='full').fit(gmm_data)

    prob = GMM.predict_proba(gmm_data)
    labels = GMM.predict(gmm_data)

    main_component = np.argmax(np.bincount(labels))  # Identify the largest component
    minor_component = np.argmin(np.bincount(labels))
    
        
    c = SkyCoord(l = clus_ast_array[:,0]*u.deg, b = clus_ast_array[:,1]*u.deg, frame = 'galactic')
    ra_dec_array = np.vstack((c.icrs.ra.deg, c.icrs.dec.deg)).T

    
    column_names = ['ra', 'dec', 'l', 'b', 'parallax', 'pmra', 'pmdec', 
                    'e_parallax', 'e_pmra', 'e_pmdec', 'g', 'bp_rp',
                    'source_id', 'gmm_mem_prob']
    all_data = np.hstack((ra_dec_array, 
                          clus_ast_array[:,:-1], 
                          clus_phot_array[:,1:], 
                          np.array([clus_phot_array[:,0]]).T,
                          np.array([prob[:, main_component]]).T))#[labels==main_component]
            
    all_dtype = ('float64_'*12)[:-1].split('_')
    all_dtype.extend(['int64', 'float64'])
    all_units = [u.deg, u.deg, u.deg, u.deg, u.mas, u.mas/u.yr, u.mas/u.yr, 
                 u.mas, u.mas/u.yr, u.mas/u.yr, u.mag, u.mag, None, None]
    
    t = Table(data = all_data, names=column_names, dtype  = all_dtype, units=all_units)
    t.write(f'{clus_mem_lists_path}{spec_clus}.ecsv', format='ascii.ecsv', overwrite=True)
    
    
def xmatch(clus_mem_lists_path):
    """Positionally matches our overdensities with known objects (e.g. GCs, OCs, dSphs)"""
    
    vb21 = ascii.read('vb21-gc.ecsv')
    hr24 = ascii.read('hr24-oc.ecsv')
    dr20 = ascii.read('dr20-dsph.ecsv')
    vra, vdec = vb21['RAJ2000'], vb21['DEJ2000']
    hra, hdec = hr24['RAJ2000'], hr24['DEJ2000']
    dra, ddec = dr20['RAJ2000'], dr20['DEJ2000']

    vname = vb21['Name']
    hname = hr24['Name']
    dname = dr20['Name']

    sra, sdec, sname = 298.1687500, -22.0680556, 'Sagittarius II'
    
    cand_num = 0
    gc_num = 0
    oc_num = 0
    dsph_num = 0
    def xmatch_vb21(clus_ra, clus_dec):
        for i, xy in enumerate(zip(vra, vdec)):
            if (xy[0] - clus_ra)**2 + (xy[1] - clus_dec)**2 <= 0.5**2:
                return vname[i]
        if (sra - clus_ra)**2 + (sdec - clus_dec)**2 <= 0.5**2:
            return sname
        return False

    def xmatch_hr24(clus_ra, clus_dec):
        for j, uv in enumerate(zip(hra, hdec)):
            if (uv[0] - clus_ra)**2 + (uv[1] - clus_dec)**2 <= 0.5**2:
                return hname[j].replace('_', ' ')
        return False

    def xmatch_dsph(clus_ra, clus_dec):
        for k, qw in enumerate(zip(dra, ddec)):
            if (qw[0] - clus_ra)**2 + (qw[1] - clus_dec)**2 <= 0.5**2:
                return dname[k]
        return False

    for cl in np.sort(os.listdir(clus_mem_lists_path)):
        # cl is the cluster ID
        t = ascii.read(f'{clus_mem_lists_path}{cl}')
        clus_ra, clus_dec = np.array(t['ra']).mean(), np.array(t['dec']).mean()

        val_vb21 = xmatch_vb21(clus_ra, clus_dec)
        val_hr24 = xmatch_hr24(clus_ra, clus_dec)
        val_dr20 = xmatch_dsph(clus_ra, clus_dec)
        if val_vb21!=False:
            print(f'{cl[:-5]}:{val_vb21}')
            os.rename(f'{clus_mem_lists_path}{cl}', f'{clus_mem_lists_path}{cl[:-5]}_{val_vb21}.ecsv')
            gc_num+=1
        elif val_hr24!=False:
            print(f'{cl[:-5]}:{val_hr24}')
            os.rename(f'{clus_mem_lists_path}{cl}', f'{clus_mem_lists_path}{cl[:-5]}_{val_hr24}.ecsv')
            oc_num+=1
        elif val_dr20!=False:
            print(f'{cl[:-5]}:{val_dr20}')
            os.rename(f'{clus_mem_lists_path}{cl}', f'{clus_mem_lists_path}{cl[:-5]}_{val_dr20}.ecsv')
            dsph_num+=1

        else:
            cand_num+=1
            print(f'{cl[:-5]}:cand{cand_num}')
            os.rename(f'{clus_mem_lists_path}{cl}', f'{clus_mem_lists_path}{cl[:-5]}_cand{cand_num}.ecsv')
    
    print('\nObject tally:')
    print('N_GC:   ', gc_num)
    print('N_OC:   ', oc_num)
    print('N_dSph: ', dsph_num)
    print('N_cand: ', cand_num)
    
def cst(clus_mem_lists_path):
    """Performs cluster significance test on all clusters and candidates. Creates an associated data table."""
    unique_files = set()
    for x in np.sort(os.listdir(clus_mem_lists_path)):
        spl = x.split('_')
        fname = '{0}_{1}_{2}_{3}'.format(*spl[:4])
        
        unique_files.add(fname)

    pval_array = []
    i = 0
    k = 10 # apply kNN with N=10 to compute kNNDs

    for file in unique_files:
        disc_position = 'below' if '-' in file else 'above' 
        data_path = f'../data/clustering/{disc_position}_disc/{file}-result.fits.gz'
        ast_array, phot_array = process_data(data_path)

        for x in np.sort(os.listdir(clus_mem_lists_path)):
            if file in x:
                # open clus data, compute r_50 and extract all points with r_50 of cluster centre
                spl = x.split('_')
                clus_name = spl[-1][:-5]
                pers_path = r'../data/clustering/clus_info/pers/{}_{}_{}_{}_pers_{}.npy'.format(*spl[:5])
                pers_data = np.load(pers_path, allow_pickle=True)[()]
                pers_val = pers_data[f'{spl[5]}'][int(spl[6])][int(spl[7])]

                t = ascii.read(f'{clus_mem_lists_path}{x}')
                clus_source_id = np.array(t['source_id'], dtype='int64')
                clus_l, clus_b = np.array(t['l']), np.array(t['b'])
                if len(clus_b)<=k:
                    print(f'{clus_name}:insufficient no. of sources')
                    pval_array.append([clus_name, pers_val, 99]) # flag signalling insufficient number of sources in cluster
                else:
                    clus_centre = np.array([clus_l.mean(), clus_b.mean()])
                    radius = compute_angular_radius(np.array([clus_l,clus_b]).T)
                    ind1 = (ast_array[:,0] - clus_centre[0])**2 + (ast_array[:,1] - clus_centre[1])**2 <= (3*radius)**2
                    spec_ast_array = ast_array[ind1]
                    
                    spec_source_id = np.int64(spec_ast_array[:,-1])
                    
                    # Rescale data:
                    spec_ast_array_resc = rescale(spec_ast_array[:,:5])

                    comm_id, comm_ind1, comm_ind2 = np.intersect1d(clus_source_id, spec_source_id, return_indices=True)
                    back_array = np.delete(spec_ast_array_resc, comm_ind2, 0)
                    clus_array = spec_ast_array_resc[comm_ind2]

                    
                    if clus_array.shape[0]<k+1:
                        print(f'{clus_name}:insufficient no. of sources')
                        pval_array.append([clus_name, pers_val, 99])
                    else:
                        # apply kNN with N=10 to compute kNNDs:
                        neigh = NearestNeighbors(n_neighbors=k+1)

                        # clus NNDs:
                        neigh.fit(clus_array)
                        clus_dist, clus_indices = neigh.kneighbors(clus_array, return_distance=True)
                        clus_10NND = np.vstack((clus_dist[:,-1], np.ones(clus_dist.shape[0]))).T

                        # bkg NNDs:
                        neigh = NearestNeighbors(n_neighbors=k+1)
                        neigh.fit(back_array)
                        back_dist, back_indices = neigh.kneighbors(back_array, return_distance=True)
                        back_10NND = np.vstack((back_dist[:,-1], np.zeros(back_dist.shape[0]))).T

                        # Carry out CST:
                        U1, p_value = mannwhitneyu(clus_10NND[:,0], back_10NND[:,0], alternative='less')
                        pval_array.append([clus_name, pers_val, p_value])

                        # Show whether cluster passed/failed CST:
                        five_sig = 2.867e-7
                        if p_value<five_sig: # 5-sigma significance
                            print(f'{clus_name}: statiscally significant - p = {p_value}')
                        else:
                            print(f'{clus_name}: not statiscally significant - p = {p_value}')

            else:
                continue   
                

    clus_sig_table = Table(data = np.array(pval_array), 
                           names=['Cluster Name', 'hdb_pers', 'p-value'], 
                           dtype = [str, 'float64', 'float64'])
    clus_sig_table.write('../data/cnn_classification/prelim_tables/hdb-pers_p-val_table.ecsv', 
                         format='ascii.ecsv', overwrite=True)
    
