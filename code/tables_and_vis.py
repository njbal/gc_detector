from astropy.io import ascii
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})
from astropy.table import Table
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from clusterer import compute_angular_radius
import astropy.units as u
from classifier import delete_files_in_directory

label_size = 20
plt.rcParams['axes.labelsize'] = label_size

def ax_config(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(labelsize=14, direction='in', which='major', length = 5)
    ax.tick_params(which='minor', direction='in', length = 2)
    ax.tick_params(top=True, right=False, which='both', direction='in')


def four_panel(file, save=True, display=False):
    T = ascii.read(file, format='ecsv')
    desired_mem_prob = 0.9
    ind = np.array(T['gmm_mem_prob'])>=desired_mem_prob
    ra, dec, plx, pmra, pmdec, l, b = T['ra'], T['dec'], T['parallax'], T['pmra'], T['pmdec'], T['l'], T['b']
    e_plx, e_pmra, e_pmdec = T['e_parallax'], T['e_pmra'], T['e_pmdec']
    point_size=20
    fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(12,12))
    clus_name = file.split('/')[-1].split('_')[-1][:-5]
#     print(clus_name)
    if 'cand' in clus_name:
        plot_title = f'Candidate {clus_name[4:]}'
    else:
        plot_title = clus_name
    plt.suptitle(f'{plot_title}', size=30)
    # Position:
    ax1.scatter(l[~ind], b[~ind], s=point_size, fc = 'none', ec = 'red', label = fr'$p_{{\rm mem}} < {desired_mem_prob}$')
    ax1.scatter(l[ind], b[ind], s=point_size, fc = 'none', ec = 'black', label = fr'$p_{{\rm mem}} \geq {desired_mem_prob}$')
    ax1.set_xlabel(r'$l \ \rm [deg]$')
    ax1.set_ylabel(r'$b \ \rm [deg]$')
    ax1.set_xlim(l[ind].mean()-0.5, l[ind].mean()+0.5)
    ax1.set_ylim(b[ind].mean()-0.5, b[ind].mean()+0.5)
    
    if clus_name=='Ursa Minor':
        ax1.set_xlim(l[ind].mean()-1, l[ind].mean()+1)
        ax1.set_ylim(b[ind].mean()-1, b[ind].mean()+1)
        
    ax1.legend()
    # Parallax or dist:
    if int(len(plx)/15)<=0:
        nbins = 5
    else:
        nbins = int(len(plx)/15)

    ax2.hist(plx[~ind], bins = nbins, histtype='step', color='red', label = fr'$p_{{\rm mem}} < {desired_mem_prob}$')
    ax2.hist(plx[ind], bins = nbins, histtype='step', color='black', label = fr'$p_{{\rm mem}} \geq {desired_mem_prob}$')
    ax2.set_ylabel(r'$\rm Counts \ per \ bin$')
    ax2.set_xlabel(r'$\varpi \ \rm [mas]$')
    ax2.set_xlim(plx[ind].min()-0.5, plx[ind].max()+0.5)
    ax2.legend()
    # Proper motion:
    ax3.scatter(pmra[~ind], pmdec[~ind], s=point_size, fc = 'none', ec = 'red', label = fr'$p_{{\rm mem}} < {desired_mem_prob}$')
    ax3.scatter(pmra[ind], pmdec[ind], s=point_size, fc = 'none', ec = 'black', label = fr'$p_{{\rm mem}} \geq {desired_mem_prob}$')
    ax3.set_xlabel(r'$\mu_{\alpha^*} \ \rm [mas/yr]$')
    ax3.set_ylabel(r'$\mu_{\delta} \ \rm [mas/yr]$')
    ax3.set_xlim(pmra[ind].mean()-5, pmra[ind].mean()+5)
    ax3.set_ylim(pmdec[ind].mean()-5, pmdec[ind].mean()+5)
    ax3.legend()
    # CMD:
    ax4.scatter(T['bp_rp'][~ind],T['g'][~ind], s=point_size, fc = 'none', ec = 'red', label = fr'$p_{{\rm mem}} < {desired_mem_prob}$')
    ax4.scatter(T['bp_rp'][ind],T['g'][ind], s=point_size, fc = 'none', ec = 'black', label = fr'$p_{{\rm mem}} \geq {desired_mem_prob}$')
    ax4.set_xlabel(r'$\rm BP-RP \ [mag]$')
    ax4.set_ylabel(r'$G \rm \ [mag] $')
    ax4.invert_yaxis()
    ax4.legend()
    ax_config(ax1)
    ax_config(ax2)
    ax_config(ax3)
    ax_config(ax4)
    fig.tight_layout()
    
    if display==True:
        plt.show()
    if save==False:
        plt.close()
    else:
        plt.savefig(f'../results/visualisations/four_panel/{clus_name}.png')
        plt.close()
    

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mk_probs_table(path_to_prelim_tables, prelim_mem_lists_path, results_dir):
    """Writes a probabilities table consisting of:
    - CNN Classification probability
    - HDBSCAN persistence
    - HDBSCAN min_cluster_size value used to detect object
    - p-value derived  from the CST""" 
    
    pval_pers_file_path = f'{path_to_prelim_tables}hdb-pers_p-val_table.ecsv'

    # retrieve p_class:
    p_class_file_path = f'{path_to_prelim_tables}p_class-table.ecsv'
    P = ascii.read(p_class_file_path) # class prob table

    # match cluster name with p_class
    p_class_arr = []
    for x in os.listdir(prelim_mem_lists_path):
        for name, val in P:
            if name[:-4] in x:
                clus_name = x.split('_')[-1][:-5]
                spec_mcls = x.split('_')[-3]
                p_class_arr.append([clus_name, val, spec_mcls])
            else:
                continue


    p_class_arr = np.array(p_class_arr)  
    
    # match clus name from p_class array to pval-pers table:
    T = ascii.read(pval_pers_file_path) # hdbscan pers-pval table
    t_arr = np.array([T['Cluster Name'], T['hdb_pers'], T['p-value']]).T
    indices = np.intersect1d(t_arr[:,0], p_class_arr[:,0], return_indices = True)
    t_arr, p_class_arr = t_arr[indices[1]], p_class_arr[indices[2]]


    probs_table = Table(np.hstack((p_class_arr, t_arr[:,1:])), names = ['Cluster Name', 
                                                                            r'$p_{\rm class}$', 
                                                                            r'${min\_cluster\_size}$', 
                                                                            r'$p_{\rm hdb}$', 
                                                                            r'$p_{\rm CST}$'], 
                                                                   dtype = ['str', 'float64', 'int8', 
                                                                            'float64', 'float64'])

    
    tables_dir = f'{results_dir}tables/'
    ensure_dir(tables_dir)    
    probs_table.write(f'{tables_dir}probs_table.ecsv', format = 'ascii.ecsv', overwrite=True)
    return probs_table

def compute_astrometric_error(vals, errors, sys_error):
    mean_val = np.sum(vals / errors ** 2) / np.sum(1 / errors ** 2)
    u_stat = np.sqrt(1 / np.sum(1 / errors ** 2))
    u_val = u_stat + sys_error
#     u_val = np.sqrt(u_stat**2+sys_error**2)
    return np.array([mean_val, u_val])

def cluster_stat(clus_file_path):
    """Returns mean astrometric values for a given cluster's membership list file"""
    data = np.genfromtxt(clus_file_path)[1:]
    ind = data[:,-1]>=0.9
    data = data[ind]

    stat = []

    name = clus_file_path.split('/')[-1].split('_')[-1][:-5]

    # number of members:
    n_members = data.shape[0]

    # position stats - ra, dec, l, b:
    pos_mean = data[:,:4].mean(axis=0)

    # angular radius:
    rad = np.round(compute_angular_radius(data[:,2:4]), 3)

    # plx - pm stats:
    plx_pm = []
    sys_errors = [0.01, 0.02, 0.02] # systematic errors in plx, pmra and pmdec

    for i in range(3):
        x = compute_astrometric_error(data[:,4+i], data[:,7+i], sys_errors[i])
        plx_pm.append(x)
    cluster_stat = np.array([name, data.shape[0], 
                             rad, *np.round(pos_mean, 6), 
                             *np.round(np.array(plx_pm).flatten(),3)])
    
    return cluster_stat

def mk_ast_table(prelim_clus_mem_lists_path, results_dir):
    """Writes a table consisting of the mean astrometric values of each cluster/candidate"""
    ast_stats = []
    for f in os.listdir(prelim_clus_mem_lists_path):
        file = f'{prelim_clus_mem_lists_path}{f}'
        s = cluster_stat(file)
        ast_stats.append(s)
    ast_stats = np.array(ast_stats)
    
    sorted_ind = np.argsort(ast_stats[:,0])
    ast_table = Table(ast_stats[sorted_ind], 
                      names = ['Cluster Name',r'$N_{\rm mem}$', r'$R_{50}$', r'$\alpha$',
                                r'$\delta$', r'$l$', r'$b$',
                                r'$\varpi$', r'$u(\varpi)$', r'$\mu_{\alpha^*}$',
                                r'$u(\mu_{\alpha^*})$', r'$\mu_{\delta}$', r'$u(\mu_{\delta})$'],                   
                      dtype = ['str', 'int64', 'float64', 'float64',
                               'float64', 'float64', 'float64', 'float64',
                               'float64', 'float64', 'float64', 'float64',
                               'float64'], 
                      units = [None, None, u.deg, u.deg,
                               u.deg, u.deg, u.deg, u.mas,
                               u.mas, u.mas/u.yr, u.mas/u.yr, u.mas/u.yr, 
                               u.mas/u.yr])
    
    tables_dir = f'{results_dir}tables/'
    ast_table.write(f'{tables_dir}ast_table.ecsv', format = 'ascii.ecsv', overwrite=True)
    return ast_table

def rename_clus_mem_lists(prelim_clus_mem_lists_path, results_dir):
    """Removes cluster IDs from filenames of cluster membership lists, and copies files to results directory"""
    
    final_clus_mem_lists_path = f'{results_dir}clus_mem_lists/'
    ensure_dir(final_clus_mem_lists_path)
    delete_files_in_directory(final_clus_mem_lists_path)

    for f in os.listdir(prelim_clus_mem_lists_path):
        clus_name = f.split('_')[-1][:-5]
        command = f'cp {prelim_clus_mem_lists_path}"{f}" {final_clus_mem_lists_path}'
        os.system(command)
        os.rename(f'{final_clus_mem_lists_path}{f}', f'{final_clus_mem_lists_path}{clus_name}.ecsv')