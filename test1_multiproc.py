#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:58:39 2021

@author: Sonja
"""

from multiprocessing import Pool
import numpy as np
from moduleVGP import coord_VGP_globe, comp_B_long, count
from chaosmagpy.model_utils import synth_values
from tqdm import tqdm
import time
from typing import List
import sys 
import os

_PATH_CD = os.getcwd()
_PATH_NPZ = _PATH_CD + "/NPZ/"

# create full grid  all caps
radius = 6371.2  # km, core-mantle boundary
theta = np.linspace(1., 179., num=17)  # colatitude in degrees
phi = np.linspace(-180., 180., num=24)  # longitude in degrees

phi_grid, theta_grid = np.meshgrid(phi, theta)
radius_grid = radius*np.ones(phi_grid.shape) 
phi_1D = phi_grid.flatten()
theta_1D = theta_grid.flatten()
radius_1D = radius*np.ones_like(phi_1D)

# Wicht criteria
theta_c = 45.
Tn = 400.
Ts = 1000.

def split_file(fname, number_steps):
    
    npzfile = np.load(fname)
    temps = npzfile["time"]
    ghlm = npzfile["ghlm"]
    
    temps_ = np.array_split(temps, number_steps)
    ghlm_ = np.array_split(ghlm, number_steps)
    
    step_list = []
    for i in range(len(temps_)) :
        temp_step = len(temps_[i])
        step_list.append(temp_step)
    
    index_steps=[]
    temp = 0
    for step in step_list :
        temp = temp+step
        index_steps.append(temp)
    
    return index_steps, temps_ , ghlm_   

def fichiers(index:int, temps:List[float], ghlm:np.ndarray): # index

    coord = []
    for i,j in zip(range(len(temps)), tqdm(range(len(temps)),
                    initial=1, desc="Running", colour="blue")):
        gh = ghlm[i,:]
        B_rad, B_th, B_p = comp_B_long(radius_1D, theta_1D, phi_1D, gh)
        
        VGP_lat, VGP_lon = coord_VGP_globe(theta_1D, phi_1D,
                                                B_rad, B_th, B_p)#, lat_site,lon_site)

        coord.append((VGP_lat, VGP_lon))
        
        sys.stdout.flush()
               
    vgp_lat, vgp_lon = zip(*coord)

    if not os.path.exists(_PATH_NPZ):
        os.makedirs(_PATH_NPZ)

    file_name = _PATH_NPZ + "vgp_history" + f"_{index}"

    np.savez(file_name, theta_site = theta_1D, phi_site = phi_1D, 
             time = temps, vgp_lat_history = vgp_lat, vgp_lon_history = vgp_lon)
          
    return 

def fichiers_one_arg(args):
    return fichiers(*args)

def merge_files():
    import glob

    fname_list = glob.glob("vgp_history_*.npz")
    fname_number = []
    for name in fname_list:
        name = name[12:-4]
        fname_number.append(name)

    sorted_numbers = sorted(fname_number, key = int)
    flist = []     # sorted fnames
    for number in sorted_numbers:
        for filename in fname_list:
            if "vgp_history_" + number + ".npz" == filename:
                flist.append(filename) 

    for k,file in enumerate(flist):
        data = np.load(file)
        #print(np.shape(data['time']))
        if k == 0:
            my_time = data['time']
            my_vgp_lat = data['vgp_lat_history']
        else:
            my_time = np.concatenate( (my_time, data['time']), dtype=float)
            my_vgp_lat = np.concatenate( (my_vgp_lat, data['vgp_lat_history']), dtype=float)

    # print(np.shape(my_time))
   
    all_time = np.array( my_time, dtype = float )
    all_vgp_lat = np.array( my_vgp_lat, dtype=float )

    theta_site = data['theta_site']
    phi_site = data['phi_site']

    np.savez(_PATH_NPZ + 'all_vgp_history.npz', time = all_time, 
             theta_site = theta_site, phi_site = phi_site,
             vgp_lat_history = all_vgp_lat)

    return 

def Wicht(fname): #, lat_site, lon_site):

    npzfile = np.load(fname)
    theta_site = npzfile["theta_site"]
    phi_site = npzfile["phi_site"]
    vgp_hist_lat = npzfile["vgp_lat_history"]   
    #vgp_hist_lon = npzfile["vgp_lon_history"]
    temps = npzfile["time"]

    mean_exc = np.zeros_like(theta_site)
    mean_rev = np.zeros_like(theta_site)

    exc = np.zeros_like(theta_site)
    rev = np.zeros_like(theta_site)

    for i_site ,j in zip(range(len(theta_site)), tqdm(range(len(theta_site)), 
                    initial=1, desc="Running", colour="blue")):
        exc[i_site], exc_time, rev[i_site], rev_time = count(temps, vgp_hist_lat[:,i_site], theta_c, Tn, Ts)
        mean_exc[i_site] = np.mean(exc_time)
        mean_rev[i_site] = np.mean(rev_time)
        # print(np.shape(exc), np.shape(exc_time))
        # print(np.shape(rev), np.shape(rev_time))
    
    file_name = _PATH_NPZ + "Wicht_" + fname
    
    np.savez(file_name, theta_c = theta_c, Tn = Tn, Ts = Ts,
             theta_site = theta_site, phi_site = phi_site,
             excursions = exc, length_excursion = mean_exc,
             reversals = rev, length_reversal = mean_rev)
    
    return

def Wicht_one_arg(args):
    return Wicht(*args)

def execute():
    import glob

    fname = _PATH_NPZ + "all_vgp_history.npz"
    # nproc = int( os.getenv("OMP_NUM_THREADS") )

    """
    t1 = time.time()
    with Pool(nproc) as p:
        p.map(Wicht, fname)
    t2 = time.time()
    print(f'Time to process with multiprocessing : {t2-t1:4.3f}')
    """
    Wicht(fname)

def main():
    
    fname = "t_gauss_nskip1_E1e4_Ra1.5e4_Pm3.5_hdiff_short.npz"
    nproc = int( os.getenv("OMP_NUM_THREADS") )
    ind, temps, ghlm = split_file(fname, nproc)
    
    """
    t1 = time.time() # retourne temps sur ordi en s precision ns
    for it in range(len(temps)):
        fichiers(ind[it], temps[it], ghlm[it])
    t2 = time.time()
    print(f'Time to process sequentially : {t2-t1:4.3f}')
    """

    t1 = time.time()
    with Pool(nproc) as p:
        p.map(fichiers_one_arg, zip(ind, temps,ghlm))
    t2 = time.time()
    print(f'Time to process with multiprocessing : {t2-t1:4.3f}')

if __name__ == "__main__":
    main()
    merge_files()
    execute()
