#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:58:39 2021

@author: Sonja
"""

from multiprocessing import Pool
import numpy as np
import moduleVGP as vgp
from chaosmagpy.model_utils import synth_values
from tqdm import tqdm
import time
from typing import List
import sys 
import os

# create full grid  all caps
radius = 6371.2  # km, core-mantle boundary
theta = np.linspace(1., 179., num=37)  # colatitude in degrees
phi = np.linspace(-180., 180., num=72)  # longitude in degrees

phi_grid, theta_grid = np.meshgrid(phi, theta)
radius_grid = radius*np.ones(phi_grid.shape) 
temp_array = np.ndarray(theta_grid, phi_grid)
I = temp_array.flatten('F')

# Wicht criteria
theta_c = 45.
Tn = 400.
Ts = 1000.

# Site coordinates
lat_site = 0
lon_site = 0

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
    
    ilat = range(len(theta))
    ilon = range(len(phi))

    nlat = len(ilat)
    nlon = len(ilon)

    num_site = np.ones(phi_grid.shape)

    for i in ilat:
        for j in ilon:
            num_site[i,j] = i*nlon + j

    # potentiellement num_site = ilat*nlon + ilon

    coord = []
    for i,j in zip(range(len(temps)), tqdm(range(len(temps)),
                    initial=1, desc="Running", colour="blue")):
        gh = ghlm[i,:]
        B_rad, B_th, B_p = vgp.comp_B_long(radius_grid, theta_grid, phi_grid, gh)
        
        VGP_lat, VGP_lon = vgp.coord_VGP_globe(theta, phi,  phi_grid, 
                                                B_rad, B_th, B_p)#, lat_site,lon_site)
        coord.append((VGP_lat, VGP_lon))
        
        sys.stdout.flush()
               
    vgp_lat, vgp_lon = zip(*coord)
      
    vgp_lat_history = zip(num_site, temps, vgp_lat)
    vgp_lon_history = zip(num_site, temps, vgp_lon)

    file_name = "vgp_history" + f"_{index}"
    # file_name_lon = "vgp_lon_history" + f"_{index}"

    #np.savez(file_name, time = temps, vgp_lat = vgp_lat, vgp_lon = vgp_lon,
    #         vgp_history = vgp_history)
    # np.savez(file_name_lat, vgp_lat_history = vgp_lat_history)
    ## np.savez(file_name_lon, vgp_lon_history = vgp_lon_history)

    np.savez(file_name, vgp_lat_history = vgp_lat_history, vgp_lon_history = vgp_lon_history)
          
    return 

def fichiers_one_arg(args):
    return fichiers(*args)

def merge_files():

    import glob
    flist = glob.glob("vgp_history_*.npz")
    
    data_all = [np.load(fname) for fname in flist]
    merged_data = {}

    for data in data_all:
        [merged_data.update({k:v}) for k,v in data.items()]
    np.savez('all_vgp_history.npz', **merged_data)

    return 
    
def Wicht(fname, lat_site, lon_site):
    
    npzfile = np.load(fname)
    temps = npzfile["time"]
    latitude = npzfile["vgp_lat"]
    longitude = npzfile["vgp_lon"]
    
    
    a = 360 // len(phi)
    b = 180 // len(theta)
    
    phi_index = (lon_site + 180) // a     # hardcoded, see how to do if we change the 
    theta_index = (lat_site + 90) // b    #  sizes of theta and phi
    # period_index = [i for i in range(temps) if temps[i] == period]
    
    vgp_lat, vgp_lon = latitude[:, theta_index, phi_index], longitude[:, theta_index, phi_index]
    
    exc, exc_time, rev, rev_time = vgp.count(temps, vgp_lat, theta_c, Tn, Ts)
    
    mean_exc = np.mean(exc_time)
    mean_rev = np.mean(rev_time)
    
    file_name = "Wicht" + fname
    
    np.savez(file_name, theta_c = theta_c, Tn = Tn, Ts = Ts, 
             vgp_lat = vgp_lat, vgp_lon = vgp_lon, 
             excursions = exc, length_excursion = mean_exc,
             reversals = rev, length_reversal = mean_rev)
    
    return

def Wicht_one_arg(args):
    return Wicht(*args)

def execute():
    
    import glob
    #fname = ["vgp_history_119808.npz","vgp_history_239616.npz","vgp_history_359423.npz",
    #         "vgp_history_479230.npz","vgp_history_599037.npz","vgp_history_718844.npz"]
    fname = glob.glob("vgp_history_*.npz")
    nproc = int( os.getenv("OMP_NUM_THREADS") )
    
    latitude_site = lat_site*np.ones(len(fname))
    longitude_site = lon_site*np.ones(len(fname))

    t1 = time.time()
    with Pool(nproc) as p:
        p.map(Wicht_one_arg, zip(fname, latitude_site, longitude_site))
    t2 = time.time()
    print(f'Time to process with multiprocessing : {t2-t1:4.3f}')
    
def main():
    
    fname = "t_gauss_nskip1_E1e4_Ra1.5e4_Pm3.5_hdiff_short.npz"
    nproc = int( os.getenv("OMP_NUM_THREADS") )
    ind, temps, ghlm = split_file(fname, nproc)
   
    for num_site in I :

   
    # npzfile = np.load(fname)
    # temps = npzfile["time"]
    # ghlm = npzfile["ghlm"]
    
    # t1 = time.time() # retourne temps sur ordi en s precision ns
    
    # for it in range(len(temps)):
    #     fichiers(ind[it], temps[it], ghlm[it])
    
    # t2 = time.time()
    # print(f'Time to process sequentially : {t2-t1:4.3f}')
    
    
    t1 = time.time()
    with Pool(nproc) as p:
        p.map(fichiers_one_arg, zip(ind, temps,ghlm))
        # p.map(fichiers, temps, ghlm)
    t2 = time.time()
    print(f'Time to process with multiprocessing : {t2-t1:4.3f}')
    
if __name__ == "__main__":
    main()
    execute()
    
