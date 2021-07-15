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
    
    # base_name = "30Ma_lat_temps"

    coord = []
    for i,j in zip(range(len(temps)), tqdm(range(len(temps)),
                    initial=1, desc="Running", colour="blue")):
        gh = ghlm[i,:]
        B_rad, B_th, B_p = vgp.comp_B_long(radius_grid, theta_grid, phi_grid, gh)
        
        VGP_lat, VGP_lon = vgp.coord_VGP_globe(theta, phi,  phi_grid, 
                                                B_rad, B_th, B_p)#, lat_site,lon_site)
        coord.append((VGP_lat, VGP_lon))
        
        sys.stdout.flush()
        
    vgp_history = zip(temps, coord)
        
    vgp_lat, vgp_lon = zip(*coord)
    
    # file_name = base_name + f""_{index}
    
    file_name = "vgp_history" + f"_{index}"

    #np.savez(file_name, time = temps, vgp_lat = vgp_lat, vgp_lon = vgp_lon,
    #         vgp_history = vgp_history)
    np.savez(file_name,  vgp_history = vgp_history)
          
    return vgp_history 

def fichiers_one_arg(args):
    return fichiers(*args)

def fichiers_coord_VGP(index:int, temps:List[float], ghlm:np.ndarray):
    
    ghlm = ghlm[:,None,None,:]

    B_rad, B_th, B_p = vgp.comp_B(radius, theta, phi, ghlm)
    # B_rad, B_th, B_p = synth_values(ghlm, radius, theta, phi, 
    #                             nmax=13, mmax=13, grid=True)
    
    coord = []
    
    VGP_lat, VGP_lon = vgp.coord_VGP_globe(theta, phi, phi_grid, 
                                           B_rad, B_th, B_p)#, lat_site, lon_site)
    
    coord.append(VGP_lat, VGP_lon)
    vgp_history = zip(temps, coord)
    
    vgp_lat,vgp_lon = zip(*coord)
    file_name = "vgp_history" + f"_{index}" + "_test"
    #np.savez(file_name, time = temps, vgp_lat = vgp_lat, vgp_lon = vgp_lon,
    #         vgp_history = vgp_history)
    np.savez(file_name,  vgp_history = vgp_history)
    
def Wicht(fname, lat_site, lon_site):
    
    npzfile = np.load(fname)
    temps = npzfile["time"]
    latitude = npzfile["vgp_lat"]
    longitude = npzfile["vgp_lon"]
    
    a = 180 // len(phi)
    b = 90 // len(theta)
    
    phi_site = int(lon_site + (180//a)-1)     # hardcoded, see how to do if we change the 
    theta_site = int(lat_site + (90//b)-1)    #  sizes of theta and phi
    # period_index = [i for i in range(temps) if temps[i] == period]
    
    vgp_lat, vgp_lon = latitude[:, theta_site, phi_site], longitude[:, theta_site, phi_site]
    
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

# for it in range(len(temps)):
#     fichiers(temps[it], ghlm[it], index[it])

"""
save a npz . with different Wicht criteria
    theta_c | Tn | Ts | coord site |  number exc | mean time exc | num rev | mean time rev
"""

    
