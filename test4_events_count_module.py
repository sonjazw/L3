#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:43:44 2021

@author: szwahlgren
"""

import numpy as np
import matplotlib.pyplot as plt
import moduleMaps as maps
import moduleVGP as vgp
# from tqdm import tqdm
# from chaosmagpy import synth_values

# create full grid
radius = 6371.2  
theta = np.linspace(1., 179., num=179)  # colatitude in degrees
phi = np.linspace(-180., 180., num=361)  # longitude in degrees

# Conditions de Wicht
theta_c = 45.0 # degrees
Tn = 400.0  # years
Ts = 1000.0

phi_grid, theta_grid = np.meshgrid(phi, theta) # deja dans la fonction VGP_coord
radius_grid = radius*np.ones(phi_grid.shape) 

npzfile = np.load("lat_time_20000_25000.npz")
time = npzfile["time"]
ghlm = npzfile["ghlm"]
latitude = npzfile["lat"]
longitude = npzfile["lon"]

print( np.shape(time) )
print( np.shape(ghlm) )

coord = [(latitude, longitude)]

maps.VGP_marche_map(coord)

ind, T1, T2 = vgp.periods(time, latitude, theta_c, Tn, Ts)

exc, exc_time, rev, rev_time = vgp.count(time, latitude, theta_c, Tn, Ts)

mytime = np.zeros(1+len(T1) + len(T2), dtype = float)
mytime[0] = time[ind[0]]
value = time[0:ind[0]]

for j in range(len(T1)):
    mytime[2*j+1] = mytime[2*j] + T1[j] 
    mytime[2*j+2] = mytime[2*j+1] + T2[j]
mytime[-1] = time[-1]


plt.figure('lat')
plt.plot(mytime, 45*np.ones_like(mytime), "|-", c='g', markersize=50)
plt.plot(time[0], 45, "|", c='g', markersize=50)
plt.plot(value, 45*np.ones_like(value), "-", c='g', markersize=50)
plt.plot(time, latitude) 
plt.axhline(y=45, color='red', linestyle='--', zorder=-4)
plt.axhline(y=-45, color='red', linestyle='--')