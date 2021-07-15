#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:44:04 2021

@author: szwahlgren
"""

import numpy as np
from chaosmagpy.model_utils import synth_values

def comp_B_long(radius_grid, theta_grid, phi_grid, coeffs_gauss):
    """ retourne les composantes de B
    """
    
    B_radius, B_theta, B_phi = synth_values(coeffs_gauss, radius_grid, theta_grid, phi_grid, nmax=13, mmax=13)
    
    # print(np.min(B_phi))
    # print(np.max(B_phi))
    
    return B_radius, B_theta, B_phi

def comp_B(radius, theta, phi, coeffs_gauss):
    """ retourne les composantes de B
    """
    
    B_radius, B_theta, B_phi = synth_values(coeffs_gauss, radius, theta, 
                                            phi, nmax=13, mmax=13, grid=True)
    
    # print(np.min(B_phi))
    # print(np.max(B_phi))
    
    return B_radius, B_theta, B_phi


def coord_VGP_globe(theta, phi, phi_grid, B_radius, B_theta, B_phi):#, lat_site, lon_site) :
    """ retourne les coordonnees du VGP pour un site donnée (calcul pour  tous
        le globe)
    """
    
    # Br btheta bphi are 3D arrays, fiirst dimension time, second theta, third phi
    # we want to broadcast all the arrays along this first dimension
      
    Z = -B_radius   
    Y =  B_phi
    X = -B_theta
    
    #Calcul de D et I 
    H = np.sqrt(X**2 + Y**2)
    D = np.arctan2(Y,X)
    I = np.arctan2(Z,H)
    p = np.arctan2(2, np.tan(I))
    
    cosp = np.cos(p) # array (179, 361)
    sinp = np.sin(p)
    cosD = np.cos(D)
    sinD = np.sin(D)
            
    ls = np.radians(90 - theta) # ls is an array shape (179,)
    phi_s = np.radians(phi) # np:shape(phi) = (361,)  
    
    phi_s_grid, ls_grid = np.meshgrid(phi_s, ls) # lambda_grid shape (179,361)
    phi_p = np.ones(phi_s_grid.shape)

    lp_rad = np.arcsin(np.sin(ls_grid)*cosp + np.cos(ls_grid)*sinp*cosD )
    beta = np.arcsin(sinp*sinD / np.cos(lp_rad))
    
    # for i in range (0, len(theta)):
    #     for j in range (0,len(phi)):
    #         if cosp[i,j] >= np.sin(ls_grid[i,j])*np.sin(lp_rad[i,j]):
    #             phi_p[i,j] = np.degrees(phi_s_grid[i,j]) + np.degrees(beta[i,j])
    #         else :
    #             phi_p[i,j] = np.degrees(phi_s_grid[i,j]) + 180 - np.degrees(beta[i,j])
                
    product = np.sin(ls_grid)*np.sin(lp_rad)
    condition = cosp >= product 
    
    phi_s_grid_degrees = np.degrees(phi_s_grid)
    beta_degrees = np.degrees(beta)
    
    phi_p[condition] = phi_s_grid_degrees[condition] + beta_degrees[condition]
    phi_p[~condition] = phi_s_grid_degrees[~condition] + 180 - beta_degrees[~condition]  
                
    lp = np.degrees(lp_rad)
    
    # a = 180 // len(phi)
    # b = 90 // len(theta)
    
    # phi_site = lon_site + (180//a)     # hardcoded, see how to do if we change the 
    # theta_site = lat_site + (90//b)    #  sizes of theta and phi
    
    # return VGP_lon, VGP_lat
    # return lp[theta_site, phi_site], phi_p[theta_site, phi_site]
    return lp, phi_p

# faire une fonction qui pourra déterminer combien d'inversion en combien de temps
# il y a eu a un site donné. Il prendra en entrée histoire des coordonnées de VGP
# avec le temps correspondant ainsi que les critère de Wicht

# il faudrait faire une fonction qui ne determine le vgp qu'en une seule 
# coordonées pour les temps de calculs.

def coord_VGP(theta, phi, phi_grid, B_radius, B_theta, B_phi): #, lat_site, lon_site) :
    """ Calcul les coordonnées d'un VGP à un unique site. 
        lat_site between [-89, 89] and lon_site between [-180,180]
    """
      
    Z = -B_radius   
    Y =  B_phi
    X = -B_theta
    
    #Calcul de D et I 
    H = np.sqrt(X**2 + Y**2)
    D = np.arctan2(Y,X)
    I = np.arctan2(Z,H)
    p = np.arctan2(2, np.tan(I))
    
    cosp = np.cos(p) # array (179, 361)
    sinp = np.sin(p)
    cosD = np.cos(D)
    sinD = np.sin(D)
            
    ls = np.radians(90 - theta) # ls is an array shape (179,)
    phi_s = np.radians(phi) # np:shape(phi) = (361,)  
    
    phi_s_grid, ls_grid = np.meshgrid(phi_s, ls) # lambda_grid shape (179,361)
    phi_p = np.ones(phi_s_grid.shape)

    lp_rad = np.arcsin(np.sin(ls_grid)*cosp + np.cos(ls_grid)*sinp*cosD )
    beta = np.arcsin(sinp*sinD / np.cos(lp_rad))
    
    # for i in range (0, len(theta)):
    #     for j in range (0,len(phi)):
    #         if cosp[i,j] >= np.sin(ls_grid[i,j])*np.sin(lp_rad[i,j]):
    #             phi_p[i,j] = np.degrees(phi_s_grid[i,j]) + np.degrees(beta[i,j])
    #         else :
    #             phi_p[i,j] = np.degrees(phi_s_grid[i,j]) + 180 - np.degrees(beta[i,j])
                
    product = np.sin(ls_grid)*np.sin(lp_rad)
    condition = cosp >= product 
    
    phi_s_grid_degrees = np.degrees(phi_s_grid)
    beta_degrees = np.degrees(beta)
    
    phi_p[condition] = phi_s_grid_degrees[condition] + beta_degrees[condition]
    phi_p[~condition] = phi_s_grid_degrees[~condition] + 180 - beta_degrees[~condition]  
                
    lp = np.degrees(lp_rad)
    
    a = 180 // len(phi)
    b = 90 // len(theta)
    
    phi_site = lon_site + (180//a)     # hardcoded, see how to do if we change the 
    theta_site = lat_site + (90//b)    #  sizes of theta and phi
    
    # return VGP_lon, VGP_lat
    return lp[theta_site, phi_site], phi_p[theta_site, phi_site]

def periods(time, latitude, theta_c, Tn , Ts):
    """ Returns the tables T1 qnd T2 respectivly being the array  where theta
        is between the criterions and when greater.
    """
    
    t = [] # list of all passages (te, ts, te, ts ...)
    ind = [] # list of all the index of passages (ie, is, ie, is ...)
    crossing = [] # if all is right, this shoud be an array of +/- theta_c
        
    entree = False
    for i, vals in enumerate(zip(time, latitude)) :
        temps = vals[0]
        lat = vals[1]
        if 90 - np.abs(lat) > theta_c and entree is False:
            entree = True
            t.append(temps)
            ind.append(i)   # index of crossing the criterion
            crossing.append(lat)
        if 90 - np.abs(lat) <= theta_c and entree is True:
            entree = False
            t.append(temps)
            ind.append(i)  # index of recrossing criterion
            crossing.append(lat)
            
    time_theta_beneath = [t[i+1]-t[i] for i in range (0, len(t)-1) if i%2==0]
    time_theta_above = [t[i+1]-t[i] for i in range (0, len(t)-1) if i%2!=0]
    
    # print(f'T1    {len(time_theta_beneath)}    {len(time_theta_above)}')
            
    if len(time_theta_above) == 0 or len(time_theta_beneath) == 0 :
        print('Aucun croisement du critere de latitude')
        T1 = None
        T2 = None
    elif len(time_theta_above) == len(time_theta_beneath) :
        T1 = np.array(time_theta_beneath)
        T2 = np.array(time_theta_above)
    elif len(time_theta_above) < len(time_theta_beneath) :
        T1 = np.delete(time_theta_beneath, -1)
        T2 = np.array(time_theta_above)
    elif len(time_theta_above) > len(time_theta_beneath) :
        T1 = np.array(time_theta_beneath)
        T2 = np.delete(time_theta_above, -1) 
    
    return ind, T1, T2

def count(time, latitude, theta_c, Tn , Ts):
    """ Returns interger of how many reversals there has been at a given 
        site (theta, phi) during a given period (time). This function calculates
        for chosen Wicht condition (latitude criterion and Tn) and VGP coords
        (tab)
    """
    
    t = [] # list of all passages (te, ts, te, ts ...)
    ind = [] # list of all the index of passages (ie, is, ie, is ...)
    crossing = [] # if all is right, this shoud be an array of +/- theta_c
        
    entree = False
    for i, vals in enumerate(zip(time, latitude)) :
        temps = vals[0]
        lat = vals[1]
        if 90 - np.abs(lat) > theta_c and entree is False:
            entree = True
            t.append(temps)
            ind.append(i)   # index of crossing the criterion
            crossing.append(lat)
        if 90 - np.abs(lat) <= theta_c and entree is True:
            entree = False
            t.append(temps)
            ind.append(i)  # index of recrossing criterion
            crossing.append(lat)
            
    time_theta_beneath = [t[i+1]-t[i] for i in range (0, len(t)-1) if i%2==0]
    time_theta_above = [t[i+1]-t[i] for i in range (0, len(t)-1) if i%2!=0] 
            
    
    if len(time_theta_above) == 0 or len(time_theta_beneath) == 0 :
        print('Aucun croisement du critere de latitude')
        T1 = None
        T2 = None
    elif len(time_theta_above) == len(time_theta_beneath) :
        T1 = np.array(time_theta_beneath)
        T2 = np.array(time_theta_above)
    elif len(time_theta_above) < len(time_theta_beneath) :
        T1 = np.delete(time_theta_beneath, -1)
        T2 = np.array(time_theta_above)
    elif len(time_theta_above) > len(time_theta_beneath) :
        T1 = np.array(time_theta_beneath)
        T2 = np.delete(time_theta_above, -1)     
    
    index = []
    temporary = []
    length = []
    
    if len(time_theta_above) == 0 or len(time_theta_beneath) == 0 :
        print('Aucun croisement du critere de latitude')   
    else : 
        i = 0
        while i <= len(T1) -1 :
            if T1[i] > Tn :
                if T2[i] > Ts :
                    length.append(T1[i])
                    index.append(i)
                    i = i+1
                elif T2[i] < Ts :
                    j = i
                    T1_sum = 0
                    T2_sum = 0
                    while (T2[j] < Ts) and (j < len(T2) - 1) : 
                        T1_sum = T1_sum + T1[j]
                        T2_sum = T2_sum + T2[j]
                        temporary.append(j)
                        j += 1
                    length.append(T1_sum + T1[j] + T2_sum) 
                    index.append(i)   
                    i = j+1
            else :
                i = i+1
            
        print('Count : ', len(length))
        # print('Length of events : ', length)
        
    retrieve = [2*i for i in index]
                
    excursion = 0
    reversal = 0
    
    excursion_time = []
    reversal_time = []
    
    for i, vals in enumerate(retrieve) :
        if (crossing[vals]*crossing[vals+1]) > 0 :
            excursion += 1
            excursion_time.append(length[i])
        else :
            reversal += 1
            reversal_time.append(length[i])
        
    print('Excursions : ', excursion, 'Length of excursions : ', excursion_time )          
    print('Reversals : ', reversal, 'Length of reversal : ', reversal_time)
    
    return excursion, excursion_time, reversal, reversal_time
        
"""
Les conditions de Wicht sont : time and latitude criterion
The latitude criterion speaks for itself : when the theta angle breaches a 
certain value, it means an excursion or reversal is about to take place.
Of this condition is valid longer than a certain time Tn (often 1000 yr), then
we can count it as an excursion of reversal. The difference between the two is 
if the final polarity is reversed or not.

To determine two different events : two events are separated byy a stable 
period with a duration Ts. Often this duration is the same as the core dipole
decay time Td = 29 kyr. 

As such, a stable period is defined as the time span where virtual magnetic
poles stay closer to the geographical pole than the latitude criterion  for a
total time of at least Ts. These periods should not be interrupted by periods
with theta >= theta_c longer than Tn (see the time criterion on neglecting events)

We now want to determine two things : 
    - for a certain site with a given duration, how many reversals can we count?
    - for a certain site, how long is the reversal?
    
We will use for first tests : 
    - theta_c = 45 , latitude criterion
    - Ts = Td = 29 kyr , stability period
    - Tn = 1000 yr , if events are events or not
    
"""
    

    

