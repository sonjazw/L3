#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:44:04 2021

@author: szwahlgren
"""

import numpy as np
from chaosmagpy.model_utils import synth_values

def comp_B_long(radius_grid, theta_grid, phi_grid, coeffs_gauss):
    """ Returns the components of the magnetic field. If chaosmagpy package is 
        loaded in the main, one could also directly use synth_values from 
        chaosmagpy.model_utils.

    Parameters
    ----------
    radius_grid : ndarray 
        Array containing the radius in kilometers.
    theta_grid : ndarray
        Array containing the latitude in degrees.
    phi_grid : ndarray
        Array containing the longitude in degrees.
    coeffs_gauss : ndarray
        Array containing the Gauss coefficients up to n = 13.

    Returns
    -------
    B_radius, B_theta, B_phi : ndarray
        Radial, colatitude and azimuthal field components.
    """
    
    B_radius, B_theta, B_phi = synth_values(coeffs_gauss, radius_grid, theta_grid, phi_grid, 
                                            nmax=13, mmax=13)
    return B_radius, B_theta, B_phi

def comp_B(radius, theta, phi, coeffs_gauss):
    """ Tentative de rendre les calculs plus rapides, pas terminé
    """
    B_radius, B_theta, B_phi = synth_values(coeffs_gauss, radius, theta, 
                                            phi, nmax=13, mmax=13, grid=True)
    return B_radius, B_theta, B_phi


def coord_VGP_globe(theta, phi, B_radius, B_theta, B_phi):#, lat_site, lon_site) :
    """ Returns the VGP coordinates of all the given sites.

    Parameters
    ----------
    theta : ndarray
        Array containing the colatitude in degrees.
    phi : ndarray
        Array containing the longitude in degrees.
    B_radius, B_theta, B_phi : ndarray
        Field components of the magnetic field.

    Returns
    -------
    lp : ndarray
        Array containing all the VGP latitudes for all the sites
    phi_p : ndarray
        Array containing all the VGP longitudes for all the sites

    """
    Z = -B_radius   
    Y =  B_phi
    X = -B_theta
    
    # Computing p and D 
    H = np.sqrt(X**2 + Y**2)
    D = np.arctan2(Y,X)
    I = np.arctan2(Z,H)
    p = np.arctan2(2, np.tan(I))
    
    cosp = np.cos(p) 
    sinp = np.sin(p)
    cosD = np.cos(D)
    sinD = np.sin(D)

    # Converting site degrees into radians     
    ls = np.radians(90 - theta) 
    phi_s = np.radians(phi) 
    
    phi_p = np.ones(phi_s.shape)

    # Computing with Butler's method
    lp_rad = np.arcsin(np.sin(ls)*cosp + np.cos(ls)*sinp*cosD )
    beta = np.arcsin(sinp*sinD / np.cos(lp_rad))
                
    product = np.sin(ls)*np.sin(lp_rad)
    condition = (cosp >= product) 
    
    phi_s_degrees = np.degrees(phi_s)
    beta_degrees = np.degrees(beta)
    
    phi_p[condition] = phi_s_degrees[condition] + beta_degrees[condition]
    phi_p[~condition] = phi_s_degrees[~condition] + 180 - beta_degrees[~condition]  
                
    lp = np.degrees(lp_rad)

    return lp, phi_p

def periods(time, latitude, theta_c, Tn , Ts):
    """ Returns the arrays T1 and T2 respectively being the array where theta
        is between the criterions and when greater. Used to draw latitude vs.
        time graph.

    Parameters
    ----------
    time : ndarray
        Array containing the time in years.
    latitude : ndarray
        Array containing the VGP latitude in degrees.
    theta_c : float
        Latitude criterion
    Tn : float
        Time in years determining the minimum time a crossing can be considered 
        an event
    Ts : float
        Time in years determining the end or not of an event (stable period)

    Returns
    -------
    ind : ndarray
        Array containing the index of crossings in time and latitude.
    T1 : ndarray
        Array containing the latitudes is between -theta_c and theta_c.
    T2 : ndarray
        Array containing the latitudes is greater than |theta_c| .
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
    """ Returns an interger of how many reversals and excursions there has been
        depending on the Wichts conditions and VGP latitudes.

    Parameters
    ----------
    time : ndarray
        Array containing the time in years.
    latitude : ndarray
        Array containing the VGP latitude in degrees.
    theta_c : float
        Latitude criterion
    Tn : float
        Time in years determining the minimum time a crossing can be considered 
        an event
    Ts : float
        Time in years determining the end or not of an event (stable period)

    Returns
    -------
    excursion : int
        Number of excursions for every site.
    excursion_time : ndarray
        Array containing the lengths of the excursions for every site.
    reversal : int
        Number of reversals for every site.
    reversal_time : ndarray
        Array containing the lengths of the reversals for every site.
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
        
    # print('Excursions : ', excursion) , 'Length of excursions : ', excursion_time )          
    # print('Reversals : ', reversal) ,'Length of reversal : ', reversal_time)
    
    return excursion, excursion_time, reversal, reversal_time

    

    

