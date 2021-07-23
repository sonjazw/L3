#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:13:03 2021

@author: szwahlgren
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from chaosmagpy.plot_utils import plot_maps

def VGP_map(lon, lat):
    """ Plots one VGP on a world map, Mollweide projection
    """
    fig = plt.figure('VGP')
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    
    ax.plot(lon, lat,'^',
              markersize=1, markerfacecolor='darkred', markeredgecolor='darkred', 
              alpha = 1, transform=ccrs.Geodetic())
    
    return fig, ax

def VGP_marche_map(tab):
    """ Plots the path of a VGP of one site on a world map, Mollweide projection. 
    """
    lat, lon = zip(*tab)
    fig = plt.figure('VGP path')
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    
    ax.plot(lon, lat,'.',
              markersize=1, markerfacecolor='darkred', markeredgecolor='darkred', 
              alpha = 1, transform=ccrs.Geodetic())
    
    return fig, ax

# npzfile = np.load('Wicht_all_vgp_history.npz')
# theta_site = npzfile['theta_site']
# phi_site = npzfile['phi_site']
# excursions = npzfile['excursions']
# excursion_time = npzfile['length_excursion']
# reversals = npzfile['reversals']
# reversal_time = npzfile['length_reversal']

def map_wicht_imshow(theta_site, phi_site, reversal_time): 
    """ Returns a world map with the lengths of reversal for every site
        using imshow.
        PlateCarree projection. 
    
    Parameters
    ----------
    theta_site : ndarray
        Array containing the colatitude in degrees.
    phi_site : ndarray
        Array containing the longitude in degrees.
    reversal_time : ndarray
        Array containing the data to plot.

    Returns
    -------
    fig : class:`matplotlib.figure.Figure`
        Matplotlib figure.
    ax : TYPE
        DESCRIPTION.
    """
    lat_site = 90 - theta_site  # y
    lon_site = phi_site         # x
    points_site = np.vstack((phi_site,theta_site)).T
    
    # theta_interp = np.linspace(theta_site[0], theta_site[-1], 179)
    # phi_interp = np.linspace(phi_site[0], phi_site[-1], 361)
    lat_interp = np.linspace(lat_site[0], lat_site[-1], 179)
    lon_interp = np.linspace(lon_site[0], lon_site[-1], 361)
    lon_grid_interp, lat_grid_interp = np.meshgrid(lon_interp, lat_interp)
    lat_interp = lat_grid_interp.flatten()
    lon_interp = lon_grid_interp.flatten()
    points_interp = np.vstack((lon_interp,lat_interp)).T
    
    grid_interp = griddata((lon_site, lat_site), reversal_time, 
                           (lon_grid_interp, lat_grid_interp), 
                           method = 'linear')
    
    fig = plt.figure('length en fonction du site')
    ax = plt.axes(projection = ccrs.PlateCarree())
    ax.gridlines()
    ax.coastlines(resolution='110m')
    ax.imshow(np.flipud(grid_interp), cmap = 'YlOrRd', aspect='auto', 
               extent=[lon_site[0], lon_site[-1], lat_site[0], lat_site[-1]])
    
    ax.scatter(lon_site, lat_site, c=reversal_time, cmap='YlOrRd', 
                edgecolor='black')
    
    # plt.plot(lon_site, lat_site, '.', markersize=1, 
    #         markerfacecolor='darkred', markeredgecolor='darkred', 
    #         alpha=1, transform=ccrs.Geodetic())     # plots the grid
    
    return fig, ax
    
def carte_wicht(theta_site, phi_site, reversal_time):
    """ Plots a world map of reversal length estimates in years for 
        every site using predefined plot_maps from the chaosmagpy
        package.

    Parameters
    ----------
    theta_site : ndarray
        Array containing the colatitude in degrees.
    phi_site : ndarray
        Array containing the longitude in degrees.
    reversal_time : ndarray
        2D array containing the data to plot.

    Returns
    -------
    fig : class:`matplotlib.figure.Figure`
        Matplotlib figure.
    ax : TYPE
        DESCRIPTION.
    """
    # Computing the number of different values of theta and phi
    # (the lengths of the original unflattened theta and phi)
    step_phi = 0
    for i in range(len(theta_site)) :
        if theta_site[i] != theta_site[i+1]:
            step_phi = i+1
            break

    step_theta = len(reversal_time) // step_phi
    
    reversal_time_2D = np.ones((step_theta, step_phi))
    
    # Reshaping the 1D arrays into 2D for the plot_maps function
    reversal_time_2D = np.reshape(reversal_time, (step_theta, step_phi))
    theta_2D = np.reshape(theta_site, (step_theta, step_phi))
    phi_2D = np.reshape(phi_site, (step_theta, step_phi))
    
    rev_min = min(reversal_time)
    rev_max = max(reversal_time)
    
    fig, ax = plot_maps(theta_2D, phi_2D, reversal_time_2D, 
                        titles = 'Longueurs en années d''évènements en fonction du site',
                        label = 'Length of reversal in years', cmap = 'YlOrRd',
                        vmin = rev_min, vmax = rev_max)
    
    return fig, ax
    