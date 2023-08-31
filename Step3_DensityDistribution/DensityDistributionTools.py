# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:40:27 2022

@author: NicoP
"""

import matplotlib.pyplot as plt
import numpy as np
import mplstereonet as stereo
from scipy.interpolate import griddata
import matplotlib.cm as cm
from matplotlib import path
from mpl_toolkits import mplot3d


def dif_angular(selected_sigi,Sigma_ix,Sigma_iy):
    
    ## esta función genera un vector "ang" que mide los angulos que hay entre las
    ## soluciones de Sigma_ix y Sigma_iy (vectores) y la solución específica 
    ## selected_sigi en formato(lon,lat)
    
    bidirectional=True
    lon1=selected_sigi[0]
    lat1=selected_sigi[1]
    ang=[]
    for i in range(0,len(Sigma_ix)): #### este código fue obetnido de stereonet angular_distance
        lat2=Sigma_iy[i] 
        lon2=Sigma_ix[i]    
        lon1, lat1, lon2, lat2 = np.atleast_1d(lon1, lat1, lon2, lat2)
        xyz1 = stereo.stereonet_math.sph2cart(lon1, lat1)
        xyz2 = stereo.stereonet_math.sph2cart(lon2, lat2)
        # This is just a dot product, but we need to work with multiple measurements
        # at once, so einsum is quicker than apply_along_axis.
        dot = np.einsum('ij,ij->j', xyz1, xyz2)
        angle = np.arccos(dot)
    
        # There are numerical sensitivity issues around 180 and 0 degrees...
        # Sometimes a result will have an absolute value slighly over 1.
        if np.any(np.isnan(angle)):
            rtol = 1e-4
            angle[np.isclose(dot, -1, rtol)] = np.pi
            angle[np.isclose(dot, 1, rtol)] = 0
    
        if bidirectional:
            mask = angle > np.pi / 2
            angle[mask] = np.pi - angle[mask]
        
        ang.append(np.rad2deg(angle)[0])

    return ang


def write_density_distribution(strike,dip,output_name):

    ## stike/dip tiene que ser los planos que representen polos de Sigma 1 (trend/plunge)
    a=stereo.density_grid(np.array(strike), np.array(dip))
    ## a[0]=lon a[1]=lat a[1]=valor de densidad
    
    lat=[]
    lon=[]
    trend=[]
    plunge=[]
    val=[]
    for i in range(0,np.shape(a)[1]):
        for j in range(0,np.shape(a)[2]):
            lon.append(a[0][i][j])
            lat.append(a[1][i][j])
            bas=stereo.geographic2plunge_bearing(lon[-1],lat[-1])
            trend.append(bas[1][0])
            plunge.append(bas[0][0])
            val.append(a[2][i][j])
    
    idN=-1
    fid=open(output_name,'w')
    for i in range(0,len(trend)):
        idN=idN+1
        fid.write(' '+str(idN)+' '+str(trend[i])+' '+str(plunge[i])+' '+str(val[i])+' \n')
    

def density_distribution(Sigma_ix,Sigma_iy,formato_sigmas,write,plot,ax_plot):
    
    ### para Kamb metodology use Stereo.density_grid; 
    ## esta función es para definir una grilla de densidades parecido a un histograma
    ## cantidad de datos por cada 2 grados
    
    ## Paso1: primero se generar los archivos x,y en lat lon de de datos a calcular distribución
    if formato_sigmas=='latlon':
        x=Sigma_ix
        y=Sigma_iy
    elif formato_sigmas=='trend_plunge':
        x=[]
        y=[]
        for i in range(0,len(Sigma_ix)):
            pole=stereo.plunge_bearing2pole(Sigma_ix,Sigma_iy) ## plano (strike,dip) cuyo polo 
            # es la linea definida por t,p
            S=stereo.pole(pole[0],pole[1]) ## linea definida por t,p escrita en lat lon.
            x.append(S[0])
            y.append(S[1])
    else:
        print('error en el nombre de formato revisar nombres posibles')
    
    
    
    ## Paso 2: Se define la grilla de la función de densidad 
    x_vec=np.linspace(-np.pi/2,np.pi/2,100)
    y_vec=np.linspace(-np.pi/2,np.pi/2,100)
    
    x_grid=[] ## grilla en coordendas lat lon vectorialmente
    y_grid=[]
    tt=[] ## vector para guardar coordendas trend/plunge 
    pp=[] 
    
    for xi in x_vec:
        for yi in y_vec:
            # if xi**2+yi**2<=np.pi:
            x_grid.append(xi) ## se guardan las coordenadas en lat lon
            y_grid.append(yi)
            bas=stereo.geographic2plunge_bearing(xi,yi)
            tt.append(bas[1]) ## se guardan las coordenasd en trend plunge también
            pp.append(bas[0]) 
            
            
    ## Paso 3: Definir valores de densidad para cada punto de grilla
    
    dif_aceptable=np.deg2rad(5) ## angulo entre pixel y sigma_ix aceptable para contarlo
    ## dentro del pixel.    
    val=[]
    points=np.ndarray(shape=(len(x_grid),2),dtype=float)
    cont=0 
    
    for i in range(0,len(x_grid)):
        difx=abs(x_grid[i]-np.array(x))
        dify=abs(y_grid[i]-np.array(y))
        val.append(np.sum((difx<dif_aceptable) & (dify<dif_aceptable)))
        points[cont,0]=x_grid[i]
        points[cont,1]=y_grid[i]
        cont=cont+1
    

    ## Paso 4: plotear y/o escribir output
        
    if plot:
        
        niveles=[0.9,1,2,5,10,15,20,25,30,50]
        delta=np.deg2rad(2)
        mesh_x, mesh_y = np.mgrid[min(x_grid):max(x_grid):delta,min(y_grid):max(y_grid):delta]
               
        grid_densi = griddata(points, val, (mesh_x, mesh_y), method='linear')
        # ax_plot=figd.add_subplot(1,2,2,projection='stereonet')
        ax_plot.contourf(mesh_x,mesh_y,grid_densi,cmap=cm.RdYlBu_r,levels=niveles)
        
        
    if write:
        idN=-1
        fid=open('density_distribution.txt','w')
        for i in range(0,len(x_grid)):
            idN=idN+1
            fid.write(' '+str(idN)+' '+str(tt[i])+' '+str(pp[i])+' '+str(val[i])+' \n')
        
  