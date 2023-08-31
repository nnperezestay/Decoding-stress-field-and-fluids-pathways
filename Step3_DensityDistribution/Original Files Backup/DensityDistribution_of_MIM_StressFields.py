# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:46:12 2020

@author: NicoP
"""

import matplotlib.pyplot as plt
import numpy as np
import mplstereonet as stereo
import matplotlib.cm as cm
from matplotlib import path
import DensityDistributionTools as DDT
import time

tic = time.time()

##############################################################################
########################## Definition of settings ############################
##############################################################################


## Option of 1 file
name='Test_PuyuhuapiLongTerm' ## name of the file
names=[name]

## Optios of several files
# names=['Test_PuyuhuapiLongTerm','Test_PuyuhuapiLongTerm']
## This option is available for analyzing several files at ones.
 
percentile_target=75


## plot settings
plot_principal_stress_distribution_figure=True
plot_fhi_extracted_histogram=True
plot_stress_figure=True

## saving plots settings
save_stress_distribution_fig=True
save_stress_fig=False

## export settings
write_sigmas_prob=True

n_fig_principal=1
## Fixed parameters, do not change
n_fig=11 ## esto es para plotear todo junto (subplot) NO CAMBIAR
n_cases=len(names)
case=0
    

##############################################################################
################################## Main Code #################################
##############################################################################


for name in names: ## for each file
    
    ##########################################################################
    ######################## Cargar resultados de MIM ########################
    ##########################################################################

    
    Sigma_1=[]
    Sigma_2=[]
    Sigma_3=[]
    fhi=[]
    
    tes1=[]
    tes2=[]
    
    with open(name+'_outputMIM.txt') as f:
        for line in f:
            Sigma_1.append([float(line.split()[0]),float(line.split()[1])])   
            tes1.append(float(line.split()[0]))
            tes2.append(float(line.split()[1]))
            Sigma_3.append([float(line.split()[2]),float(line.split()[3])])
            fhi.append(float(line.split()[4]))
    
    Sigma_1x=[] ## esto es para tener sigma1 en lat lon y calcular angulos
    Sigma_1y=[]
    Sigma_3x=[] ## esto es para tener sigma 2 en lat lon
    Sigma_3y=[]
    
    ##########################################################################
    ########################## Calcular sigma 2 ##############################
    ##########################################################################
    
    Sigma_2=[]
    Sigma_2x=[] ## esto es para plotear más adelante todos los sigma2
    Sigma_2y=[]
    
    
    for i in range(0,len(Sigma_1)):
        Sigma1=Sigma_1[i]
        Sigma3=Sigma_3[i]
    
        P1=stereo.plunge_bearing2pole(Sigma1[1], Sigma1[0])
        S1=stereo.pole(P1[0],P1[1])
        s1=stereo.stereonet2xyz(S1[0][0],S1[1][0])
        s1=np.array([s1[0][0],s1[1][0],s1[2][0]])
                           
        P3=stereo.plunge_bearing2pole(Sigma3[1], Sigma3[0])
        S3=stereo.pole(P3[0],P3[1])
        s3=stereo.stereonet2xyz(S3[0][0],S3[1][0])
        s3=np.array([s3[0][0],s3[1][0],s3[2][0]])
                           
        s2=np.cross(s1, s3)
        if s2[2]>0:
            s2=-s2
        ## Si z > 0 de s2 ... Significa que el vector (x,y,z) está en 
        # el hemisferio superir de la proyección de stereonet; con un negativo se 
        # proyecta en el hemiserfio inferior.
        S2=stereo.xyz2stereonet(s2[0],s2[1],s2[2])
            
        Sigma2=stereo.vector2plunge_bearing(s2[0], s2[1], s2[2])
        Sigma_2.append([Sigma2[1][0],Sigma2[0][0]])
        
        ## Lo sigueiente es para plotear todos los sigma2 y calcular confindence interval
        ## se guardan en coordenadas lat lon    
        Sigma_1x.append(S1[0])
        Sigma_1y.append(S1[1])
        Sigma_2x.append(S2[0])
        Sigma_2y.append(S2[1])
        Sigma_3x.append(S3[0])
        Sigma_3y.append(S3[1])
        
        
    
    ##########################################################################
    ######################### Selecting Sigma trio ###########################
    ##########################################################################

    ## Step1: Pre re-organization of variables
        
    ## se separan en vectores de sigma 1 trend and plunge... sigma 2 ... y sigma 3.
    ## Es un poco torpe/bruto este paso, pero se me hizo más fácil para la programacion
    ## de las figuras y density_grid (puede hacerse en un paso antes... pero sigue)
    ## siendo rápido ya que el largo de los resultdos de MIM no es tan largo 
    ##(226 en el caso de Puyuhuapi)
        
    S1_s=[]
    S1_d=[]
    S2_s=[]
    S2_d=[]
    S3_s=[]
    S3_d=[]
    
    
    for i in range(0,len(Sigma_1)):
        
        poles_i=stereo.plunge_bearing2pole(Sigma_1[i][1], Sigma_1[i][0])
        S1_s.append(poles_i[0][0]) ## esto es strike/dip de plano cuyo polo es Sigma1
        ## aunque suene raro, hay que ordenarlas así para poder hacer el density grid
        ## y density_contourf de stereonet bien. Esto se puede entender considerando
        ## que dichas funciones de stereonet entregan las densidades de polos de planos 
        S1_d.append(poles_i[1][0])
        poles_i=stereo.plunge_bearing2pole(Sigma_2[i][1], Sigma_2[i][0])
        S2_s.append(poles_i[0][0])
        S2_d.append(poles_i[1][0])
        poles_i=stereo.plunge_bearing2pole(Sigma_3[i][1], Sigma_3[i][0])
        S3_s.append(poles_i[0][0])
        S3_d.append(poles_i[1][0])
    
        
        
    ### Step 2: Seleccion of S1 and S3
        
    ## aquí empieza la eleccion de la solucion... se elige la solución de mayor densidad
    ## de sigma 1 y sigma 3... y se calcula simga 2.     
        
    inf=stereo.density_grid(S1_s,S1_d) ## al usar la funcion Density grid de stereo
    # estamos utilizando el algortmo de Kamb actualizado para datos geográficos por
    # Vollmer 1995. Esto es probabilistico, no solo cantidad de datos tipo histograma.
    x=np.reshape(inf[0],-1)
    y=np.reshape(inf[1],-1)
    z=np.reshape(inf[2],-1)
    
    pos=np.argmax(z)
    selected_sig1=[x[pos],y[pos]]  ## está en lat lon listo para plotear
    
    inf=stereo.density_grid(S3_s,S3_d)
    x=np.reshape(inf[0],-1)
    y=np.reshape(inf[1],-1)
    z=np.reshape(inf[2],-1)
    
    pos=np.argmax(z)
    selected_sig3=[x[pos],y[pos]]  ## está en lat lon listo para plotear
       
    
    ## Step 3: calculation of S2
    
    s1=stereo.stereonet2xyz(selected_sig1[0],selected_sig1[1])
    s1=np.array([s1[0][0],s1[1][0],s1[2][0]])                    
    s3=stereo.stereonet2xyz(selected_sig3[0],selected_sig3[1])
    s3=np.array([s3[0][0],s3[1][0],s3[2][0]])
                       
    s2=np.cross(s1, s3)
    if s2[2]>0:
        s2=-s2
    ## Si z > 0 de s2 ... Significa que el vector (x,y,z) está en 
    # el hemisferio superir de la proyección de stereonet; con un negativo se 
    # proyecta en el hemiserfio inferior.
    S2=stereo.xyz2stereonet(s2[0],s2[1],s2[2])
    selected_sig2=[S2[0][0],S2[1][0]]
    
    Sigma1=stereo.vector2plunge_bearing(s1[0], s1[1], s1[2])
    selected_sig1_geo=[Sigma1[1][0],Sigma1[0][0]]
    Sigma2=stereo.vector2plunge_bearing(s2[0], s2[1], s2[2])
    selected_sig2_geo=[Sigma2[1][0],Sigma2[0][0]]
    Sigma3=stereo.vector2plunge_bearing(s3[0], s3[1], s3[2])
    selected_sig3_geo=[Sigma3[1][0],Sigma3[0][0]]
    
    
    ## Step 4: Selection of fhi
    
    ## sacar una estadistica a un radio de sigma 1 elegido
    
    ang=10 ## angulo de diferencia donde se calculará la estadísitica de fhi
    
    ## se calcula un vector que simule un cono de ANG°
    cono1=stereo.stereonet_math.cone(selected_sig1_geo[1],selected_sig1_geo[0],ang)
    cono3=stereo.stereonet_math.cone(selected_sig3_geo[1],selected_sig3_geo[0],ang)
    
    ## el cono se pasa a poligono
    polygonS1=[]     
    polygonS3=[]     
    
    for i in range(0,np.size(cono1,2)):
        polygonS1.append((cono1[0][0][i],cono1[1][0][i]))
        polygonS3.append((cono3[0][0][i],cono3[1][0][i]))   
    
    poly1=path.Path(polygonS1)
    poly3=path.Path(polygonS3)
        
    ## se pasan los datos de sigm1 a lat lon
    in_lat_lonS1=stereo.pole(S1_s,S1_d)
    in_lat_lonS3=stereo.pole(S3_s,S3_d)
    
    ## se genera el booliano que dice que puntos están más cerca q  Ang° grados 
    ## de sigma 1 elegido
    points=[]
    for i in range(0,np.size(in_lat_lonS1,1)):
        points.append((in_lat_lonS1[0][i],in_lat_lonS1[1][i]))    
    in_poly1=poly1.contains_points(np.array(points)) 
    
    points=[]
    for i in range(0,np.size(in_lat_lonS3,1)):
        points.append((in_lat_lonS3[0][i],in_lat_lonS3[1][i]))    
    in_poly3=poly3.contains_points(np.array(points)) 
    
    ## se extraen fhi a menos de 10°
    fhi=np.array(fhi)
    fhi_extracted=np.append(fhi[in_poly1],fhi[in_poly3],0)
    
    ## Se selecciona la mediana de los fhi extraidos, como representativo de los fhi,
    ## para definir el prinicpal stress setting.
    fhi_selected=np.median(fhi_extracted)
    
    if plot_fhi_extracted_histogram:        
        
        case=case+1
        fig_hist=n_fig_principal+200
        plt.figure(fig_hist+case)
        plt.hist(fhi,bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        plt.hist(fhi_extracted,bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        plt.legend(['Fhi complete statistics','Fhi_extracted for median fhi calculation'])
        plt.title('Fhi of case '+name)
    
    ## Here, the stress setting is finished
    
    ##########################################################################
    ############# Extraction of X percentile-target MIM-results ##############
    ##########################################################################
    

    ## 1) medir angulos "misfit" a posibilidades
        
    ang1=DDT.dif_angular(selected_sig1,Sigma_1x,Sigma_1y)
    ang2=DDT.dif_angular(selected_sig2,Sigma_2x,Sigma_2y)
    ang3=DDT.dif_angular(selected_sig3,Sigma_3x,Sigma_3y)
    
    ## seleccionar el XX% de las soluciones más cercanas a la elegida
    # percentile_target=75 ## percentile a eliminar definido al principio
    
    ## coordenads en lat lon de Sigmas con misfit menor a XX%
    s1x_90=np.array(Sigma_1x)[ang1<np.percentile(ang1,percentile_target)]
    s1y_90=np.array(Sigma_1y)[ang1<np.percentile(ang1,percentile_target)]
    s2x_90=np.array(Sigma_2x)[ang2<np.percentile(ang2,percentile_target)]
    s2y_90=np.array(Sigma_2y)[ang2<np.percentile(ang2,percentile_target)]
    s3x_90=np.array(Sigma_3x)[ang3<np.percentile(ang3,percentile_target)]
    s3y_90=np.array(Sigma_3y)[ang3<np.percentile(ang3,percentile_target)]
    
    ## planos strike/dip del xx% de datos
    S1s_90=np.array(S1_s)[ang1<np.percentile(ang1,percentile_target)]
    S1d_90=np.array(S1_d)[ang1<np.percentile(ang1,percentile_target)]
    S2s_90=np.array(S2_s)[ang2<np.percentile(ang2,percentile_target)]
    S2d_90=np.array(S2_d)[ang2<np.percentile(ang2,percentile_target)]
    S3s_90=np.array(S3_s)[ang3<np.percentile(ang3,percentile_target)]
    S3d_90=np.array(S3_d)[ang3<np.percentile(ang3,percentile_target)]
    
    ##############################################################################
    ########################### Plot de distribuciones ###########################
    ##############################################################################
    
    if plot_principal_stress_distribution_figure:
            
        main_fig=plt.figure(n_fig_principal,figsize=(20.0,15.0))
        
        ### plotaer datos
        ax11=main_fig.add_subplot(3,5,1,projection='stereonet')
        ax12=main_fig.add_subplot(3,5,6,projection='stereonet')
        ax13=main_fig.add_subplot(3,5,11,projection='stereonet')
        plt.figure(n_fig_principal)
        plt.subplot(3,5,1)
        plt.plot(Sigma_1x,Sigma_1y,'k.')
        plt.plot(s1x_90,s1y_90,'b.')
        plt.ylabel('Sigma 1')
        plt.title('Data')
        plt.subplot(3,5,6)
        plt.plot(Sigma_2x,Sigma_2y,'k.')
        plt.plot(s2x_90,s2y_90,'b.')
        plt.ylabel('Sigma 2')
        plt.subplot(3,5,11)
        plt.plot(Sigma_3x,Sigma_3y,'k.')
        plt.plot(s3x_90,s3y_90,'b.')
        plt.ylabel('Sigma 3')
    
    
        ### plotear Kamb distribution 100%
        
        niveles=[0.9,1,2,5,10,15,20,25,30,50]
        
        a=stereo.density_grid(np.array(S1_s), np.array(S1_d))
        ax21=main_fig.add_subplot(3,5,2,projection='stereonet')
        ax21.contourf(a[0],a[1],a[2],cmap=cm.RdYlBu_r,levels=niveles)
        a=stereo.density_grid(np.array(S2_s), np.array(S2_d))
        ax22=main_fig.add_subplot(3,5,7,projection='stereonet')
        ax22.contourf(a[0],a[1],a[2],cmap=cm.RdYlBu_r,levels=niveles)
        a=stereo.density_grid(np.array(S3_s), np.array(S3_d))
        ax23=main_fig.add_subplot(3,5,12,projection='stereonet')
        ax23.contourf(a[0],a[1],a[2],cmap=cm.RdYlBu_r,levels=niveles)
        
        ### plotear Kamb distribution XX% of the data
        
        a2=stereo.density_grid(np.array(S1s_90), np.array(S1d_90))
        ax31=main_fig.add_subplot(3,5,3,projection='stereonet')
        ax31.contourf(a2[0],a2[1],a2[2],cmap=cm.RdYlBu_r,levels=niveles)
        a2=stereo.density_grid(np.array(S2s_90), np.array(S2d_90))
        ax32=main_fig.add_subplot(3,5,8,projection='stereonet')
        ax32.contourf(a2[0],a2[1],a2[2],cmap=cm.RdYlBu_r,levels=niveles)
        a2=stereo.density_grid(np.array(S3s_90), np.array(S3d_90))
        ax33=main_fig.add_subplot(3,5,13,projection='stereonet')
        ax33.contourf(a2[0],a2[1],a2[2],cmap=cm.RdYlBu_r,levels=niveles)
        
        
        ### Ploteat distribución tipo datos.
        
        ax41=main_fig.add_subplot(3,5,4,projection='stereonet')    
        DDT.density_distribution(Sigma_1x,Sigma_1y,'latlon',False,True,ax41)
        ax42=main_fig.add_subplot(3,5,9,projection='stereonet')    
        DDT.density_distribution(Sigma_2x,Sigma_2y,'latlon',False,True,ax42)
        ax43=main_fig.add_subplot(3,5,14,projection='stereonet')    
        DDT.density_distribution(Sigma_3x,Sigma_3y,'latlon',False,True,ax43)
        
        ### Ploteat distribución tipo datos XX% of data
        ax51=main_fig.add_subplot(3,5,5,projection='stereonet')    
        DDT.density_distribution(s1x_90,s1y_90,'latlon',False,True,ax51)
        ax52=main_fig.add_subplot(3,5,10,projection='stereonet')    
        DDT.density_distribution(s2x_90,s2y_90,'latlon',False,True,ax52)
        ax53=main_fig.add_subplot(3,5,15,projection='stereonet')    
        DDT.density_distribution(s3x_90,s3y_90,'latlon',False,True,ax53)
        
        main_fig.suptitle(name)
        
        plt.figure(n_fig_principal)
        plt.subplot(3,5,2)
        plt.title('Kamb distribution 100%')
        plt.subplot(3,5,3)
        plt.title('Kamb distribution '+str(percentile_target)+'% of data')
        plt.subplot(3,5,4)
        plt.title('Histogram distribution 100%')
        plt.subplot(3,5,5)
        plt.title('Histogram distribution '+str(percentile_target)+'% of data')
        
        if save_stress_distribution_fig:
            plt.savefig(name+'_stress_distribution.png')
    
    ##############################################################################
    ########################### Plot Figura de stress ############################
    ##############################################################################    
        
    ### plotear Kamb distribution XX% of the data
    # niveles=[0.9,1,2,5,10,15,20,25,30,70]
    
    if plot_stress_figure:
            
        niveles2=[0.01,0.25,0.5,0.75,1]
        fig_stress=plt.figure(n_fig_principal+100,figsize=(10.0,4.0))      
    
        a2=stereo.density_grid(np.array(S1s_90), np.array(S1d_90))
        a2[2][a2[2]<1]=0
        as1=fig_stress.add_subplot(1,2,1,projection='stereonet')
        as1.contourf(a2[0],a2[1],a2[2]/np.max(a2[2]),cmap='Reds',levels=niveles2)
        a2=stereo.density_grid(np.array(S3s_90), np.array(S3d_90))
        a2[2][a2[2]<1]=0
        as1.contourf(a2[0],a2[1],a2[2]/np.max(a2[2]),cmap='Blues',levels=niveles2)    
    
        plt.subplot(1,2,2)
        plt.hist(fhi,bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        
        fig_stress.suptitle(name)
    
        if save_stress_fig:
            plt.savefig(name+'_stress.png')
            plt.savefig(name+'_stress.eps')
        
    
    ##############################################################################
    ################################# Outputs  ###################################
    ##############################################################################
    
    
    print('Results of the selected stress setting')
    print('Sigma1 [trend,plunge]')
    print(selected_sig1_geo)
    print('Sigma2 [trend,plunge]')
    print(selected_sig2_geo)
    print('Sigma3 [trend,plunge]')
    print(selected_sig3_geo)
    print('Fhi selected')
    print(fhi_selected)
    
    if write_sigmas_prob:
            
        ### escribir sigmas
        nameS1=name+'_S1.txt'
        DDT.write_density_distribution(S1s_90,S1d_90,nameS1)
        nameS3=name+'_S3.txt'
        DDT.write_density_distribution(S3s_90,S3d_90,nameS3)
    
        ### escibir fhi
        name_fhi=name+'_fhi.txt'
        fhi_90=np.array(fhi)[(ang1<np.percentile(ang1,percentile_target)) & (ang3<np.percentile(ang3,percentile_target))]
        
        fhi_histo=np.histogram(fhi_90,bins=50)
        fhi_val=fhi_histo[1]
        fhi_prob=fhi_histo[0]	
		
        idN=-1
        fidfhi=open(name_fhi,'w')
        for i in range(0,len(fhi_val)-1):
            idN=idN+1
            fhi_escribir=(fhi_val[i]+fhi_val[i+1])/2
            fidfhi.write(' '+str(idN)+' '+str(fhi_escribir)+' '+str(fhi_prob[i])+' \n')
        
        fidfhi.close()
            
    n_fig_principal=n_fig_principal+1 ## this is for several files proceced at ones.
   