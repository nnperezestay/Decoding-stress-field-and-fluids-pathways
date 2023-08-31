# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:01:28 2020

@author: NicoP
"""

## Loading libreries
import matplotlib.pyplot as plt
import numpy as np
import mplstereonet as stereo
from scipy.interpolate import griddata
import matplotlib.cm as cm
import time

#%%
##############################################################################
################################### Settings #################################
##############################################################################  

tic = time.time()

                             ## Settings of file ##
 
name='Test_PuyuhuapiLongTerm'
name_case=name

                     ## Setting of plots and saving figures ##

plot_sigmas=False ## It will plot the S1 and S3 randomly selected in monte carlo simulation
fig_sigmas=101 ## number of the selected sigmas figure.

plot_results_of_each_monte_carlo=False ## plots results of each case of monte carlo
## WARNING: Do not put True for more than 50-100 Monte Carlo cases, depends on your computer.
if plot_results_of_each_monte_carlo:
    n_fig_for_each_montecarlo=1000  ## number of the figure with the first monte carlo case


plot_dilatational_tendency_results=True ## Figure n°1
save_dilatational_tendency_results=True ## Figure n°1

plot_pore_pressure_results=True ## Figure n°2
save_pore_pressure_results=False
saturating_pore_pressure_to_max=True ## If Sigma 1 is largely greater than Sigma vertical, 
## pore pressure to trigger tensile fracture normalized by the vertical stress (know as lamda parameter
## ... see methods equations) could has values farther than 1, as 5 or even 10 depeding of the stress 
## setting. Here, we plot lamda parameter in an uniform scale from 0 to 2, to allow the comparison
## of different volcanic cases at the same scale. Then, when lamda values are greather than 2 this values
## are not plotted (please see results when saturating_pore_pressure_to_max is False). 
## Therefore, we create this auxliar variable, that truncate any value of lamda >= 2 to 2, just for 
## plot, and do not affect the calculations. Please test results with this parameter in True or False
## with at least 100 monte carlo cases to understant its effect on plots.

## Standard level to plot all the variable in the same range. 
## It can be change base on the aims of the invesigation, but we suggest keep this one.      
niveles_lamda=[-0.0001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
niveles_dt=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# niveles_kine=[1,2,3,4,5,6,7,8] ## this variable is not analyzed in the code, but is calculated for future works
# niveles_sense=[0,1,2] ## this variable is not analyzed in the code, but is calculated for future works

                    ## Settings of Monte Carlo Simulation ##

n_montecarlo=1000 ## amount of monte carlo cases
## between 1000-5000 the results seems to be stable with different random parameters
## 10.000 is to much, since it create a matrix of 2.6 Gb that could be difficult to transpose
## my computer bug there, and I can not reorganize the results to observe the statistics of results
## I suggest 1000 cases for oficial results, but it can change depending of the range of parameters.

Tmin=0.6 ## tensile resistence in MPa
Tmax=10
zmin=1 ## en km
zmax=15
taumin=1 ## differential stress in MPa
taumax=250

rho=2700
g=9.8

## Parameters about strike_planes and dip_planes serching
## plase select one. Options 4 and 5 are for undertanding the code and results.
## Option 1 to 3 are created for obtaining oficial results reliable to be analyzed.
## Option 1 is suggested as default.

## Op1. Minimun oficial setting (each 5° of dipping and 10° of strike)
strike_planes=np.linspace(0,360,37)
dip_planes=np.linspace(0,89.9,19)

## Op2. Oficial setting aprox. 1.1 secods per monte carlo case
## Este setting es suficiente para Dilatational Tendency y Pore Pressure;
## para Kinematics mejor correr el de mega presición.
# strike_planes=np.linspace(0,360,60)
# dip_planes=np.linspace(0,89.9,20)

## Op3. High presition (take ~9 seconds per monte carlo case) --> only for the paper
# strike_planes=np.linspace(0,360,121)
# dip_planes=np.linspace(0,89.9,91)

## Op.4 Quick setting
# strike_planes=np.linspace(0,360,15)
# dip_planes=np.linspace(0,89.9,10)

## Op.5 Mega-quick setting
# strike_planes=np.linspace(0,360,5)
# dip_planes=np.linspace(0,89.9,5)

#%%
##############################################################################
######################## Initial variable of the code ########################
##############################################################################

name=name_case

#### 3D matrix 3D with results, were i,j are the strike and dip separation selected
## in strike_planes and dip_planes variables; and k is the amount of monte carlo cases
dt_super=[] ## for dilatational tendency
lamda_super=[] ## pore pressure normalized by the lithostatic pressure.

if plot_sigmas: ## for plotting the randomly selected sigmas.
    fig2=plt.figure(fig_sigmas, figsize=(20.0, 15.0))
    ax2=fig2.add_subplot(111,projection='stereonet')      

## points_planes is an auxiliar variable to create heat map of dilatational tendency and lamda
## also help to create a 2D martix of dt and lamda results of the case "k" of monte carlo simulation
cont=0 
points_planes=np.ndarray(shape=(len(strike_planes)*len(dip_planes),2),dtype=float)       
for i in range(0,len(strike_planes)):
    for j in range(0,len(dip_planes)):
        points_planes[cont,0]=strike_planes[i]
        points_planes[cont,1]=dip_planes[j]
        cont=cont+1
            
## grid of strike and dip with 1° of resolution     
delta_strike=1
delta_dip=1
grid_strike, grid_dip = np.mgrid[min(strike_planes):max(strike_planes):delta_strike,min(dip_planes):max(dip_planes):delta_dip]           


time_step0 = time.time()-tic
print(str(time_step0)[0:-10]+' seconds')       
print('Loding intial variable for calculations')  
print('In '+str(time_step0)[0:-10]+' seconds')       
tic=time.time() 


###############################################################################
###############################################################################
##%%%%%%%%%%%%%%%%%%############ MAIN CODE #############%%%%%%%%%%%%%%%%%%%%%##
###############################################################################
###############################################################################

#%% Step1:
###############################################################################
############# Load probabilistic distribution of S1, S3 and fhi ############### 
###############################################################################

S1_t=[]
S1_p=[]
n1=[]
val1=[]

with open(name+'_S1.txt') as f:
    for line in f:
        S1_t.append(float(line.split()[1]))   
        S1_p.append(float(line.split()[2]))
        n1.append(float(line.split()[0]))
        val1.append(float(line.split()[3]))

S3_t=[]
S3_p=[]
n3=[]
val3=[]

with open(name+'_S3.txt') as f:
    for line in f:
        S3_t.append(float(line.split()[1]))   
        S3_p.append(float(line.split()[2]))
        n3.append(float(line.split()[0]))
        val3.append(float(line.split()[3]))

fhi1=[]
nfhi=[]
valfhi=[]

with open(name+'_fhi.txt') as f:
    for line in f:
        fhi1.append(float(line.split()[1]))   
        nfhi.append(float(line.split()[0]))
        valfhi.append(float(line.split()[2]))


#%% Step2:
############################################################################################
### Definition principal stresses orientation and stress ratio for each Monte Carlo case ###
############################################################################################

S1trend_sel=[]
S1plunge_sel=[]
S3trend_sel=[]
S3plunge_sel=[]

## vector of the probabilistic distribution loaded from text file in the previous step
## in this step are normalized. 
proba1=np.array(val1)/sum(val1) 
proba3=np.array(val3)/sum(val3)
    
for i in range(0,n_montecarlo):
    
    S1oS3=np.random.randint(0,2) ## variable to randomly select whether the code start defining S1 (0) or S3 (1) 
    
    s1_and_s3_not_ok=True ## auxiliar variable to check that S1 and S3 selection is ok
    
    while s1_and_s3_not_ok:
        if S1oS3==0: ## start with S1
        
            ## Choose randomly S1 base on distribution load in proba1
            sel1=np.random.choice(len(n1),1,p=proba1)[0]
            
            ## To choose S3 and be sure that it is perpendicular to S1, we create a plane with its pole 
            ## defined by S1-direction
            planoS3=stereo.plunge_bearing2pole(S1_p[sel1],S1_t[sel1]) ## strike/dip
            ## Create points of the plane of S3 in xy space
            xy_planoS3=stereo.plane(planoS3[0][0],planoS3[1][0])
            
            ## Now we build the probabilistic function to select S3, defined in each part of the plane.
            ## base on the distribution proba3
            val_planoS3=[]
            trend_planoS3=[]
            plunge_planoS3=[]
            n_planoS3=[]
            
            for j in range(0,np.shape(xy_planoS3)[1]):
                ## Calculation of trend and plunge of xy part of the plane
                auxiliar=stereo.geographic2plunge_bearing(xy_planoS3[0][j], xy_planoS3[1][j])
                ## We define the probabilistic value of this point selecting the closer value of proba3.  
                ## We use the variables S3_t (S3_trend) and S3_p (S3_plung) of proba3.
                dif=np.sqrt((auxiliar[0][0]-S3_p)**2+(auxiliar[1][0]-S3_t)**2)
                posmin=np.argmin(dif)
                val_planoS3.append(val3[posmin])    
                ## other variable to save trend and plung of S3, that will help to generate randomly S3
                n_planoS3.append(j)
                trend_planoS3.append(auxiliar[1][0])
                plunge_planoS3.append(auxiliar[0][0])
            
            ## Porbabilistic function of the plane normalized
            proba_aux=np.array(val_planoS3)/sum(val_planoS3)
            
            if np.sum(np.isnan(proba_aux))==len(proba_aux):
                ## If this statement is true, the plane selected base on S1-direction do not
                ## have any probability of been choose on S3 distribution proba3.
                s1_and_s3_not_ok=True 
                ## Then, the while continue until find a real posibility of pair S1 and S3-directions
            else:
                ## There is some posibility of S3 in the plane with S1 as pole
                ## Then, we selected S3 base on the auxiliar distribution created for the plane.
                sel3=np.random.choice(len(n_planoS3),1,p=proba_aux)[0] 
                S3trend_sel.append(trend_planoS3[sel3])
                S3plunge_sel.append(plunge_planoS3[sel3])
                S1trend_sel.append(S1_t[sel1])
                S1plunge_sel.append(S1_p[sel1])
                s1_and_s3_not_ok=False ## the while is finished
                
        elif S1oS3==1: ## start with S3
            
            ## Choose randomly S3 base on distribution load in proba3
            sel3=np.random.choice(len(n3),1,p=proba3)[0]
            
            ## To choose S1 and be sure that it is perpendicular to S3, we create a plane with its pole 
            ## defined by S3-direction
            planoS1=stereo.plunge_bearing2pole(S3_p[sel3],S3_t[sel3])
            ## Create points of the plane of S1 in xy space
            xy_planoS1=stereo.plane(planoS1[0][0],planoS1[1][0])
            
            ## Now we build the probabilistic function for selecting S1 in each part of the plane.
            ## base on the distribution proba1
            val_planoS1=[]
            trend_planoS1=[]
            plunge_planoS1=[]
            n_planoS1=[]
            
            for j in range(0,np.shape(xy_planoS1)[1]):
                
                ## Calculation of trend and plunge of xy part of the plane
                auxiliar=stereo.geographic2plunge_bearing(xy_planoS1[0][j], xy_planoS1[1][j])
                ## We define the probabilistic value of this point selecting the closer value of proba1.  
                ## We use the variables S1_t (S1_trend) and S1_p (S1_plung) of proba1.
                dif=np.sqrt((auxiliar[0][0]-S1_p)**2+(auxiliar[1][0]-S1_t)**2)
                posmin=np.argmin(dif)
                val_planoS1.append(val1[posmin])    
                ## other variable to save trend and plung of S1, that will help to generate randomly S1
                n_planoS1.append(j)
                trend_planoS1.append(auxiliar[1][0])
                plunge_planoS1.append(auxiliar[0][0])
            
            ## Porbabilistic function of the plane normalized
            proba_aux=np.array(val_planoS1)/sum(val_planoS1)
            
            if np.sum(np.isnan(proba_aux))==len(proba_aux):
                ## If this statement is true, the plane selected base on S3-direction do not
                ## have any probability of been choose on S1 distribution proba1.
                s1_and_s3_not_ok=True ## The while should continue
                ## Then, the while continue until find a real posibility of pair S1 and S3-directions
            
            else: 
                ## There is some posibility of S1 in the plane with S3 as pole
                ## Then, we selected S1 base on the auxiliar distribution created for the plane.
                sel1=np.random.choice(len(n_planoS1),1,p=proba_aux)[0]
                S1trend_sel.append(trend_planoS1[sel1])
                S1plunge_sel.append(plunge_planoS1[sel1])
                S3trend_sel.append(S3_t[sel3])
                S3plunge_sel.append(S3_p[sel3])
                s1_and_s3_not_ok=False ## the while is finished
    
    
    toc = time.time()    
    print('selected stress n='+str(i))
    print(str(toc-tic)[0:-10]+' seconds')  

dif=[] ## we erase this variable for saving memory


  
## Selection of fhi
probafhi=np.array(valfhi)/sum(valfhi)
selfhi=np.random.choice(len(nfhi),n_montecarlo,p=probafhi)

fhi_sel=[]
for seli in selfhi:          
    fhi_sel.append(fhi1[seli])


time_step1 = time.time()-tic
print(str(time_step1)[0:-10]+' seconds')       
print('The random selection of principal stress orientation and stress ratio if finished')  
print('In '+str(time_step1)[0:-10]+' seconds')       
tic=time.time() 


#%% Step 3: Monte carlo simulation
##############################################################################
########################## Monte Carlo simulation ############################
##############################################################################

for n_monte in range(0,n_montecarlo):  ## for each monte carlo case
    
    ### S1, S3 orientations and fhi are defined
    Sigma1=[S1trend_sel[n_monte],S1plunge_sel[n_monte]]    
    Sigma3=[S3trend_sel[n_monte],S3plunge_sel[n_monte]]
    fhi=fhi_sel[n_monte]

    ##########################################################################
    #################### Calculate S2 for "n_monte" case #####################
    ##########################################################################
    
    P1=stereo.plunge_bearing2pole(Sigma1[1], Sigma1[0])
    S1=stereo.pole(P1[0],P1[1])
    s1xyz=stereo.stereonet2xyz(S1[0][0],S1[1][0])
    s1xyz=np.array([s1xyz[0][0],s1xyz[1][0],s1xyz[2][0]]) ## S1 in xyz coordinates
                       
    P3=stereo.plunge_bearing2pole(Sigma3[1], Sigma3[0])
    S3=stereo.pole(P3[0],P3[1])
    s3xyz=stereo.stereonet2xyz(S3[0][0],S3[1][0])
    s3xyz=np.array([s3xyz[0][0],s3xyz[1][0],s3xyz[2][0]]) ## S3 in xyz coordinates
                       
    s2xyz=np.cross(s1xyz, s3xyz) ## calculation of S2
    if s2xyz[2]>0:
        s2xyz=-s2xyz
    ## If z>0 in S2, the vector (x,y,z) is located in the upper hemisphere of the stereonet
    ## and should be multiply by -1 to be in the lower hemisphere. If z<0 S2 is on the 
    ## lower hemisphere, and do not required any corrections.
    S2=stereo.xyz2stereonet(s2xyz[0],s2xyz[1],s2xyz[2])
            
    Sigma_2=stereo.vector2plunge_bearing(s2xyz[0], s2xyz[1], s2xyz[2]) ## S2 as trend and plunge
    Sigma2=[Sigma_2[1][0],Sigma_2[0][0]]

    ## Step 3.1: Random parameters
    ##########################################################################
    ######### Define the other Monte Carlo parameters randonmly  #############
    ##########################################################################
    
    not_ok=True ## auxiliar variable to check that S1 is greater than Sv
    cont_vueltas=0
    while not_ok:
        
        ### Random selection of Tensile Strength (T), Depth (zi), and Diferential Stress (xi)
        T=np.random.randint(Tmin,Tmax)+np.random.random(1)[0] ## en MPa
        zi=np.random.randint(zmin,zmax)+np.random.random(1)[0] ## en km
        xi=np.random.randint(taumin,taumax) ## stress diferencial en MPa (según fórmulas)
    
        ## Step 3.2. Aplication of the non-andersonian equation of methods
        ##########################################################################
        ############# Calculation of principal stress scalar values ##############
        ##########################################################################
        
        sin1=np.sin(np.deg2rad(Sigma1[1]))
        sin2=np.sin(np.deg2rad(Sigma2[1]))
        sin3=np.sin(np.deg2rad(Sigma3[1]))
    
        ## calculation of principal stress on MPa by no-andersonean equations
        sv=rho*g*zi*10**3/10**6# z*10**3= in meters  /10**6--> to MPa
        s3=(sv-xi*(sin1+fhi*sin2))/(sin1+sin2+sin3) # z*10**3= in meters  /10**6--> to MPa
        s2=fhi*xi+s3
        s1=xi+s3
        
        if sv>s1: ## this is a mathematical degeneration of the problem since the assumtion used to
            ## define a scalar value of s1, s2 and s3 do not strictly restricted that s1 should be 
            ## the maximum strees
            not_ok=True
        if sv<s1:
            not_ok=False
        
        cont_vueltas=cont_vueltas+1
        if cont_vueltas>1000:
            not_ok=False ## this allow stop a bug without end, that rarely occurs
            
        
    ## Step. 3.3. Auxiliar variable to project stress on planes
    ###############################################################################
    ######### Projecting principal stresses to the planes (strike/dip) ############
    ###############################################################################
    
    ## construction of matrix that tranform prinicipal stress to North-East-Down coordinates
    ## this will be uselful to calculate the norma and shear stress on each plane.
    ## recomended lecture: Allmendinger, Cardozo and Ficher, 2012. Structural geology algorithms vectors and tensors
    
    if plot_sigmas:## ploting principal stress to have a control on the S1, S2, and S3 selections
        ax2.line(Sigma1[1],Sigma1[0],'ro',markersize=5)
        ax2.line(Sigma2[1],Sigma2[0],'go',markersize=5)
        ax2.line(Sigma3[1],Sigma3[0],'bo',markersize=5)
     
    pl1=np.deg2rad(Sigma1[1])
    pl2=np.deg2rad(Sigma2[1])
    pl3=np.deg2rad(Sigma3[1])
    
    ## trend of geology to NED angles
    ## 0 of tred is on the x-axis and y-axis is 90° (counterclockwise, opposite to geology nomesclature) 
    trd1=450-Sigma1[0]
    trd2=450-Sigma2[0]
    trd3=450-Sigma3[0]
    
    if trd1>=360:
        trd1=trd1-360
    if trd2>=360:
        trd2=trd2-360
    if trd3>=360:
        trd3=trd3-360
    tr1=np.deg2rad(trd1)
    tr2=np.deg2rad(trd2)
    tr3=np.deg2rad(trd3)
    
    ## Transformation matrix to pass principal stress to NED (North-East-Down)
    aij=[[np.sin(tr1)*np.cos(pl1),np.cos(tr1)*np.cos(pl1),np.sin(pl1)],
         [np.sin(tr2)*np.cos(pl2),np.cos(tr2)*np.cos(pl2),np.sin(pl2)],
         [np.sin(tr3)*np.cos(pl3),np.cos(tr3)*np.cos(pl3),np.sin(pl3)]]
    
    
    ##Step 3.4. The core of the monte carlo simulation.
    ############################################################################
    ### Definition of the grid evaluation of Dt and lamda at each strike/dip ###
    ############################################################################

    lamda_plane=[]
    dt_plane=[]
    st_plane=[]
    kine_plane=[]
    sense_plane=[]
    
    ######## calculation of each strike and dip defined in settings
    for strike_lin in strike_planes:
        for dip_lin in dip_planes:
            
            ## Here we caculate s/|s| --> the slip normalized direction
            ## and normal and shear stress in the different planes--> sig_n y tau_n
            ## recomended lecture: Allmendinger, Cardozo and Ficher, 2012. 
            ## Structural geology algorithms vectors and tensors
    
            ## Step 3.4.1 Definition of normal direction in NED    
            n_dir=[] ## normal direction
            
            ## cosine direction of normal vector of plane strike_lin, dip_lin
            n_ang=stereo.pole2plunge_bearing(strike_lin,dip_lin)
            n_dir=stereo.pole(strike_lin,dip_lin)
            n_dir=stereo.stereonet2xyz(n_dir[0],n_dir[1])
            
            ## transform n_direction from xyz to NED coodinate system.
            n_dir=np.array([n_dir[1][0],n_dir[0][0],-n_dir[2][0]])
            ## to pass to clasic xyz 
            # xyz_n=np.array([n_dir[0][0],n_dir[1][0],n_dir[2][0]])
            
            ### Now, the required transformation are already defined... aij and n_dir
            ## now we can project the stress on the plane "i" (strike_lin,dip_lin)
            
            ### Step 3.4.2. Calculate the strees in the plane
            ##  n-direction to principal stress coordinate system
            n_prima=np.dot(aij,n_dir)
            n_prima=n_prima/np.linalg.norm(n_prima)
            
            ## Principal stress matrix
            S_matrix=[[s1,0,0],
                      [0,s2,0],
                      [0,0,s3]]
            
            ## p_prima, the traction stress vector in principal stress coordinetes (NOT direction)
            p_prima=[]
            p_prima=np.dot(S_matrix,n_prima)            
            
            ## b and s directions
            b_prima=[]
            b_prima=np.cross(n_prima,p_prima)
            b_prima=b_prima/np.linalg.norm(b_prima)
            
            s_prima=[]
            s_prima=np.cross(n_prima,b_prima)
            s_prima=s_prima/np.linalg.norm(s_prima)
            
            ## transformation in NED-coordinate system
            cij=[[n_prima[0],n_prima[1],n_prima[2]],
                 [b_prima[0],b_prima[1],b_prima[2]],
                 [s_prima[0],s_prima[1],s_prima[2]]]
            
            ######## Calculation of the stress in the planeo
            ## _fault it equalt to double prima of Allmendinger exercise.
            p_fault=[]
            p_fault=np.dot(cij,p_prima)
            ## p_fault[0]=normal stress; p_fault[1]=0 (B-direction) y p_fault[2]=shear stress on fault
            
            ######## Trend and plung of slip direction "s"
            aij=np.array(aij)
            ## "s" is in NED
            s=np.dot(aij.transpose(),s_prima)
            s=s/np.linalg.norm(s)
            s_ang=stereo.vector2plunge_bearing(s[1], s[0], -s[2])
         
            ## Step 3.4.3 Definiton of the main variable.    
            #######################################################################
            ######### Calculation of dilatationl tendency and pore pressure #######
            #######################################################################
            
            ### Here the most important variables are:
            # 1) Fault kinematic
            # 2) Dilatational tendency = (sig1-sig_n)/(sig1-sig3)
            # 3) Slip tendency = tau/sigma_n 
            # 4) Pore pressure requiered for trigger tensile fracture
            # Here the code can be extended to many problems as undertaning the slip tendency in a fault.
            
            ### 1) fault type (normal, reverse, strike slip) NOT FUNDAMENTAL FOR THE PAPER
            rake_s=stereo.stereonet_math.azimuth2rake(strike_lin,dip_lin,s_ang[1])
            signo_rake=s[2]/np.abs(s[2]) ## s was defined in NED coordinates, so negative
            
            ## negative values means reverse movements, and conversaly positive values are normal slip
            if abs(rake_s)>67.5: ## the 90° quadrant divided in 4 classes: R RS SR SS
                if signo_rake>0:
                    linea_kine='R'
                    val_kin=1.5
                else:
                    linea_kine='N'
                    val_kin=7.5
            elif abs(rake_s)>45:
                if signo_rake>0:
                    linea_kine='RS'
                    val_kin=2.5
                else:
                    linea_kine='NS'
                    val_kin=6.5
            elif abs(rake_s)>22.5:
                if signo_rake>0:
                    linea_kine='SR'
                    val_kin=3.5
                else:
                    linea_kine='SN'
                    val_kin=5.5
            else:
                linea_kine='SS'
                val_kin=4.5
            
            ### Sense of movement
            if signo_rake==rake_s/abs(rake_s):
                SS_sense='Dextral'
                val_sense=0.5
            else:
                SS_sense='Sinestral'
                val_sense=1.5
            
            ###################
            ### 2) y 3) slip and dilatational tendecy
            
            st=-p_fault[2]/p_fault[0]
            dt=(s1-p_fault[0])/xi
            ## p_fault[0]=normal stress; p_fault[1]=0 (B-direction) y p_fault[2]=shear stress on fault
            
            ## auxiliar variables for future works
            tau=p_fault[2]
            b=p_fault[1]
            sn=p_fault[0]
            
            ################
            ## 4) Pore pressure requiered to trigger tensile fracture in the plane strike_lin, dip_lin
            uw_rup=sn+T
            if uw_rup<0:
                uw_rup=0
            if sv>0:
                lamda_rup=uw_rup/sv       
            else:
                lamda_rup='Nan'
         
            ## save of results of fundamental variables in the planes
            lamda_plane.append(lamda_rup)
            dt_plane.append(dt)
            st_plane.append(st)
            kine_plane.append(val_kin)
            sense_plane.append(val_sense)
            
            #Here the case of strike_lin, dip_lin is end.
    
    
    ## Here we came back the "n_monte" case of the simulation
    ## Dilatational tendency and Pore pressure have been calculated for all directions
    
    
    
    ##Step 3.5. Plot and save the results.
    ##########################################################################
    ####################### Create grid of Dt and Lamda ######################
    ##########################################################################
    
    ### grid with the reuslts (2D-matrix)
    grid_dt_plane = griddata(points_planes, dt_plane, (grid_strike, grid_dip), method='linear')
    grid_lamda_plane = griddata(points_planes, lamda_plane, (grid_strike, grid_dip), method='linear')
    grid_sense = griddata(points_planes, sense_plane, (grid_strike, grid_dip), method='linear')
    grid_kine= griddata(points_planes, kine_plane, (grid_strike, grid_dip), method='nearest')
    
    ### the grids are save in the mega 3D matrix with the grids of all monte carlo cases.
    dt_super.append(grid_dt_plane)
    lamda_super.append(grid_lamda_plane)

    if plot_results_of_each_monte_carlo:
        ## WARNING: here we can plot the results of each monte carlo case, but is 
        ## very conuming of RAM and graphical stats. DO NOT run for more than 100-200 monte carlo cases
        n_fig_for_each_montecarlo=n_fig_for_each_montecarlo+1                   
        plt.figure(n_fig_for_each_montecarlo)            
        plt.subplot(1,3,1)
        plt.title('Dilatation tendency')
        plt.contourf(grid_strike,grid_dip,grid_dt_plane,cmap=cm.Blues,levels=niveles_dt)
        plt.xlabel('Strike [°]')
        plt.ylabel('Dip [°]')
        cbar = plt.colorbar()
        cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
        plt.grid(linestyle='--', linewidth=1)      
            
        plt.subplot(1,3,2)
        plt.title('Lamda')
        plt.contourf(grid_strike,grid_dip,grid_lamda_plane,cmap=cm.RdYlBu_r,levels=niveles_lamda)
        plt.xlabel('Strike [°]')
        plt.ylabel('Dip [°]')
        cbar = plt.colorbar()
        cbar.set_label('Lamda_rup [MPa/Mpa]',fontsize=10)
        plt.grid(linestyle='--', linewidth=1)  
            
        plt.subplot(1,3,3)
        plt.title('Kinematics of faults')
        plt.contourf(grid_strike,grid_dip,grid_kine,cmap=cm.jet_r,levels=niveles_kine)
        plt.xlabel('Strike [°]')
        plt.ylabel('Dip [°]')
        cbar = plt.colorbar()
        cbar.set_label('Kinematics',fontsize=10) 
        plt.grid(linestyle='--', linewidth=1)    
     
    
    ##Step 3.6. Print advance of the code
    ##########################################################################
    ###################### Advancing processes printing ######################
    ##########################################################################
    toc = time.time()    
    print(str(toc-tic)[0:-10]+' seconds')   
    print('Monte Carlo case number='+str(n_monte+1))           
    print(name)
    
    ## Here the monte carlo case n_monte is finished!!! yuju!!  


time_step2 = time.time()-tic
print(str(time_step2)[0:-10]+' seconds')       
print('Good News!! Monte Carlo simulation has finished!')  
#%%     
tic=time.time() 

## Step 4. Statistical reorganization of results and plots
##############################################################################
####################### Reorganization of results ############################
##############################################################################

dt_ord=[]
lamda_ord=[]

mean_lamda=[]
median_lamda=[]
max_lamda=[]
min_lamda=[]
p25_lamda=[]
p75_lamda=[]
p10_lamda=[]
p90_lamda=[]
p5_lamda=[]
p95_lamda=[]
std_lamda=[]

mean_dt=[]
median_dt=[]
max_dt=[]
min_dt=[]
p25_dt=[]
p75_dt=[]
p10_dt=[]
p90_dt=[]
p5_dt=[]
p95_dt=[]
std_dt=[]

dt_super=np.array(dt_super)
lamda_super=np.array(lamda_super)

cont=0
points2=points_planes=np.ndarray(shape=(grid_strike.shape[0]*grid_strike.shape[1],2),dtype=float)
    
for i in range(0,grid_strike.shape[0]):
    for j in range(0,grid_strike.shape[1]):
        dt_ord.append(dt_super[:,i,j])
        lamda_ord.append(lamda_super[:,i,j])
        
        mean_dt.append(np.mean(dt_ord[-1]))
        median_dt.append(np.median(dt_ord[-1]))
        min_dt.append(np.min(dt_ord[-1]))
        max_dt.append(np.max(dt_ord[-1]))
        p10_dt.append(np.percentile(dt_ord[-1], 10))
        p90_dt.append(np.percentile(dt_ord[-1], 90))
        p5_dt.append(np.percentile(dt_ord[-1], 5))
        p95_dt.append(np.percentile(dt_ord[-1], 95))
        p25_dt.append(np.percentile(dt_ord[-1], 25))
        p75_dt.append(np.percentile(dt_ord[-1], 75))
        std_dt.append(np.std(dt_ord[-1]))
        
        
        mean_lamda.append(np.mean(lamda_ord[-1]))
        median_lamda.append(np.median(lamda_ord[-1]))
        min_lamda.append(np.min(lamda_ord[-1]))
        max_lamda.append(np.max(lamda_ord[-1]))
        p5_lamda.append(np.percentile(lamda_ord[-1], 5))
        p10_lamda.append(np.percentile(lamda_ord[-1], 10))
        p25_lamda.append(np.percentile(lamda_ord[-1], 25))
        p75_lamda.append(np.percentile(lamda_ord[-1], 75))
        p90_lamda.append(np.percentile(lamda_ord[-1], 90))
        p95_lamda.append(np.percentile(lamda_ord[-1], 95))
        std_lamda.append(np.std(lamda_ord[-1]))
        
        points2[cont,0]=grid_strike[i,j]
        points2[cont,1]=grid_dip[i,j]
        cont=cont+1
 
    
### Step. 4.1 Create dilatational tendency grids    
### create the statistical grids    
grid_dt_mean = griddata(points2, mean_dt, (grid_strike, grid_dip), method='linear')
grid_dt_median = griddata(points2, median_dt, (grid_strike, grid_dip), method='linear')
grid_dt_min = griddata(points2, min_dt, (grid_strike, grid_dip), method='linear')
grid_dt_max = griddata(points2, max_dt, (grid_strike, grid_dip), method='linear')
grid_dt_p5 = griddata(points2, p5_dt, (grid_strike, grid_dip), method='linear')
grid_dt_p10 = griddata(points2, p10_dt, (grid_strike, grid_dip), method='linear')
grid_dt_p25 = griddata(points2, p25_dt, (grid_strike, grid_dip), method='linear')
grid_dt_p75 = griddata(points2, p75_dt, (grid_strike, grid_dip), method='linear')
grid_dt_p90 = griddata(points2, p90_dt, (grid_strike, grid_dip), method='linear')
grid_dt_p95 = griddata(points2, p95_dt, (grid_strike, grid_dip), method='linear')
grid_dt_std = griddata(points2, std_dt, (grid_strike, grid_dip), method='linear')


### Step. 4.2 Plot dilatational tendency  
if plot_dilatational_tendency_results:

    plt.figure(1,figsize=(20.0,15.0))            
    plt.subplot(3,2,1)
    plt.title('Dilatation tendency')
    plt.contourf(grid_strike,grid_dip,grid_dt_median,cmap=cm.Blues,levels=niveles_dt)
    plt.xlabel('Strike [°]')
    plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
    #plt.plot(strike_planes,dip_planes,'bo')
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,3)
    plt.title('Mean')
    plt.contourf(grid_strike,grid_dip,grid_dt_mean,cmap=cm.Blues,levels=niveles_dt)
    # plt.xlabel('Strike [°]')
    plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
    #plt.plot(strike_planes,dip_planes,'bo')
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,4)
    plt.title('Standar deviation')
    plt.contourf(grid_strike,grid_dip,grid_dt_std,cmap=cm.Blues,levels=niveles_dt)
    # plt.xlabel('Strike [°]')
    # plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
    #plt.plot(strike_planes,dip_planes,'bo')
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,5)
    plt.title('Percentile 10')
    plt.contourf(grid_strike,grid_dip,grid_dt_p10,cmap=cm.Blues,levels=niveles_dt)
    # plt.xlabel('Strike [°]')
    plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
    #plt.plot(strike_planes,dip_planes,'bo')
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,6)
    plt.title('Percentile 25')
    plt.contourf(grid_strike,grid_dip,grid_dt_p25,cmap=cm.Blues,levels=niveles_dt)
    # plt.xlabel('Strike [°]')
    # plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
    #plt.plot(strike_planes,dip_planes,'bo')
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,7)
    plt.title('Percentile 75')
    plt.contourf(grid_strike,grid_dip,grid_dt_p75,cmap=cm.Blues,levels=niveles_dt)
    # plt.xlabel('Strike [°]')
    # plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
    #plt.plot(strike_planes,dip_planes,'bo')
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,8)
    plt.title('Percentile 90')
    plt.contourf(grid_strike,grid_dip,grid_dt_p90,cmap=cm.Blues,levels=niveles_dt)
    # plt.xlabel('Strike [°]')
    # plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
    #plt.plot(strike_planes,dip_planes,'bo')
    plt.grid(linestyle='--', linewidth=1)      
    
    plt.subplot(3,4,9)
    plt.title('Minimum Value')
    plt.contourf(grid_strike,grid_dip,grid_dt_min,cmap=cm.Blues,levels=niveles_dt)
    plt.xlabel('Strike [°]')
    plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
    #plt.plot(strike_planes,dip_planes,'bo')
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,10)
    plt.title('Percentile 5')
    plt.contourf(grid_strike,grid_dip,grid_dt_p5,cmap=cm.Blues,levels=niveles_dt)
    plt.xlabel('Strike [°]')
    # plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,11)
    plt.title('Percentile 95')
    plt.contourf(grid_strike,grid_dip,grid_dt_p95,cmap=cm.Blues,levels=niveles_dt)
    plt.xlabel('Strike [°]')
    # plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,12)
    plt.title('Maximum value')
    plt.contourf(grid_strike,grid_dip,grid_dt_max,cmap=cm.Blues,levels=niveles_dt)
    plt.xlabel('Strike [°]')
    # plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Dilatational_tendecy [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.suptitle(name)
    
    if save_dilatational_tendency_results:
        plt.savefig(name+'_Dilatational_tendency.png')
        plt.savefig(name+'_Dilatational_tendency.svg')


### Step. 4.3 Create lamda grids    
grid_lamda_mean = griddata(points2, mean_lamda, (grid_strike, grid_dip), method='linear')
grid_lamda_median = griddata(points2, median_lamda, (grid_strike, grid_dip), method='linear')
grid_lamda_min = griddata(points2, min_lamda, (grid_strike, grid_dip), method='linear')
grid_lamda_max = griddata(points2, max_lamda, (grid_strike, grid_dip), method='linear')
grid_lamda_p5 = griddata(points2, p5_lamda, (grid_strike, grid_dip), method='linear')
grid_lamda_p10 = griddata(points2, p10_lamda, (grid_strike, grid_dip), method='linear')
grid_lamda_p25 = griddata(points2, p25_lamda, (grid_strike, grid_dip), method='linear')
grid_lamda_p75 = griddata(points2, p75_lamda, (grid_strike, grid_dip), method='linear')
grid_lamda_p90 = griddata(points2, p90_lamda, (grid_strike, grid_dip), method='linear')
grid_lamda_p95 = griddata(points2, p95_lamda, (grid_strike, grid_dip), method='linear')
grid_lamda_std = griddata(points2, std_lamda, (grid_strike, grid_dip), method='linear')


### Step. 4.4 Plot lamda results
if plot_pore_pressure_results:
    
    if saturating_pore_pressure_to_max:
        max_level=np.max(niveles_lamda)
        grid_lamda_median[grid_lamda_median>=max_level]=max_level
        grid_lamda_mean[grid_lamda_mean>=max_level]=max_level
        grid_lamda_max[grid_lamda_max>=max_level]=max_level
        grid_lamda_p5[grid_lamda_p5>=max_level]=max_level
        grid_lamda_p10[grid_lamda_p10>=max_level]=max_level
        grid_lamda_p25[grid_lamda_p25>=max_level]=max_level
        grid_lamda_p75[grid_lamda_p75>=max_level]=max_level
        grid_lamda_p90[grid_lamda_p90>=max_level]=max_level
        grid_lamda_p95[grid_lamda_p95>=max_level]=max_level
        
    
    plt.figure(2,figsize=(20.0,15.0))
    plt.subplot(3,2,1)        
    plt.title('Median of Lamda')
    plt.contourf(grid_strike,grid_dip,grid_lamda_median,cmap=cm.RdYlBu_r,levels=niveles_lamda)
    plt.xlabel('Strike [°]')
    plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Lamda_rup [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1) 
    
    plt.subplot(3,4,3)
    plt.title('Mean')
    plt.contourf(grid_strike,grid_dip,grid_lamda_mean,cmap=cm.RdYlBu_r,levels=niveles_lamda)
    plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Lamda [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,4)
    plt.title('Standar deviation')
    plt.contourf(grid_strike,grid_dip,grid_lamda_std,cmap=cm.RdYlBu_r,levels=niveles_lamda)
    cbar = plt.colorbar()
    cbar.set_label('Lamda [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,5)
    plt.title('Percentile 10')
    plt.contourf(grid_strike,grid_dip,grid_lamda_p10,cmap=cm.RdYlBu_r,levels=niveles_lamda)
    plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Lamda [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,6)
    plt.title('Percentile 25')
    plt.contourf(grid_strike,grid_dip,grid_lamda_p25,cmap=cm.RdYlBu_r,levels=niveles_lamda)
    cbar = plt.colorbar()
    cbar.set_label('Lamda [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,7)
    plt.title('Percentile 75')
    plt.contourf(grid_strike,grid_dip,grid_lamda_p75,cmap=cm.RdYlBu_r,levels=niveles_lamda)
    cbar = plt.colorbar()
    cbar.set_label('Lamda [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,8)
    plt.title('Percentile 90')
    plt.contourf(grid_strike,grid_dip,grid_lamda_p90,cmap=cm.RdYlBu_r,levels=niveles_lamda)
    cbar = plt.colorbar()
    cbar.set_label('Lamda [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)   
    
    plt.subplot(3,4,9)
    plt.title('Minimum value')
    plt.contourf(grid_strike,grid_dip,grid_lamda_min,cmap=cm.RdYlBu_r,levels=niveles_lamda)
    plt.xlabel('Strike [°]')
    plt.ylabel('Dip [°]')
    cbar = plt.colorbar()
    cbar.set_label('Lamda [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,10)
    plt.title('Percentile 5')
    plt.contourf(grid_strike,grid_dip,grid_lamda_p5,cmap=cm.RdYlBu_r,levels=niveles_lamda)
    plt.xlabel('Strike [°]')
    cbar = plt.colorbar()
    cbar.set_label('Lamda [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,11)
    plt.title('Percentile 95')
    plt.contourf(grid_strike,grid_dip,grid_lamda_p95,cmap=cm.RdYlBu_r,levels=niveles_lamda)
    plt.xlabel('Strike [°]')
    cbar = plt.colorbar()
    cbar.set_label('Lamda [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)        
    
    plt.subplot(3,4,12)
    plt.title('Maximum value')
    plt.contourf(grid_strike,grid_dip,grid_lamda_max,cmap=cm.RdYlBu_r,levels=niveles_lamda)
    plt.xlabel('Strike [°]')
    cbar = plt.colorbar()
    cbar.set_label('Lamda [MPa/Mpa]',fontsize=10)
    plt.grid(linestyle='--', linewidth=1)          
    
    plt.suptitle(name)
    
    if save_pore_pressure_results:
        plt.savefig(name+'_PorePressure.png')
        plt.savefig(name+'_PorePressure.svg')


## Print the time requiered to run the code. 
time_step3 = time.time()-tic
print(str(time_step0)+' seconds at Step 1: Initial variables creation')
print(str(time_step1)[0:-7]+' seconds at Step 2: Randomly principal stress axes selection')
print(str(time_step2)[0:-10]+' seconds at Step 3: Monte carlo simulation')
print(str(time_step3)[0:-10]+' seconds at Step 4: Plotting results')    


## Code End