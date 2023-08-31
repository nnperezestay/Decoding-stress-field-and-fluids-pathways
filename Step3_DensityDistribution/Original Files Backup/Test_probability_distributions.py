# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:45:47 2022

@author: NicoP
"""

## Loading libreries
import matplotlib.pyplot as plt
import numpy as np
import mplstereonet as stereo
import time

### This test code show how S1, S3 and fhi can be randomly selected, and should
## represent a similar distribution to MIM-results. To run this code, you first
## must run the "DensityDistribution_of_MIM_StressFields.py" code to create
## the files of S1, S3 and fhi distribution.    

###############################################################################
################################### Settings ################################## 
###############################################################################

tic = time.time()

### Test the selection of S1 and S3 base on density distribution write in text files
name='Test_PuyuhuapiLongTerm'

n_test=200 ## amount of test that you want to prove

fig2=plt.figure(1000, figsize=(20.0, 15.0))
ax2=fig2.add_subplot(111,projection='stereonet')

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
    
for i in range(0,n_test):
    
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
selfhi=np.random.choice(len(nfhi),n_test,p=probafhi)

fhi_sel=[]
for seli in selfhi:          
    fhi_sel.append(fhi1[seli])

time_step1 = time.time()-tic
print(str(time_step1)[0:-10]+' seconds')       
print('The random selection of principal stress orientation and stress ratio if finished')  
print('In '+str(time_step1)[0:-10]+' seconds')       
tic=time.time() 

#%% Step 3:
##############################################################################
############################### Plotting #####################################
##############################################################################

for i in range(0,n_test):  

    ### S1, S3 orientations and fhi are defined
    Sigma1=[S1trend_sel[i],S1plunge_sel[i]]    
    Sigma3=[S3trend_sel[i],S3plunge_sel[i]]
    fhi=fhi_sel[i]

    ### Calculate S2 for "i" case 
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

    ## Plot
    ax2.line(Sigma1[1],Sigma1[0],'ro',markersize=5)
    ax2.line(Sigma2[1],Sigma2[0],'go',markersize=5)
    ax2.line(Sigma3[1],Sigma3[0],'bo',markersize=5)
    plt.title('Red=S1, Blue=S3, Green=S2')
    
## End
    