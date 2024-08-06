#dbs-numbers-generator.py

# from scipy.interpolate import interp1d
# from scipy.interpolate import interpn
# from scipy.signal import savgol_filter

import numpy as np
import time
import tqdm as tqdm
# from tqdm import tqdm
# import matplotlib.pyplot as plt
import os
import sys
# import cmath
# import polars as pl
import dask.array as da
# import pandas as pd
# import numba

############################################################################
############################# initialitiazion   ############################
############################################################################

# power spectrum parameters
dN=0.1
n=1
L=n*251.327
k0 = 1e11 # Mpc^-1 . '''this gives k_peak=1.287e13 Mpc^-1'''
# k0 = 1.7e9 # Mpc^-1 . '''this gives M_peak=1e-5 M_\odot'''

# initial and final k that will be integrated
ki=11
kf=15
kikf=str(ki)+str(kf)
ki=3*10**ki 
kf=3*10**kf


nkk=400 #number of steps. be careful with the number of steps. it can take a lot of time to compute. in my pc 400 steps takes ~1.2 hour. 250 may be a good number, taking like 15-20 mins

kk = np.geomspace(ki, kf, nkk,dtype='float64')
k1=kk
k2=kk

#create array for x
num_points = nkk//2  # Divide by 2 to cover the range from -1 to 1

x_positive = np.linspace(1e-2, 0.9999, num_points)
x = np.concatenate((-x_positive[::-1] , x_positive))
nx=len(x)

# File names
cwd = os.getcwd()
# file name to save bs data

# Navigate to the parent directory
# parent_directory = os.path.dirname(cwd)

# Define the directory where you want to save the file
data_directory = os.path.join(cwd, 'data')

# File name to save bs data
databs_file = f'datadbs-gth-{nkk}-steps-{kikf}-lambda-{n}L0.npy'

# Construct the full path including the directory
full_path = os.path.join(data_directory, databs_file)
full_path = f'C:\ZZZ\Laburos\Codes\\newdata\datadbs-gth-{nkk}-steps-3e{kikf}-lambda-{n}L0.npy'

# def wm(k):
#     if k>k0*L:
#         return np.sqrt( (k/k0)**2. -k/k0 *L)
#     elif k<k0*L:
#         return 1j*np.sqrt( k/k0 *L-(k/k0)**2. )
#     else:
#         return 0

# the definition of w_- \equiv wm is between eqs 3.101 and 3.102
def wm(k):
    conditions = [
        k > k0 * L,
        k < k0 * L
    ]
    
    choices = [
        np.sqrt(  abs((k/k0)**2. - k/k0 * L)  ),
        1j * np.sqrt(  abs(k/k0 * L - (k/k0)**2. ) )
    ]
    
    return np.select(conditions, choices, default=1e-2)

# sin(a+bi)=sin(a)cosh(b)+icos(a)sinh(b)
# sin(1j*20) = np.sinh(20) * 1j

# see eq. 3.104
def sinwm(k):
    a = np.sin(wm(k)*dN)
    b = wm(k)
    return (a/b).real
    # return np.where( np.logical_or(wm(k)==1e-6 , wm(k)==0) , dN*np.cos(wm(k)) , (a/b).real)  
    # return np.where( wm(k)==1e-6, dN, (a/b).real)  

# see eq. 3.102
def GC(k):
    a= ( -1j*(1.+2.*(k/k0)**2.)*k0/k*np.cos(wm(k)*dN).real -2.*(k/k0)**2.*sinwm(k) )*np.cos(k/k0*np.exp(-dN/2.))
    b= ( (2j+1j*(k0/k)**2.-2.*k/k0)*np.cos(wm(k)*dN).real +1j*((2.+(k0/k)**2.)*wm(k)**2.-2j*k/k0)*sinwm(k)  )*np.sin(k/k0*np.exp(-dN/2.))
    return 1j/(8.*k**3.) *(a+b)
    
# see eq. 3.103    
def GS(k):
    a = ( (2j*k/k0*wm(k)**2.+1.+2.*(k/k0)**2.)*sinwm(k) -(1.-2.*(k/k0)**2.-2j*k/k0)*np.cos(wm(k)*dN).real )*np.cos(k/k0*np.exp(-dN/2.))
    b = ( (1.+2.*(k/k0)**2.)*k/k0*sinwm(k) -2j*(k/k0)**2.*np.cos(wm(k)*dN).real )*np.sin(k/k0*np.exp(-dN/2.))
    return 1j/(8.*k**3.) *(a+b)

# see eq. 3.101 
def bs(k1,k2,x):
    a = GC(k1)*GC(k2)*GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*( 1./(wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *np.sin((wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN) +1./(wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *np.sin((wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)  +1./(-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *np.sin((-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN) +1./(-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *np.sin((-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN) )
    b = GS(k1)*GS(k2)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))/(wm(k1)*wm(k2)*wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *( 1./(wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1) +1./(wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1)  +1./(-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1) +1./(-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1)     )
    
    c1 = GC(k1)*GC(k2)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))/wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*(-1./(wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1) +1./(-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1) +1./(-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1) -1./(wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1) )
    d1 = GC(k1)*GS(k2)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))/(wm(k2)*wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *( -1./(wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN) +1./(wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN) +1./(wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN) -1./(-wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((-wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN) )
    c2 = GC(k1)*GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k2)/wm(k2)*(-1./(wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*(np.cos((wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*dN)-1) +1./(-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*(np.cos((-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*dN)-1) +1./(-wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*(np.cos((-wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*dN)-1) -1./(wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*(np.cos((wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*dN)-1) )
    d2 = GC(k1)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k2)/(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*wm(k2)) *( -1./(wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*np.sin((wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*dN) +1./(wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*np.sin((wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*dN) +1./(wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*np.sin((wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*dN) -1./(-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*np.sin((-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*dN) )
    c3 = GC(k2)*GC(k1)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))/wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*(-1./(wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1) +1./(-wm(k2)+wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k2)+wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1) +1./(-wm(k2)-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k2)-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1) -1./(wm(k2)-wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k2)-wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN)-1) )
    d3 = GC(k2)*GS(k1)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))/(wm(k1)*wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *( -1./(wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN) +1./(wm(k2)-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k2)-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN) +1./(wm(k2)+wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k2)+wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN) -1./(-wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((-wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*dN) )
    c4 = GC(k2)*GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k1)/wm(k1)*(-1./(wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*(np.cos((wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*dN)-1) +1./(-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*(np.cos((-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*dN)-1) +1./(-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*(np.cos((-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*dN)-1) -1./(wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*(np.cos((wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*dN)-1) )
    d4 = GC(k2)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k1)/(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*wm(k1)) *( -1./(wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*np.sin((wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*dN) +1./(wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*np.sin((wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*dN) +1./(wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*np.sin((wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*dN) -1./(-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*np.sin((-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*dN) )
    c5 = GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GC(k2)*GS(k1)/wm(k1)*(-1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*(np.cos((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*dN)-1) +1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)-wm(k1))*(np.cos((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)-wm(k1))*dN)-1) +1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)+wm(k1))*(np.cos((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)+wm(k1))*dN)-1) -1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)-wm(k1))*(np.cos((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)-wm(k1))*dN)-1) )
    d5 = GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k2)*GS(k1)/(wm(k2)*wm(k1)) *( -1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*dN) +1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)+wm(k1))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)+wm(k1))*dN) +1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)-wm(k1))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)-wm(k1))*dN) -1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*np.sin((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*dN) )
    c6 = GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GC(k1)*GS(k2)/wm(k2)*(-1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*(np.cos((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*dN)-1) +1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)-wm(k2))*(np.cos((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)-wm(k2))*dN)-1) +1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)+wm(k2))*(np.cos((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)+wm(k2))*dN)-1) -1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)-wm(k2))*(np.cos((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)-wm(k2))*dN)-1) )
    d6 = GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k1)*GS(k2)/(wm(k1)*wm(k2)) *( -1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*dN) +1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)+wm(k2))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)+wm(k2))*dN) +1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)-wm(k2))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)-wm(k2))*dN) -1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*np.sin((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*dN) )

    r = a+b+c1+d1+c2+d2+c3+d3+c4+d4+c5+d5+c6+d6
    # return 3.*r.imag # bs with dimensions
    return 3.*r.imag*(k1*k2)**2.*(k1**2.+k2**2.-2.*k1*k2*x)/(2*np.pi**2.)**2 # dimensionless bs. see eq. 4.74
    # return 3.*r.imag*(k1*k2)**2.*(k1**2.+k2**2.-2.*k1*k2*x) # another convention for the dbs




# initial time
ti = time.time()

# Convert initial time to hh:mm:ss format
initial_time_str = time.strftime('%H:%M:%S', time.localtime(ti))


# Print the initial time
print('Initial time:', initial_time_str)

# Convert numpy arrays to Dask arrays
k1_dask = da.from_array(k1.flatten(), chunks='auto')
k2_dask = da.from_array(k2.flatten(), chunks='auto')
x_dask = da.from_array(x.flatten(), chunks='auto')



# Create 3D meshgrid using Dask
K1, K2, X = da.meshgrid(k1_dask, k2_dask, x_dask, indexing='ij')

# Flatten the meshgrid arrays
K1_1D = da.ravel(K1)
K2_1D = da.ravel(K2)
X_1D = da.ravel(X)

# Apply the bs function to the Dask arrays using map_blocks
bs_results_dask = da.map_blocks(bs, K1_1D, K2_1D, X_1D, dtype=np.float64)


databs_reshaped = da.reshape(bs_results_dask, (nkk, nkk, nkk))

# Compute the results
# databs_dask = databs_reshaped.compute()
databs = databs_reshaped.compute()

tf = time.time()
# Convert initial time to hh:mm:ss format
final_time_str = time.strftime('%H:%M:%S', time.localtime(tf))

# Print the initial time
print('Final time:', final_time_str)
duration = tf - ti
print(f"Computation completed in {duration:.2f} seconds")


##########################################################



##########################################################

# # computacion sin dask

# # Create meshgrid of k1, k2, and x
# K1, K2, X = np.meshgrid(k1, k2, x, indexing='ij')

# # Reshape K1, K2, and X into 1D arrays
# K1_1D = K1.flatten()
# K2_1D = K2.flatten()
# X_1D = X.flatten()

# # Compute databs using vectorized operations
# databs_1D = bs(K1_1D, K2_1D, X_1D)

# # Reshape databs_1D into 3D array
# databs = databs_1D.reshape((nkk, nkk, nx))

##########################################################

# Save the data to a .npy file
np.save(full_path, databs)




# df = pd.DataFrame({
#     'k1': kk,
#     'k2': kk,
#     'x': x,
# })

# # Apply wm(k) function to 'k' column
# df['wm_k1'] = wm(df['k1'])
# df['wm_k2'] = wm(df['k2'])
# # df['wm_k12x'] = wm(df['k1']**2 + df['k2']**2 - 2 * df['k1'] * df['k2'] * df['x'])
# # Now, df will have columns 'k1', 'k2', 'x', and 'wm_k', where 'wm_k' contains the values computed by the wm(k) function based on the conditions specified.
# # Display the DataFrame
# print(df)

'''listo'''

# for i in range(int(nkk/50)):
#     for j in [0, int(nkk/5), int(nkk*2/5), int(nkk*3/5), int(nkk*4/5), int(nkk-1)]:
#         plt.plot(kk, databs[:, i, j], label='k[i]=%.2f, x[j]=%.2f' % (kk[i], x[j]))
#         plt.yscale('symlog')
#         plt.xscale('log')
#         plt.legend()
#         plt.show()


print(f'ki: {ki:.0e}, kf: {kf:.0e}')
print(' ')

# Find the maximum value
max_value = np.max((databs))

# Find the index of the maximum value
max_index = np.unravel_index(np.argmax((databs)), databs.shape)
print("Maximum value:", max_value)
print("Position of maximum value:", max_index)


print('k1_min: {:.2e}'.format(k1[max_index[0]]))
print('k2_min: {:.2e}'.format(k2[max_index[1]]))
print('x_max:', x[max_index[2]], 'degrees:', np.degrees(np.arccos(x[max_index[2]])))

print(' ')

# Find the minimum value
min_value = np.min((databs))

# Find the index of the minimum value
min_index = np.unravel_index(np.argmin((databs)), databs.shape)

print("Minimum value:", min_value)
print("Position of minimum value:", min_index)

print('k1_min: {:.2e}'.format(k1[min_index[0]]))
print('k2_min: {:.2e}'.format(k2[min_index[1]]))
print('x_min:', x[min_index[2]], 'degrees:', np.degrees(np.arccos(x[min_index[2]])))
print(' ')

print('Mean:', np.mean(databs))
print('Standard deviation:', np.std(databs))