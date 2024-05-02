#bs-numbers-generator.py

# from scipy.interpolate import interp1d
# from scipy.interpolate import interpn
# from scipy.signal import savgol_filter

import numpy as np
import time
import tqdm as tqdm
# from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import cmath
# import polars as pl
import dask.array as da
import pandas as pd
# import numba

############################################################################
############################# initialitiazion   ############################
############################################################################

# power spectrum parameters
dN=0.1
n=1
L=n*251.327
k0 = 1e11 # Mpc^-1 . '''this gives k_peak=1.287e13 Mpc^-1'''

# initial and final k that will be integrated
ki=8
kf=14
kikf=str(ki)+str(kf)
ki=1*10**ki 
kf=1*10**kf

nkk=400 #number of steps
spacing='geometric' # 'geometric' or 'linear'



kk = np.geomspace(ki, kf, nkk,dtype='float64')
# kk = np.linspace(ki, kf, nkk,dtype='float64', endpoint=False)
k1=kk
k2=kk
'''
probar usando linspace para todos los calculos
'''
#create array for x
num_points = nkk//2  # Divide by 2 to cover the range from -1 to 1

if spacing=='geometric':
    x_positive = np.geomspace(1e-6, 0.9999, num_points)
elif spacing=='linear':
    x_positive = np.linspace(1e-6, 0.9999, num_points)
# # Create the symmetric version covering the range from -1 to 1
# x = np.concatenate((-x_positive[::-1], [0], x_positive))
x = np.concatenate((-x_positive[::-1] , x_positive))

nx=len(x)

# File names
cwd = os.getcwd()
# file name to save bs data
# databs_file = 'full-x-'+str(nkk)+'-steps-'+spacing+'spacing-'+ str(kikf)+'.npy'
# databs_file = cwd+'\\data\\databs-'+databs_file

# Navigate to the parent directory
# parent_directory = os.path.dirname(cwd)

# Define the directory where you want to save the file
data_directory = os.path.join(cwd, 'data')

# File name to save bs data
databs_file = f'databs-gth-{nkk}-steps-{spacing}-spacing-{kikf}-lambda-{n}L0.npy'

# Construct the full path including the directory
full_path = os.path.join(data_directory, databs_file)


# def wm(k):
#     if k>k0*L:
#         return np.sqrt( (k/k0)**2. -k/k0 *L)
#     elif k<k0*L:
#         return 1j*np.sqrt( k/k0 *L-(k/k0)**2. )
#     else:
#         return 0

def wm(k):
    conditions = [
        k > k0 * L,
        k < k0 * L
    ]
    
    choices = [
        np.sqrt(  abs((k/k0)**2. - k/k0 * L)  ),
        1j * np.sqrt(  abs(k/k0 * L - (k/k0)**2. ) )
    ]
    
    return np.select(conditions, choices, default=0)

def sinwm(k):
    # sin(a+bi)=sin(a)cosh(b)+icos(a)sinh(b)
    # sin(1j×20) = np.sinh(20) * 1j
    a = np.sin(wm(k)*dN)
    b = wm(k)
    return (a.real + a.imag)/(b.real + b.imag)

def GC(k):
    a= ( -1j*(1.+2.*(k/k0)**2.)*k0/k*np.cos(wm(k)*dN).real -2.*(k/k0)**2.*sinwm(k) )*np.cos(k/k0*np.exp(-dN/2.))
    b= ( (2j+1j*(k0/k)**2.-2.*k/k0)*np.cos(wm(k)*dN).real +1j*((2.+(k0/k)**2.)*wm(k)**2.-2j*k/k0)*sinwm(k)  )*np.sin(k/k0*np.exp(-dN/2.))
    return 1j/(8.*k**3.) *(a+b)
    
def GS(k):
    a = ( (2j*k/k0*wm(k)**2.+1.+2.*(k/k0)**2.)*sinwm(k) -(1.-2.*(k/k0)**2.-2j*k/k0)*np.cos(wm(k)*dN).real )*np.cos(k/k0*np.exp(-dN/2.))
    b = ( (1.+2.*(k/k0)**2.)*k/k0*sinwm(k) -2j*(k/k0)**2.*np.cos(wm(k)*dN).real )*np.sin(k/k0*np.exp(-dN/2.))
    return 1j/(8.*k**3.) *(a+b)

# @numba.jit
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
    return 3.*r.imag
    # return 3.*(k1*k2)**2.*(k1**2.+k2**2.-2.*k1*k2*x)*r




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
# np.save(full_path, databs)
np.save(f'C:\ZZZ\Laburos\Codes\\newdata\databs-gth-{nkk}-steps-geometric-spacing-{kikf}-lambda-1L0.npy', databs)


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
