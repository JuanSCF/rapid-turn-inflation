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

############################################################################
############################# initialitiazion   ############################
############################################################################

# power spectrum parameters
deltaN=0.1
n=1
L=n*251.327
k00 = 1e11 # Mpc^-1 . '''this gives k_peak=1.287e13 Mpc^-1'''

# initial and final k that will be integrated
ki=8
kf=14
kikf=str(ki)+str(kf)
ki=1*10**ki 
kf=1*10**kf

nkk=100 #number of steps
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
databs_file = f'databs-{nkk}-steps-{spacing}-spacing-{kikf}-lambda-{n}L0.npy'

# Construct the full path including the directory
full_path = os.path.join(data_directory, databs_file)


def wm(k):
    if k>k00*L:
        return np.sqrt( (k/k00)**2. -k/k00 *L)
    elif k<k00*L:
        return 1j*np.sqrt( k/k00 *L-(k/k00)**2. )
    else:
        return 0

def sinwm(k):
    a = np.sin(wm(k)*deltaN)/wm(k)
    return a.real

def GC(k):
    a= ( -1j*(1.+2.*(k/k00)**2.)*k00/k*np.cos(wm(k)*deltaN) -2.*(k/k00)**2.*sinwm(k) )*np.cos(k/k00*np.exp(-deltaN/2.))
    b= ( (2j+1j*(k00/k)**2.-2.*k/k00)*np.cos(wm(k)*deltaN) +1j*((2.+(k00/k)**2.)*wm(k)**2.-2j*k/k00)*sinwm(k)  )*np.sin(k/k00*np.exp(-deltaN/2.))
    return 1j/(8.*k**3.) *(a+b)
    
def GS(k):
    a = ( (2j*k/k00*wm(k)**2.+1.+2.*(k/k00)**2.)*sinwm(k) -(1.-2.*(k/k00)**2.-2j*k/k00)*np.cos(wm(k)*deltaN) )*np.cos(k/k00*np.exp(-deltaN/2.))
    b = ( (1.+2.*(k/k00)**2.)*k/k00*sinwm(k) -2j*(k/k00)**2.*np.cos(wm(k)*deltaN) )*np.sin(k/k00*np.exp(-deltaN/2.))
    return 1j/(8.*k**3.) *(a+b)

def bs(k1,k2,x):
    if np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )<L*k00 and k1<k00*L and k2<k00*L:
        a = GC(k1)*GC(k2)*GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*( 1./(wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *np.sin((wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN) +1./(wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *np.sin((wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)  +1./(-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *np.sin((-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN) +1./(-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *np.sin((-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN) )
        b = GS(k1)*GS(k2)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))/(wm(k1)*wm(k2)*wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *( 1./(wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1) +1./(wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1)  +1./(-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1) +1./(-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1)     )
        
        c1 = GC(k1)*GC(k2)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))/wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*(-1./(wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1) +1./(-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1) +1./(-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1) -1./(wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k1)-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1) )
        d1 = GC(k1)*GS(k2)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))/(wm(k2)*wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *( -1./(wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN) +1./(wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k1)-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN) +1./(wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k1)+wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN) -1./(-wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((-wm(k1)+wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN) )
        c2 = GC(k1)*GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k2)/wm(k2)*(-1./(wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*(np.cos((wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*deltaN)-1) +1./(-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*(np.cos((-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*deltaN)-1) +1./(-wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*(np.cos((-wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*deltaN)-1) -1./(wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*(np.cos((wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*deltaN)-1) )
        d2 = GC(k1)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k2)/(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*wm(k2)) *( -1./(wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*np.sin((wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*deltaN) +1./(wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*np.sin((wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*deltaN) +1./(wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*np.sin((wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2))*deltaN) -1./(-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*np.sin((-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2))*deltaN) )
        c3 = GC(k2)*GC(k1)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))/wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*(-1./(wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1) +1./(-wm(k2)+wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k2)+wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1) +1./(-wm(k2)-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((-wm(k2)-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1) -1./(wm(k2)-wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*(np.cos((wm(k2)-wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN)-1) )
        d3 = GC(k2)*GS(k1)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))/(wm(k1)*wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))) *( -1./(wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN) +1./(wm(k2)-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k2)-wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN) +1./(wm(k2)+wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((wm(k2)+wm(k1)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN) -1./(-wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*np.sin((-wm(k2)+wm(k1)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x )))*deltaN) )
        c4 = GC(k2)*GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k1)/wm(k1)*(-1./(wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*(np.cos((wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*deltaN)-1) +1./(-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*(np.cos((-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*deltaN)-1) +1./(-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*(np.cos((-wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*deltaN)-1) -1./(wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*(np.cos((wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*deltaN)-1) )
        d4 = GC(k2)*GS(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k1)/(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*wm(k1)) *( -1./(wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*np.sin((wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*deltaN) +1./(wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*np.sin((wm(k2)-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*deltaN) +1./(wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*np.sin((wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1))*deltaN) -1./(-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*np.sin((-wm(k2)+wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1))*deltaN) )
        c5 = GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GC(k2)*GS(k1)/wm(k1)*(-1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*(np.cos((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*deltaN)-1) +1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)-wm(k1))*(np.cos((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)-wm(k1))*deltaN)-1) +1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)+wm(k1))*(np.cos((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)+wm(k1))*deltaN)-1) -1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)-wm(k1))*(np.cos((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)-wm(k1))*deltaN)-1) )
        d5 = GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k2)*GS(k1)/(wm(k2)*wm(k1)) *( -1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*deltaN) +1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)+wm(k1))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k2)+wm(k1))*deltaN) +1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)-wm(k1))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)-wm(k1))*deltaN) -1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*np.sin((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k2)+wm(k1))*deltaN) )
        c6 = GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GC(k1)*GS(k2)/wm(k2)*(-1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*(np.cos((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*deltaN)-1) +1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)-wm(k2))*(np.cos((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)-wm(k2))*deltaN)-1) +1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)+wm(k2))*(np.cos((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)+wm(k2))*deltaN)-1) -1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)-wm(k2))*(np.cos((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)-wm(k2))*deltaN)-1) )
        d6 = GC(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))*GS(k1)*GS(k2)/(wm(k1)*wm(k2)) *( -1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*deltaN) +1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)+wm(k2))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))-wm(k1)+wm(k2))*deltaN) +1./(wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)-wm(k2))*np.sin((wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)-wm(k2))*deltaN) -1./(-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*np.sin((-wm(np.sqrt( k1**2.+k2**2.-2.*k1*k2*x ))+wm(k1)+wm(k2))*deltaN) )

        r = a+b+c1+d1+c2+d2+c3+d3+c4+d4+c5+d5+c6+d6
        return 3.*r.imag
    else:
        return 0.

def bs_vec(k1, k2, x):
    # k1=k1/k00
    # k2=k2/k00
    # Calculate distance term
    dist = np.sqrt(k1[:, None, None]**2 + k2[None, :, None]**2 - 2*k1[:, None, None]*k2[None, :, None]*x[None, None, :])
    
    # Define conditions
    condition = (dist < L) & (k1[:, None, None] < L) & (k2[None, :, None] < L)
    
    # Initialize result arrays
    bs_results = np.zeros_like(dist, dtype=np.float64)
    # bs_results = np.zeros_like(dist, dtype=np.complex128)
    # bs_results_imag = np.zeros_like(dist, dtype=np.complex128)
    
    # Apply conditions using np.where and compute results
    bs_results = np.where(condition, bs(k1[:, None, None], k2[None, :, None], x[None, None, :]), bs_results)
    
    # bs_results_imag = np.where(condition, bs(k1[:, None, None]/k00, k2[None, :, None]/k00, x[None, None, :]), bs_results_imag)
    
    return bs_results #, bs_results_imag


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

# Apply the bs_vec function to the Dask arrays
bs_results_dask = da.map_blocks(bs_vec, k1_dask, k2_dask, x_dask)

# Compute the results
databs = bs_results_dask.compute()

tf = time.time()
# Convert initial time to hh:mm:ss format
final_time_str = time.strftime('%H:%M:%S', time.localtime(tf))

# Print the initial time
print('Final time:', final_time_str)
duration = tf - ti
print(f"Computation completed in {duration:.2f} seconds")


# approach number #
# the next approach below uses too much memory, terabyte order
# # Generate meshgrid of indices
# i, j, k = np.meshgrid(range(nkk), range(nkk), range(nx), indexing='ij')

# # Compute results using bs_vec function
# bs_results = bs_vec(k1[i]/k00, k2[j]/k00, x[k])



# # Vectorized calculation of bs_results and bs_results_imag
# for i in tqdm.tqdm(range(nkk)):
#     k1_normalized = k1[i] / k00
#     k2_normalized = k2 / k00
#     results = bs(np.expand_dims(k1_normalized, axis=1), np.expand_dims(k2_normalized, axis=0), x)
#     bs_results[i] = np.real(results)
#     # bs_results_imag[i] = np.imag(results)

##########################################################
    

##########################################################

# Save the data to a .npy file
# np.save(full_path, databs)


# print(time.gmtime())
deltat=tf-ti
print('computation time: ',deltat)
'''listo'''


for i in [0, 99, 199, 299, 399, 499]:
    for j in [0, 49, 177, 250, 380, 430]:
        plt.plot(kk, databs[:, i, j], label='k[i]=%.2f, x[j]=%.2f' % (kk[i], x[j]))
        plt.yscale('symlog')
        plt.xscale('log')
        plt.legend()
        plt.show()
# plt.plot(kk,databs[0,9,:])
# plt.show()

# plt.plot(kk,databs[0,9,:])

# plt.show()

i=250
j=-1     
plt.plot(kk, databs[:, i, j], label='k2[i]=%.2f, x[j]=%.2f' % (kk[i], x[j]))
plt.plot(kk, abs(databs[:, i, j]), label='abs. k2[i]=%.2f, x[j]=%.2f' % (kk[i], x[j]))
plt.yscale('symlog')
plt.xscale('log')
plt.legend()
plt.show()