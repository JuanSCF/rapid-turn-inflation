#bs-numbers-generator.py

# from scipy.interpolate import interp1d
# from scipy.interpolate import interpn
# from scipy.signal import savgol_filter

import numpy as np
import tqdm as tqdm
# from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import time
# import polars as pl
import dask.array as da
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
############################################################################
############################# initialitiazion   ############################
############################################################################

# power spectrum parameters
deltaN=0.1
n=4
L0=251.327
L=n*L0

k00 = 1e11 # Mpc^-1 . '''this gives k_peak=1.287e13 Mpc^-1'''

# initial and final k that will be integrated
ki=10
kf=14
kikf=str(ki)+str(kf)
ki=1*10**ki 
kf=1*10**kf

Wf='Wthtf'
nkk=300 #number of steps for bs(k1,k2,x) and xi3(MH(k))
spacing='geometric' # 'geometric' or 'linear'

size=4000


kk = np.geomspace(ki, kf, nkk,dtype='float64', endpoint=False)
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

# Define the directory where you want to save the file

# File name to save bs data
# databs_file = f'full-x-{nkk}-steps-{spacing}-spacing-{kikf}.npy'
databs_file = f'databs-{nkk}-steps-{spacing}-spacing-{kikf}-lambda-{L}.npy'

# Construct the full path including the directory
databs_file = os.path.join(cwd, 'data\\'+databs_file)

xi3_file=f'xi3-{Wf}-{nkk}-steps-{kikf}-{spacing}-spacing-{n}L0'
xi3_data = os.path.join(cwd, 'data\\'+xi3_file)
xi3_fig = os.path.join(cwd, 'figs\\'+xi3_file)

databs=np.load(databs_file)




#####################################################################################
###################### gaussian data extraction #####################################
#####################################################################################

gamma=0.36
if Wf=='Wg':
  deltac = 0.18
  C=1.44
else:
  deltac = 0.5
  C=4.
OmegaCDM=0.264

cwd = os.getcwd()
# fgaussian_data_file = os.path.join(data_directory, 'gaussian-data-C'+str(C)+'-deltac'+str(deltac)+'-'+Wf+'.npz')
# fgaussian_data_file = os.path.join(cwd, 'gaussian-data-C4.0-deltac0.5-Wthtf.npz')
fgaussian_data_file = os.path.join(cwd, f'data\\gaussian-data-C{C}-deltac{deltac}-{Wf}.npz')
fgaussian_data = np.load(fgaussian_data_file)

kz=fgaussian_data['kz']
# Mmax=Mz[xi3maxindex]
sigmaR2=fgaussian_data['sigmaR2']


csrad=np.sqrt(1./3.)
Omegam = 0.315 #???
Meq=(2.8)*10**17. #solar masses
keq=0.01*(Omegam/0.31) #Mpc^-1



MH = Meq*(keq/kz)**2
sigmapeak=np.amax(sigmaR2)
mhpeak=np.argmin(np.abs(sigmaR2-sigmapeak))
kzpeak=kz[int(mhpeak)]
MHp=MH[int(mhpeak)]

mzpeak=np.argmin(np.abs(sigmaR2-sigmapeak))
Mzp=MH[int(mzpeak)]
#####################################################################################

##########################################################
    

##########################################################










# Vectorized 1D integration
def Intarray_vec(f, array):
    # Calculate differences between consecutive elements of array
    diff_array = np.diff(array)
    
    # Calculate the average of f for each interval
    avg_f = 0.5 * (f[:-1] + f[1:])
    
    # Calculate the product of differences and average f values
    product = diff_array * avg_f
    
    # Sum up the products
    integral = np.sum(product)
    
    return integral

# Vectorized 3D integration
def Intarray3D_vec(f, array1, array2, array3):

    # Calculate differentials for each dimension
    diff1 = np.diff(array1)
    diff2 = np.diff(array2)
    diff3 = np.diff(array3)

    # Calculate volumes for each cell
    dV = np.multiply.outer(diff1, diff2)
    dV = np.multiply.outer(dV, diff3)

    # Calculate values for each corner of each cell
    f_corners = f[:-1, :-1, :-1] + f[:-1, :-1, 1:] + f[:-1, 1:, :-1] + f[:-1, 1:, 1:] + \
                f[1:, :-1, :-1] + f[1:, :-1, 1:] + f[1:, 1:, :-1] + f[1:, 1:, 1:]

    # Multiply values with coefficients and sum them up
    integral = 0.125 * np.sum(dV * f_corners)

    return integral













###################################################################

def kofMH(M):
    return keq*(Meq/M)**0.5
def MHofk(k):
    return (keq/k)**2.*Meq



'''
Mucho
ojo
aqui
!!
'''



kzsmall = np.zeros(nkk)
varsmall = np.zeros(nkk)

# creo que esto está malo, revisar
# indices = np.argmin(np.abs(kk[:, np.newaxis] - kz), axis=1)
# kzsmall = kz[indices]
# varsmall = sigmaR2[indices]
for i in range(nkk):
    index=np.argmin(np.abs(kk[i]-kz)) ##  !!!
    kzsmall[i] = kz[index]
    varsmall[i] = sigmaR2[index]




MHsmall = MHofk(kzsmall)
LMH=np.log(MHsmall)

Mz = np.geomspace(MHsmall[-1], 10**(-1)*MHsmall[0], size) 
LMz = np.log(Mz)


########################################################################
# Smoothing function as a function of the collapsing scale q=RH^-1
########################################################################
    # Gaussian
if Wf=='Wg':
    def W(k,q):    
        return np.exp(-0.5*(k/q)**2.)   
    # return np.exp(-(k/keq)**2*(MH/Meq))
    # top-hat
elif Wf=='Wth':
    def W(k,q):        
        a=3.*(np.sin(k/q)-k/q*np.cos(k/q))/(k/q)**3. 
        return a
#
    # tophat+transfer
elif Wf=='Wthtf':
    def W(k,q):
        a=3.*(np.sin(k/q)-k/q*np.cos(k/q))/(k/q)**3. 
        b=3.*(np.sin(csrad*k/q)-csrad*k/q*np.cos(csrad*k/q))/(csrad*k/q)**3. 
        return a*b
#
    # sharp-k
    # return a*b
    # return np.heaviside(1./q-k,1.)  # este anula a xi3


def Mcal(k,q):
    m=4./9. *(k/q)**2.  *W(k,q)
    return m
########################################################################
########################################################################


########## xi3 calculation #########

# perturbatiby parameter g H^2 Delta N
gmax=1

# vectorized xi3
def int_xi3(m1,m2,wx):
    a = k00**2*0.5*m1[:, None, None]*m2[None, :, None]*wx*databs
    return a

def integrandxi3(Mh,k1,k2,x):
    k1=k1/k00
    k2=k2/k00
    q=kofMH(Mh)/k00
    m1=Mcal(k1,q)
    m2=Mcal(k2,q)
    k12x = np.sqrt(k1[:, None, None]**2 + k2[None, :, None]**2 - 2*k1[:, None, None]*k2[None, :, None]*x[None, None, :])
    wx=4./9.*q**(-2)*W(k12x,q)

    condition = (k12x < L) & (k1[:, None, None] < L) & (k2[None, :, None] < L)
    a=np.zeros_like(databs)
    a = np.where(condition, int_xi3(m1,m2,wx), a)
    return a

ti= time.time()
xi3 = np.zeros(len(MHsmall))

print('xi3 calc')
initial_time_str = time.strftime('%H:%M:%S', time.localtime(ti))
print('Initial time:', initial_time_str)
for i in tqdm.tqdm(range(len(MHsmall))):
    xi3[i] = Intarray3D_vec(integrandxi3(MHsmall[i], k1, k2, x), k1, k2, x)

# def xi3calc():
#     for i in tqdm.tqdm(range(len(MHsmall))):
#         xi3[i] = Intarray3D_vec(integrandxi3(MHsmall[i], k1, k2, x), [k1, k2, x])

tf = time.time()
duration = tf - ti

t_xi3_MH=duration


xi3max=max(xi3)
xi3maxindex=np.argmax(xi3)
kmax=kk[xi3maxindex]
MHmax=MHsmall[xi3maxindex]


# 
'''
when computing f(M),i think i need to normalize first before summing up the non-gaussian part to the gaussian integrand
'''
def intfdeM(M,MHsmall,varsmall,xi3):
    mu = (M/(C*MHsmall))
    Integrand_f=-2/(OmegaCDM)/(np.sqrt(np.pi*2*varsmall))*np.exp(-(mu**(1./gamma)+deltac)**2/(2*varsmall))*M/MHsmall*(1./gamma)*(M/(C*MHsmall))**(1./gamma)*np.sqrt(Meq/MHsmall)
    # f= Intarray_vec(Integrand1, LMH) # ojo: tengo abs() aca!

    deltaM=(mu**(1./gamma)+deltac)**2
#   deltaM=deltac
    Integrand_ngcont=Integrand_f*(1./6.)*xi3*(deltaM**3/varsmall**3-3*deltaM/varsmall**2)
    Integrand_ftot=Integrand_f*(1+(1./6.)*xi3*(deltaM**3/varsmall**3-3*deltaM/varsmall**2) )
    # f2=  Intarray_vec(Integrand2, LMH)  # ojo: tengo abs() aca!
    # fng= Intarray_vec(Integrand4, LMH)  # ojo: tengo abs() aca!

    return Integrand_f, Integrand_ngcont, Integrand_ftot

ti= time.time()
print('f(M) calc')
initial_time_str = time.strftime('%H:%M:%S', time.localtime(ti))
print('Initial time:', initial_time_str)

f=np.zeros(size)
fng=np.zeros(size)
f2=np.zeros(size)
for i in tqdm.tqdm(range(0, len(Mz))):
    a,b,c=intfdeM(Mz[i],MHsmall,varsmall,xi3)
    f[i] = Intarray_vec( a, LMH)
    f2[i] = Intarray_vec( b, LMH)
    fng[i] = Intarray_vec( c, LMH)

# def fdeMcalc():
#     f=np.zeros(size)
#     fng=np.zeros(size)
#     f2=np.zeros(size)
#     for i in tqdm.tqdm(range(0, len(Mz))):
#         a,b,c=intfdeM(Mz[i],MHsmall,varsmall,xi3)
#         f[i] = Intarray_vec( a, LMH)
#         f2[i] = Intarray_vec( b, LMH)
#         fng[i] = Intarray_vec( c, LMH)


tf = time.time()
duration = tf - ti

t_f_ng=duration


print('xi3 duration:', t_xi3_MH)
print('f_ng duration:', t_f_ng)

def compare_arrays(arr1, arr2, tolerance):
    comparison = np.abs(arr1 - arr2) <= tolerance
    return np.all(comparison)

compare_arrays(f2, fng, 1e-2)


np.savez( xi3_data, xi3=xi3,f=f, f2=f2, fng=fng, t_xi3_MH=t_xi3_MH)

'''listo'''

f=abs(f)
fng=abs(fng)

LMz=np.log(Mz)
f_pbh = Intarray_vec(f,LMz)
f_pbh_ng = Intarray_vec(fng,LMz)

fpeak=np.amax(abs(f))
mpeak=np.argmin(np.abs(abs(f)-fpeak)) 
Mp=Mz[int(mpeak)]

fngpeak=np.amax(abs(fng))
mngpeak=np.argmin(np.abs(fng-fngpeak))
Mpng=Mz[int(mngpeak)]





print('xi3 duration:', t_xi3_MH)
print('f_ng duration:', t_f_ng)

plt.loglog(Mz,f, 'o',label='f')
plt.loglog(Mz,fng, 'o',label='f_ng')
plt.axvline(x=Mp)
plt.axvline(x=Mpng, color='orange')
plt.legend()
plt.savefig(xi3_fig+'.svg')
plt.savefig(xi3_fig+'.png')
plt.show()


# # plt.loglog(Mz,f, 'o',label='f')
# plt.loglog(Mz,f/f_pbh)
# # plt.loglog(Mz,fng, 'o',label='f_ng')
# plt.axvline(x=Mp)
# plt.loglog(Mz,fng/f_pbh_ng, color='orange')
# plt.axvline(x=Mpng, color='orange')
# plt.legend()
# # plt.savefig(xi3_fig+'.svg')
# plt.show()


plt.loglog(Mz,f/f_pbh, 'o',label='f')
plt.loglog(Mz,fng/f_pbh_ng, 'o',label='f_ng')
plt.legend()
plt.axvline(x=Mp)
plt.axvline(x=Mpng, color='orange')
plt.savefig(xi3_fig+'-norm.svg')
plt.savefig(xi3_fig+'-norm.png')
plt.show()



'''
agregar codigo que grabe la figura con nombre automatizado
'''