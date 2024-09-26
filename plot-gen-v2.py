
import os, sys
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# from scipy.interpolate import interpn
# from scipy.signal import savgol_filter
# import dask.array as da

############################################################################
############################# initialitiazion   ############################
############################################################################

# power spectrum parameters
deltaN=0.1

# L = 251.327
# L = 233.158 # -> mono: f_peak = 0.997 fpbh = 33.47%    d = x*np.pi
# L = 233.716 # -> ex: fpeak = 0.9983 f_pbh = 89.12%    d = x*np.pi

##################  Wthtf  ######################
L = 233.548 # fpbh_ex = 56.2174%, fpbh_mono = 99.8%, fpeak_ex = 0.6304, fpeak_mono = 2.927, max ps = 1.61e-2
# L = 233.758 # fpbh_ex = 99.7919%, fpbh_mono = 176.136%, fpeak_ex = 1.118, fpeak_mono = 5.123, max ps = 1.62e-2
#################################################

###################  Wg4  #######################
# L = 226.206 ##### # fpbh mono 99.752%, fpbh ex 41.6659%
# L = 226.518 ##### #fpbh mono 236.731%, fpbh ex 99.7571%
#################################################

k0 = 1e11 # Mpc^-1 . '''this gives k_peak=1.287e13 Mpc^-1'''

# initial and final k that will be integrated
ki = 11
kf = 15
kp = k0*L/2.

kikf = str(ki)+str(kf)
ki = 3*10**ki 
kf = 3*10**kf
print(f'ki: {ki:.0e}, kf: {kf:.1e}, kp: {kp:.0e}') #, Mp: {MHofk(kp):.2e}')

Wf = 'Wthtf' # Wg, Wg4, Wth, Wthtf. Wg4tf little smaller than Wg4 alone

nkk = 350 #number of steps
size = 4000


kk = np.geomspace(ki, kf, nkk, dtype='float64')
# kk = np.linspace(ki, kf, nkk,dtype='float64', endpoint=False)
k1=kk
k2=kk
'''
probar usando linspace para todos los calculos
'''
#create array for x
num_points = nkk//2  # Divide by 2 to cover the range from -1 to 1
x_positive = np.linspace(1e-2, 0.9999, num_points)
# # Create the symmetric version covering the range from -1 to 1
# x = np.concatenate((-x_positive[::-1], [0], x_positive))
x = np.concatenate((-x_positive[::-1] , x_positive))

nx=len(x)

####################################################################
############################ File names ############################
####################################################################

gamma=0.36
C = 4.
OmegaCDM = 0.264



# File names
cwd = os.getcwd()

# Define the directory where you want to save the file
data_directory = os.path.join(cwd, 'data')

# File name to save bs data
# databs_file = f'databs-gth-{nkk}-steps-{kikf}-lambda-{n}L0.npy'

# Construct the full path including the directory
# databs_file = os.path.join(data_directory, databs_file)
# databs=np.load(databs_file)
# databs = np.load(f'C:\ZZZ\Laburos\Codes\\newdata\databs-gth-{nkk}-steps-{kikf}-lambda-{n}L0.npy')

# xi3_file=f'xi3-gth-{Wf}-{nkk}-steps-{kikf}-lambda-{n}L0.npy'
xi3_file=f'xi3d-gth-{Wf}-{nkk}-steps-3e{kikf}-L-{L}.npy'
xi3_file = os.path.join(data_directory, xi3_file)
xi3 = np.load(xi3_file)



############################################################################
############################# initialitiazion   ############################
############################################################################

Omegam = 0.315 #???
Meq=(2.8)*10**17. #solar masses
keq=0.01*(Omegam/0.31) #Mpc^-1
keq = 0.01


def kofMH(M):
    return keq*(Meq/M)**0.5
def MHofk(k):
    return (keq/k)**2.*Meq

def kofMH2(M):
    return 1e6*(17/M)**0.5
def MHofk2(k):
    return (1e6/k)**2.*17


########################################################################
# Smoothing function as a function of the collapsing scale q=R_H^-1
########################################################################

# Gaussian4
if Wf=='Wg4':
    deltac = 0.18
    def W(k,q):    
        return np.exp(-0.25*(k/q)**2.)     
# tophat+transfer
elif Wf=='Wthtf':
    deltac = 0.41
    csrad=np.sqrt(1./3.)
    def W(k,q):
        a=3.*(np.sin(k/q)-k/q*np.cos(k/q))/(k/q)**3. 
        b=3.*(np.sin(csrad*k/q)-csrad*k/q*np.cos(csrad*k/q))/(csrad*k/q)**3. 
        return a*b 
# Gaussian4+transfer
elif Wf=='Wg4tf':
    deltac = 0.18
    def W(k,q):    
        a = np.exp(-0.25*(k/q)**2.) 
        b = 3.*(np.sin(csrad*k/q)-csrad*k/q*np.cos(csrad*k/q))/(csrad*k/q)**3. 
        return a*b  
#Gaussian
elif Wf=='Wg':
    deltac = 0.18
    def W(k,q):    
        return np.exp(-0.5*(k/q)**2.)   
    # return np.exp(-(k/keq)**2*(MH/Meq))
# top-hat
elif Wf=='Wth':
    deltac = 0.41
    def W(k,q):        
        a=3.*(np.sin(k/q)-k/q*np.cos(k/q))/(k/q)**3. 
        return a


print(f'Wf: {Wf}, deltac: {deltac}')
print(' ')
    

# fgaussian_data_file = os.path.join(cwd, f'data\\gaussian-data-{Wf}.npz')#-233.npz')#
# np.savez(fgaussian_data_file, kz=kz, sigmaR2=sigmaR2, kpz=k0, pz=PfkE, fmono=fmono, fex=f)
fgaussian_data_file = os.path.join(cwd, f'data\\gaussian-data-{Wf}-L-{L}.npz')
fgaussian_data = np.load(fgaussian_data_file)

kz=fgaussian_data['kz']
sigmaR2=fgaussian_data['sigmaR2']
MH = MHofk(kz)
LMH = np.log(MH)
var = sigmaR2
kzsmall =np.zeros(nkk)
varsmall =np.zeros(nkk)


for i in range(nkk):
    index=np.argmin(np.abs(kk[i]-kz)) ##  !!!
    kzsmall[i]=kz[index]
    varsmall[i]=sigmaR2[index]

# faster than what is above?
# indices = np.argmin(np.abs(kk[:, np.newaxis] - kz), axis=1)
# kzsmall = kz[indices]
# varsmall = sigmaR2[indices]

MHsmall = MHofk(kzsmall)
Lkzsmall = np.log(kzsmall)
LMHsmall = np.log(MHsmall)

Mz = np.geomspace(MHsmall[-1], MHsmall[0], size) 

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


########################################################################
########################################################################


# plt.plot(kzsmall, xi3,'o')
# plt.plot(kzsmall, xi3)
# plt.xscale('log')
# plt.yscale('symlog')
# plt.axhline(y=0, color='k', linestyle='--')
# plt.title(r'$\xi_3$, 'f'{Wf}')
# plt.xlabel(r'k [Mpc$^{-1}$]')
# plt.show()


# These lines are to extend the xi3 function to the whole range of MH
# MHsmall=MHsmall[:309]
# xi3=xi3[:309]
# varsmall=varsmall[:309]
# LMHsmall=LMHsmall[:309]
xi3i = interp1d(MHsmall, xi3,bounds_error=False, kind='cubic',fill_value="extrapolate")

xi3E = np.zeros(len(MH))
xi3fE = np.zeros(len(MH))
# for i in range(0,len(MH)):
#     if MH[i]>MHsmall[0]:
#         xi3E[i] = 0
#     else: 
#         xi3E[i]=xi3i(MH[i])
for i in range(0, len(MH)):
    if MHsmall[-1]<MH[i] and MH[i]<MHsmall[0]:
        xi3E[i]=xi3i(MH[i])
        # xi3fE[i]=xi3fi(MH[i])
    # else: 
    #     xi3E[i] = 0
    elif MH[i] <= MHsmall[-1]:
        xi3E[i] = xi3[-1]
    elif MHsmall[0] <= MH[i]:
        xi3E[i] = xi3[0]


# plt.plot(MH, xi3E,'o')
# plt.plot(MH, xi3E)
# plt.xscale('log')
# plt.yscale('symlog')
# plt.axhline(y=0, color='k', linestyle='--')
# # plt.xlim(2.5e-16, 2.5e-8)
# plt.yticks(list(plt.yticks()[0]) + [0])
# # plt.legend()
# plt.title(r'$\xi_3$, 'f'{Wf}')
# plt.xlabel(r'$M_H/M_\odot$')
# plt.show()


# lets compute a g value for every xi3 value
# gvec = 6.*varsmall**3/(deltac**3.) /abs(xi3)
# gvec = 6.*var**3/(deltac**3.) /abs(xi3E)

# maximum g value allowed according to the perturbation condition
# deltac = 1
gcrits = 6*np.amax(varsmall)**3/abs(xi3[np.argmax(varsmall)])/(deltac**3)
gcrit = 6*np.amax(var)**3/abs(xi3E[np.argmax(var)])/(deltac**3)
print(f'gcrit = {gcrit:.4e}, wf = {Wf}, deltac = {deltac}')

# computing g with the k that peaks the PS
kpz=fgaussian_data['kpz']
pz=fgaussian_data['pz']
kpzmax = kpz[np.argmax(pz)]
Mpzmax = MHofk(kpzmax)
ipz = np.argmin(abs(kpzmax-kz))

gcritpz = 6*var[ipz]**3/abs(xi3E[ipz])/(deltac**3)
print(f'gcrit pz = {gcritpz:.4e}')

varIk = interp1d(kz, var,bounds_error=False, kind='cubic',fill_value="extrapolate")
xi3Ik = interp1d(kzsmall, xi3,bounds_error=False, kind='cubic',fill_value="extrapolate")
gcritpzI = 6*varIk(kpzmax)**3/abs(xi3Ik(kpzmax))/(deltac**3)
print(f'gcrit pz Interpolated = {gcritpzI:.4e}')
print(f'kpz max = {kpzmax:.4e}')

databs = np.load(f'C:\ZZZ\Laburos\Codes\\newdata\datadbs-gth-{nkk}-steps-3e{kikf}-L-{L}.npy')#{n}L0.npy')
max_value = np.max((databs))
max_index = np.unravel_index(np.argmax((databs)), databs.shape)
del databs
kbs = kk[max_index[0]]
gcritbsI = 6*varIk(kbs)**3/abs(xi3Ik(kbs))/(deltac**3)
print(f'gcrit bs Interpolated = {gcritbsI:.4e}')

n = -1 # n=30-35 makes a ng peak appear
g0 = gcrits *n
g0 = gcritpzI *n 
# g0 = 1
g0n = -gcrit #*n
# g0 = gvec
print(' ')
print(f'n={n}')
print(' ')
# plt.plot(MH,gcrits*xi3E,'o')
# plt.plot(MH,gcrits*xi3E)
# plt.xscale('log')
# plt.yscale('symlog')
# plt.axhline(y=0, color='k', linestyle='--')
# # plt.xlim(2.5e-16, 2.5e-8)
# plt.yticks(list(plt.yticks()[0]) + [0])
# # plt.legend()
# plt.title(r'$\xi_3$, 'f'{Wf}')
# plt.xlabel(r'$M_H/M_\odot$')
# plt.show()

skew = (xi3)/varsmall**1.5
# plt.figure(00)
# # plt.plot(MH, g0*(xi3E)/var**1.5, label='skewE w/sigma**3')
# plt.plot(MHsmall, g0*(xi3)/varsmall**1.5, 'o', label='skew w/sigma**3')
# plt.plot(MHsmall, g0*(xi3)/varsmall**1.5, label='skew w/sigma**3')
# # plt.plot(kzsmall, g0*(xi3)/varsmall**1.5, 'o')
# # plt.plot(kzsmall, g0*(xi3)/varsmall**1.5)
# plt.ylabel(r'$\kappa_3/g_{eff}$')
# plt.xlabel(r'$M_H/M_\odot$')
# # plt.xlabel(r'k [Mpc$^{-1}$]')
# plt.axhline(y=0, color='k', linestyle='--')
# plt.xscale('log')
# plt.yscale('symlog')
# # plt.legend()
# plt.yticks(list(plt.yticks()[0]) + [0])
# plt.title(f'Skewness, {Wf}')
# # plt.xlim(1e-9, 1e-3)
# # plt.xlim(1e-14, 1e-8)
# plt.show()


# Define a function for setting the ticks and labels on the secondary axis
def format_func0(value, tick_number): #this one is for checking if the values of the ticks kz fit with the x values from MH
    return f'{kofMH(value):.1e}'
def format_func(value, tick_number):
    exponent = int(np.log10(kofMH(value)))
    return r'$10^{{{}}}$'.format(exponent)

# fig, ax1 = plt.subplots()
# ax1.plot(MHsmall, g0*(xi3)/varsmall**1.5, 'o', label='skew w/sigma**3')
# ax1.plot(MHsmall, g0*(xi3)/varsmall**1.5, label='skew w/sigma**3')
# ax1.set_ylabel(r'$\kappa_3=\dfrac{\xi_3}{\sigma^3}$')
# ax1.set_xlabel(r'$M_H/M_\odot$')
# ax1.axhline(y=0, color='k', linestyle='--')
# ax1.set_xscale('log')
# ax1.set_yscale('symlog')
# # ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# ax1.set_title(f'Skewness, {Wf}')
# # Create a secondary x-axis
# secax = ax1.twiny()
# # Set the tick positions and labels for the secondary x-axis
# secax.set_xscale('log')
# secax.set_xlim(ax1.get_xlim())
# major_ticks = np.geomspace(MHsmall[-1], MHsmall[0], 5)  # Adjust the number of major ticks as needed
# # Use the conversion function to set the secondary x-axis tick labels
# secax.set_xticks(major_ticks)
# secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
# secax.set_xlabel(r'k [Mpc$^{-1}$]')
# plt.show()


# sys.exit()


ngcont = g0*(xi3)*(deltac/varsmall)**3/6
ngcontE = g0*(xi3E)*(deltac/var)**3/6
ingcontmax = np.argmax(ngcont)
ngcontmax = np.amax(ngcont)
MHngmax = MHsmall[ingcontmax]


# plt.plot(MH, -0.5*deltac**2/var, label='gaussian')
# plt.plot(MH, ngcontE,'o', label='ngcontE')
# plt.plot(MHsmall, ngcont, label='ngcont')

ngmod = -(xi3)*deltac/varsmall**2/3.
# for g in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
# for g in [0.1,0.4,0.7,1]:
#     plt.plot(MHsmall, g*gcrits*ngmod, label=f'{g}*' r'$\xi_3*\frac{\delta_c}{3\sigma^4}$')
#     plt.xscale('log')
#     plt.yscale('symlog')
#     plt.axhline(y=1, color='r', linestyle='--')
#     plt.axhline(y=-1, color='r', linestyle='--')
#     plt.axhline(y=0, color='k', linestyle='--')
#     plt.axvline(x=MHngmax, color='k', linestyle='--')
#     # plt.title(r'ngcont: $\xi_3*\frac{\delta_c^3}{6\sigma^6}$, '+f'{Wf}')
#     plt.title(f'ng deviation {Wf}')
#     plt.legend()
#     plt.show()

# sys.exit()

if n==-1:
    var=var[2499:3170]
    xi3E=xi3E[2499:3170]
    MH=MH[2499:3170]
    LMH=LMH[2499:3170]
#######################################
######### monochromatic plots #########
#######################################

# abundance, high peak limit

frac = 0.2
frac = 1

beta_mono_g = 2*frac*var**0.5/(deltac*np.sqrt(2*np.pi))*np.exp(-0.5*deltac**2/var)
beta_mono_gsmall = 2*frac*varsmall**0.5/(deltac*np.sqrt(2*np.pi))*np.exp(-0.5*deltac**2/varsmall)

beta_mono_pert = beta_mono_g*(1+g0*(xi3E)*(deltac/var)**3/6)
beta_mono_pertsmall = beta_mono_gsmall*(1+g0*(xi3)*(deltac/varsmall)**3/6)

beta_mono_resum = 2*frac*var**0.5/(deltac*np.sqrt(2*np.pi))*np.exp(-0.5*deltac**2/var+g0*(xi3E)*(deltac/var)**3/6)
beta_mono_resum = beta_mono_g*np.exp(g0*(xi3E)*(deltac/var)**3/6)
beta_mono_resumsmall = beta_mono_gsmall*np.exp(g0*(xi3)*(deltac/varsmall)**3/6)


# # should i plot frac*MH or MH vs beta,f, etc?
# plt.plot(frac*MH, beta_mono_g, label='Gaussian')
# # plt.plot(MH, beta_mono_pert, label='pert')
# plt.plot(frac*MH, beta_mono_resum,  label='non-Gaussian')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.title(f'Monochromatic PBH Abundance, {Wf}, ' r'$\lambda_0=$' f'{L}')
# # plt.ylim(1e-20, 1e3)
# plt.xlim(1e-13, 3e-10)
# plt.ylim(1e-20, 1e-14)
# plt.ylabel(r'$\beta$')
# if frac == 1:
#     plt.xlabel(r'$M_H\,/\,M_\odot$')
# else:
#     plt.xlabel(f'{frac}*' r'$M_H\,/\,M_\odot$')
# plt.show()

# plt.plot(frac*MHsmall, beta_mono_gsmall, 'o', label='gaussian')
# # plt.plot(MHsmall, beta_mono_pertsmall, 'o', label='pert')
# plt.plot(frac*MHsmall, beta_mono_resumsmall, 'o', label='resum')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlim(1e-13, 3e-10)
# plt.ylim(1e-20, 1e-14)
# plt.axvline(x=frac*MHngmax, color='k', linestyle='--')
# plt.title(f'Monochromatic $\\beta(M_H)$ small, {Wf}')
# plt.xlabel(f'{frac}*' r'$M_H\,/\,M_\odot$')
# plt.show()


# monochromatic f
# f_mono_resum[np.isnan(f_mono_resum)] = 0
# f_mono_resum[np.isinf(f_mono_resum)] = 0

f_mono_g = OmegaCDM**-1*(Meq/(frac*MH))**0.5*beta_mono_g
f_mono_gsmall = OmegaCDM**-1*(Meq/(frac*MHsmall))**0.5*beta_mono_gsmall
fpbh_mono_g = -Intarray_vec(f_mono_g, LMH)
fpbh_mono_gsmall = -Intarray_vec(f_mono_gsmall, LMHsmall)
OmegaPBH_mono_g = OmegaCDM*fpbh_mono_g
OmegaPBH_mono_gsmall = OmegaCDM*fpbh_mono_gsmall

f_mono_pert = OmegaCDM**-1*(Meq/(frac*MH))**0.5*beta_mono_pert
f_mono_pertsmall = OmegaCDM**-1*(Meq/(frac*MHsmall))**0.5*beta_mono_pertsmall
fpbh_mono_pert = -Intarray_vec(f_mono_pert, LMH)
fpbh_mono_pertsmall = -Intarray_vec(f_mono_pertsmall, LMHsmall)
OmegaPBH_mono_pert = OmegaCDM*fpbh_mono_pert
OmegaPBH_mono_pertsmall = OmegaCDM*fpbh_mono_pertsmall

f_mono_resum = OmegaCDM**-1*(Meq/(frac*MH))**0.5*beta_mono_resum
f_mono_resumsmall = OmegaCDM**-1*(Meq/(frac*MHsmall))**0.5*beta_mono_resumsmall
fpbh_mono_resum = -Intarray_vec(f_mono_resum, LMH)
OmegaPBH_mono_resum = OmegaCDM*fpbh_mono_resum
fpbh_mono_resumsmall = -Intarray_vec(f_mono_resumsmall, LMHsmall)
OmegaPBH_mono_resumsmall = OmegaCDM*fpbh_mono_resumsmall


# plt.plot(frac*MHsmall, beta_mono_gsmall/fpbh_mono_gsmall, 'o', label='gaussian')
# # plt.plot(MHsmall, beta_mono_pertsmall, 'o', label='pert')
# plt.plot(frac*MHsmall, beta_mono_resumsmall/fpbh_mono_resumsmall, 'o', label='resum')
# plt.xscale('log')
# plt.yscale('log')
# # plt.ylim(1e-16, 4)
# plt.ylim(1e-20, 1e-14)
# # plt.axvline(x=0.2*MHngmax, color='k', linestyle='--')
# plt.axvline(x=frac*MHsmall[np.argmax(beta_mono_resumsmall/fpbh_mono_resumsmall)], color='k', linestyle='--')
# plt.axvline(x=frac*MHsmall[120], color='orange', linestyle='--')
# plt.axhline(y=5.66e-16, color='cyan', linestyle='--', label='present day PBH density constraint')
# plt.xlabel(r'$M_{\rm PBH}=$' f'{frac}' r'$\,M_H\,/\,M_\odot$')
# plt.title(r'Monochromatic $\beta(M_H)/f_{PBH}$, ' f'{Wf}')
# plt.legend()
# plt.show()

# plt.plot(frac*MH, beta_mono_g/fpbh_mono_g, label='gaussian')
# # plt.plot(MHsmall, beta_mono_pertsmall, 'o', label='pert')
# plt.plot(frac*MH, beta_mono_resum/fpbh_mono_resum, label='resum')
# plt.xscale('log')
# plt.yscale('log')
# # plt.ylim(1e-16, 4)
# plt.ylim(1e-20, 1e-14)
# # plt.axvline(x=0.2*MHngmax, color='k', linestyle='--')
# plt.axvline(x=frac*MHsmall[np.argmax(beta_mono_resumsmall/fpbh_mono_resumsmall)], color='k', linestyle='--')
# # plt.axhline(y=5.66e-16, color='cyan', linestyle='--', label='outdated present day femtolensing constraint')
# plt.xlabel(r'$M_{\rm PBH}=$' f'{frac}' r'$\,M_H\,/\,M_\odot$')
# plt.title(r'Monochromatic $\beta(M_H)/f_{PBH}$, ' f'{Wf}')
# plt.legend()
# plt.show()


# print(f'OmegaPBH_mono_g: {OmegaPBH_mono_g:.2e}, OmegaPBH_mono_gsmall: {OmegaPBH_mono_gsmall:.2e}')
iMHmax_mono = np.argmax(f_mono_resum)
MHmax_mono = MH[np.argmax(f_mono_resum)]


plt.plot(frac*MH, f_mono_g, label='Gaussian')
# plt.plot(frac*MH, f_mono_pert/fpbh_mono_pert,'o', label='pert')
plt.plot(frac*MH, f_mono_resum, label='non-Gaussian')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$f(M)$')
plt.legend()
plt.title(f'Monochromatic f(M), {Wf}, ' r'$\lambda_0=$' f'{L}')
plt.ylim(1e-14, 70)
plt.xlim(3e-14, 3e-10)
if frac == 1:
    plt.xlabel(r'$M_H\,/\,M_\odot$')
else:
    plt.xlabel(f'{frac}*' r'$M_H\,/\,M_\odot$')
# plt.axvline(x=frac*MHmax_mono, color='k')
# plt.axvline(x=frac*MHngmax, color='k', linestyle='--')
plt.show()


# plt.plot(frac*MH, f_mono_g/fpbh_mono_g, '--', label='gaussian')
# # plt.plot(frac*MH, f_mono_pert/fpbh_mono_pert,'o', label='pert')
# plt.plot(frac*MH, f_mono_resum/fpbh_mono_resum, label='resum')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$M_{\rm PBH}\,/\,M_\odot$')
# plt.legend(loc='center right', bbox_to_anchor=(0.999, 0.86))
# plt.title(f'Monochromatic f(M)/fpbh, {Wf}')
# # plt.ylim(1e-10, 1e1)
# plt.ylim(1e-17, 1e1)
# plt.xlim(1e-19, 1e-5)
# plt.axhline(y=1, color='r', linestyle='--')
# # plt.yticks(list(plt.yticks()[0]) + [1])
# # plt.axvline(x=frac*MHngmax, color='k', linestyle='--')
# plt.axvline(x=frac*MH[np.argmax(f_mono_g/fpbh_mono_g)], color='k', linestyle=':')
# plt.show()


#######################################
########### extended plots ############
#######################################

#######################################
# extended beta
#######################################
def intbetaextended(M, MH, var, xi3, g, deltac):
    mu = M/(C*MH)
    deltaM = mu**(1./gamma) + deltac
    bg = 2/np.sqrt(2*np.pi*var)*np.exp(-0.5*deltaM**2/var)*M/MH/gamma*mu**(1./gamma)
    ngc = g*xi3*(deltac/var)**3/6
    return bg, bg*(1+ngc), bg*np.exp(ngc)


beta_ex_g = np.zeros(len(MH))
beta_ex_pert = np.zeros(len(MH))
beta_ex_resum = np.zeros(len(MH))

for i in tqdm.tqdm(range(len(MH))):
    a, b, c = intbetaextended(MH[i], MH, var, xi3E, g0, deltac)
    beta_ex_g[i] = -Intarray_vec(a, LMH)
    beta_ex_pert[i] = -Intarray_vec(b, LMH)
    beta_ex_resum[i] = -Intarray_vec(c, LMH)

outdated = 1e-19*(MHsmall/(2.5137e-19))**0.5
# # plt.plot( MH, beta_ex_pert, label='pert')
# plt.plot( MH, beta_ex_g, label='Gaussian')
# plt.plot( MH, beta_ex_resum, label='non-Gaussian')
# plt.plot(MHsmall, outdated, 'r--', label='Outdated Femtolensing Constraint')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$M_H\,/\,M_\odot$')
# plt.legend()
# plt.title(f'Extended PBH Abundance, {Wf}, ' r'$\lambda_0=$' f'{L}')
# # plt.savefig(f'beta_ex_{Wf}_{n}g.pdf')
# plt.xlim(1e-14, 3e-10)
# plt.ylim(1e-20, 1e-14)
# plt.ylabel(r'$\beta$')
# if frac == 1:
#     plt.xlabel(r'$M_H\,/\,M_\odot$')
# else:
#     plt.xlabel(f'{frac}*' r'$M_H\,/\,M_\odot$')
# plt.show()


# plt.plot( MH, beta_mono_resum, label='NG Monochromatic')
# plt.plot( MH, beta_ex_resum, label='NG Extended')
# plt.plot(MHsmall, outdated, 'r--', label='Outdated Femtolensing Constraint')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$M_H\,/\,M_\odot$')
# plt.legend()
# plt.title(f'NG PBH Abundance, {Wf}, ' r'$\lambda_0=$' f'{L}')
# # plt.savefig(f'beta_ex_{Wf}_{n}g.pdf')
# plt.xlim(1e-14, 3e-10)
# plt.ylim(1e-20, 1e-14)
# plt.ylabel(r'$\beta$')
# # plt.axhline(y=outdated, color='red', linestyle='--', label='outdated present day femtolensing constraint')
# if frac == 1:
#     plt.xlabel(r'$M_H\,/\,M_\odot$')
# else:
#     plt.xlabel(f'{frac}*' r'$M_H\,/\,M_\odot$')
# plt.show()

#######################################
# f extended integration
#######################################
def intfextended(M, MH, var, xi3, g, deltac):
    mu = M/(C*MH)
    deltaM = mu**(1./gamma) + deltac
    fg = 2/OmegaCDM/np.sqrt(2*np.pi*var)*np.exp(-0.5*deltaM**2/var)*M/MH/gamma*mu**(1./gamma)*np.sqrt(Meq/MH)
    ngc = g*xi3*(deltac/var)**3/6
    return fg, fg*(1+ngc), fg*np.exp(ngc)


f_ex_g = np.zeros(len(MH))
f_ex_pert = np.zeros(len(MH))
f_ex_resum = np.zeros(len(MH))

for i in tqdm.tqdm(range(len(MH))):
    a, b, c = intfextended(MH[i], MH, var, xi3E, g0, deltac)
    f_ex_g[i] = -Intarray_vec(a, LMH)
    f_ex_pert[i] = -Intarray_vec(b, LMH)
    f_ex_resum[i] = -Intarray_vec(c, LMH)




fpbh_ex_g = -Intarray_vec(f_ex_g, LMH)
# fpbh_ex_gsmall = -Intarray_vec(f_ex_gsmall, LMHsmall)
OmegaPBH_ex_g = OmegaCDM*fpbh_ex_g
# OmegaPBH_ex_gsmall = OmegaCDM*fpbh_ex_gsmall


fpbh_ex_pert = -Intarray_vec(f_ex_pert, LMH)
# fpbh_ex_pertsmall = -Intarray_vec(f_ex_pertsmall, LMHsmall)
OmegaPBH_ex_pert = OmegaCDM*fpbh_ex_pert
# OmegaPBH_ex_pertsmall = OmegaCDM*fpbh_ex_pertsmall


fpbh_ex_resum = -Intarray_vec(f_ex_resum, LMH)
OmegaPBH_ex_resum = OmegaCDM*fpbh_ex_resum
# fpbh_ex_resumsmall = -Intarray_vec(f_ex_resumsmall, LMHsmall)
# OmegaPBH_ex_resumsmall = OmegaCDM*fpbh_ex_resumsmall


# print(f'OmegaPBH_ex_g: {OmegaPBH_ex_g:.2e}')#), OmegaPBH_ex_gsmall: {OmegaPBH_ex_gsmall:.2e}')

# plt.plot( MH, beta_ex_g, label='Gaussian')
# plt.plot( MH, beta_ex_resum, label='non-Gaussian')
# plt.plot( MH, beta_ex_g/fpbh_ex_g, label='gaussian')
# plt.plot( MH, beta_ex_resum/fpbh_ex_resum, label='resum')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$M_H\,/\,M_\odot$')
# plt.legend()
# plt.title(r'Extended $\beta(MH)/f_{\rm PBH}$' f', g={n}g$_c$, {Wf}')
# plt.ylim(1e-39, 1e-7)
# # plt.savefig(f'beta_ex_{Wf}_{n}g_fpbh.pdf')
# plt.show()


########################################################
''' DEBERÍA GRAFICAR CON RESPECTO A MPBH=0.2MH O MH? '''
########################################################
# note that M_peak ~ MHofk(L0*k0*0.5) = 1.77e-13 and maxMpbh = 2.37e-13

plt.plot( MH, f_ex_g, label='Gaussian')
# plt.plot(frac*MH, f_ex_pert/fpbh_ex_pert,'o', label='pert')
plt.plot( MH, f_ex_resum, label='non-Gaussian')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\rm PBH}\,/\,M_\odot$')
plt.ylabel(r'$f(M_{\rm PBH})$')
plt.legend()
plt.title(f'Extended PBH Mass Function, {Wf}, ' r'$\lambda_0=$' f'{L}')
plt.ylim(1e-18,80)
plt.xlim(1e-18, 3e-8)
plt.show()



# plt.plot(MH, f_ex_g/fpbh_ex_g, linestyle='--', color='#6EB5FF', label='gaussian')
# # plt.plot(MH, f_ex_g/fpbh_ex_g,'k--', label='gaussian')
# # plt.plot(frac*MH, f_ex_pert/fpbh_ex_pert,'o', label='pert')
# plt.plot( MH, f_ex_resum/fpbh_ex_resum, color='#ff7f0e', label='resum')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$M_{\rm PBH}=$' f'{frac}*' r'$M_H\,/\,M_\odot$')
# plt.legend(loc='center right', bbox_to_anchor=(0.999, 0.86))
# plt.title(f'Extended f(M)/fpbh, {Wf}')
# # plt.ylim(1e-10, 1e1)
# plt.ylim(1e-17, 1e1)
# plt.xlim(1e-19, 1e-5)
# plt.axhline(y=1, color='r', linestyle='--')
# # plt.axvline(x=MH[imaxM], color='k')
# # plt.axvline(x=frac*MH[np.argmax(f_ex_g/fpbh_ex_g)], color='red')
# # plt.axvline(x=MH[np.argmax(f_ex_resum/fpbh_ex_resum)], color='cyan')
# # plt.yticks(list(plt.yticks()[0]) + [1])
# plt.show()


# print(np.amax(f_ex_g/fpbh_ex_g), np.amax(f_ex_resum/fpbh_ex_resum))



# plt.plot(frac*MH, f_mono_g/fpbh_mono_g,'--', label='gaussian mono')
# plt.plot(frac*MH, f_ex_g/fpbh_ex_g,'-.', label='gaussian ex')

# plt.plot( frac*MH, f_mono_resum/fpbh_mono_resum, '--', label='resum mono')
# plt.plot(MH, f_ex_resum/fpbh_ex_resum, label='resum ex')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$M_{\rm PBH}\,/\,M_\odot$')
# plt.title(f'f(M)/fpbh, {Wf}')
# plt.ylim(1e-17, 1e1)
# plt.xlim(1e-19, 1e-5)
# plt.axhline(y=1, color='r', linestyle='--')
# # plt.axvline(x = MH[imaxM], color='k', label='max var')
# # plt.axvline(x = MH[np.argmax(f_mono_resum/fpbh_mono_resum)], color='r', label='max f mono')
# # plt.axvline(x = MH[np.argmax(f_ex_resum/fpbh_ex_resum)], color='cyan', label='max f ex')
# # plt.yticks(list(plt.yticks()[0]) + [1])
# plt.legend(loc='center right', bbox_to_anchor=(0.999, 0.86))
# plt.show()

imaxM = np.argmax(var)
maxM = MH[imaxM]
maxMpbh = frac*maxM
# imaxM = np.argmin(np.abs(MH-maxMpbh))
# imaxM = np.argmin(np.abs(MH-MH[np.argmax(f_mono_resum/fpbh_mono_resum)]))

# imaxM = iMHmax_mono +55
# imaxM = ingcontmax # ng pdf goes dirac
delta = np.linspace(-0.5, 0.5, 2000)
g_pdf = 1./(np.sqrt(2*np.pi*var[imaxM]))*np.exp(-0.5*delta**2/var[imaxM])
ng_pdf = 1./(np.sqrt(2*np.pi*var[imaxM]))*np.exp(-0.5*delta**2/var[imaxM]+g0*xi3E[imaxM]*(delta/var[imaxM])**3./6.)
g_norm = Intarray_vec(g_pdf, delta)
ng_norm = Intarray_vec(ng_pdf, delta)

# plt.plot(delta, g_pdf, label='Gaussian')
# plt.plot(delta, ng_pdf, label='non-Gaussian')
# plt.axvline(x=deltac, color='r', linestyle='--', label = f'$\delta_c={deltac}$')
# plt.axhline(y=0, color='gray', linewidth=0.5)
# plt.legend()
# # plt.title(f'PDF for $M_H={MH[imaxM]:.2e}$, {Wf}')
# plt.title(r'PDF for max $\sigma^2(M_H)$ value, 'f'{Wf}')
# # plt.axvline(x=MHngmax, color='k', linestyle='--')
# plt.show()

fpeak_mono_g = np.amax(f_mono_g)
fpeak_mono_resum = np.amax(f_mono_resum)
fpeak_ex_g = np.amax(f_ex_g)
fpeak_ex_resum = np.amax(f_ex_resum)
#####################################################
############# plots with multiple W's ###############
#####################################################

if Wf == 'Wg4':
    xi3_Wg4 = xi3E
    var_Wg4 = var
    g_pdf_Wg4 = g_pdf
    ng_pdf_Wg4 = ng_pdf
    g_norm_Wg4 = g_norm
    ng_norm_Wg4 = ng_norm

    beta_mono_g_Wg4 = beta_mono_g
    beta_mono_resum_Wg4 = beta_mono_resum
    f_mono_g_Wg4 = f_mono_g
    f_mono_resum_Wg4 = f_mono_resum
    fpbh_mono_g_Wg4 = fpbh_mono_g
    fpbh_mono_resum_Wg4 = fpbh_mono_resum

    beta_ex_g_Wg4 = beta_ex_g
    beta_ex_resum_Wg4 = beta_ex_resum
    f_ex_g_Wg4 = f_ex_g
    f_ex_resum_Wg4 = f_ex_resum
    fpbh_ex_g_Wg4 = fpbh_ex_g
    fpbh_ex_resum_Wg4 = fpbh_ex_resum

elif Wf == 'Wthtf':
    xi3_Wthtf = xi3E
    var_Wthtf = var
    g_pdf_Wthtf = g_pdf
    ng_pdf_Wthtf = ng_pdf
    g_norm_Wthtf = g_norm
    ng_norm_Wthtf = ng_norm

    beta_mono_g_Wthtf = beta_mono_g
    beta_mono_resum_Wthtf = beta_mono_resum
    f_mono_g_Wthtf = f_mono_g
    f_mono_resum_Wthtf = f_mono_resum
    fpbh_mono_g_Wthtf = fpbh_mono_g
    fpbh_mono_resum_Wthtf = fpbh_mono_resum

    beta_ex_g_Wthtf = beta_ex_g
    beta_ex_resum_Wthtf = beta_ex_resum
    f_ex_g_Wthtf = f_ex_g
    f_ex_resum_Wthtf = f_ex_resum
    fpbh_ex_g_Wthtf = fpbh_ex_g
    fpbh_ex_resum_Wthtf = fpbh_ex_resum
#########################
if Wf == 'Wthtf' and L == 233.548:
    g0_1 = g0
    xi3_1 = xi3E
    var_1 = var
    xi3small_1 = xi3
    varsmall_1 = varsmall
    skew_1 = skew
    L1 = L
    Mpeak_mono_g_1 = MH[np.argmax(f_mono_g)]
    Mpeak_mono_resum_1 = MH[np.argmax(f_mono_resum)]

    beta_mono_g_1 = beta_mono_g
    beta_mono_resum_1 = beta_mono_resum
    f_mono_g_1 = f_mono_g
    f_mono_resum_1 = f_mono_resum
    fpbh_mono_g_1 = fpbh_mono_g
    fpbh_mono_resum_1 = fpbh_mono_resum

    beta_ex_g_1 = beta_ex_g
    beta_ex_resum_1 = beta_ex_resum
    f_ex_g_1 = f_ex_g
    f_ex_resum_1 = f_ex_resum
    fpbh_ex_g_1 = fpbh_ex_g
    fpbh_ex_resum_1 = fpbh_ex_resum
    
elif Wf == 'Wthtf' and L == 233.758:
    g0_2 = g0
    xi3_2 = xi3E
    var_2 = var
    xi3small_2 = xi3
    varsmall_2 = varsmall
    skew_2 = skew
    L2 = L
    Mpeak_mono_g_2 = MH[np.argmax(f_mono_g)]
    Mpeak_mono_resum_2 = MH[np.argmax(f_mono_resum)]

    beta_mono_g_2 = beta_mono_g
    beta_mono_resum_2 = beta_mono_resum
    f_mono_g_2 = f_mono_g
    f_mono_resum_2 = f_mono_resum
    fpbh_mono_g_2 = fpbh_mono_g
    fpbh_mono_resum_2 = fpbh_mono_resum

    beta_ex_g_2 = beta_ex_g
    beta_ex_resum_2 = beta_ex_resum
    f_ex_g_2 = f_ex_g
    f_ex_resum_2 = f_ex_resum
    fpbh_ex_g_2 = fpbh_ex_g
    fpbh_ex_resum_2 = fpbh_ex_resum

elif Wf == 'Wg4' and  L == 226.206:
    g0_3 = g0
    xi3_3 = xi3E
    var_3 = var
    xi3small_3 = xi3
    varsmall_3 = varsmall
    skew_3 = skew
    L3 = L
    Mpeak_mono_g_3 = MH[np.argmax(f_mono_g)]
    Mpeak_mono_resum_3 = MH[np.argmax(f_mono_resum)]

    beta_mono_g_3 = beta_mono_g
    beta_mono_resum_3 = beta_mono_resum
    f_mono_g_3 = f_mono_g
    f_mono_resum_3 = f_mono_resum
    fpbh_mono_g_3 = fpbh_mono_g
    fpbh_mono_resum_3 = fpbh_mono_resum

    beta_ex_g_3 = beta_ex_g
    beta_ex_resum_3 = beta_ex_resum
    f_ex_g_3 = f_ex_g
    f_ex_resum_3 = f_ex_resum
    fpbh_ex_g_3 = fpbh_ex_g
    fpbh_ex_resum_3 = fpbh_ex_resum

elif Wf == 'Wg4' and L == 226.518:
    g0_4 = g0
    xi3_4 = xi3E
    var_4 = var
    xi3small_4 = xi3
    varsmall_4 = varsmall
    skew_4 = skew
    L4 = L
    Mpeak_mono_g_4 = MH[np.argmax(f_mono_g)]
    Mpeak_mono_resum_4 = MH[np.argmax(f_mono_resum)]

    beta_mono_g_4 = beta_mono_g
    beta_mono_resum_4 = beta_mono_resum
    f_mono_g_4 = f_mono_g
    f_mono_resum_4 = f_mono_resum
    fpbh_mono_g_4 = fpbh_mono_g
    fpbh_mono_resum_4 = fpbh_mono_resum

    beta_ex_g_4 = beta_ex_g
    beta_ex_resum_4 = beta_ex_resum
    f_ex_g_4 = f_ex_g
    f_ex_resum_4 = f_ex_resum
    fpbh_ex_g_4 = fpbh_ex_g
    fpbh_ex_resum_4 = fpbh_ex_resum


print(f'L = {L}')
print(f'gcrit = {gcrit:.2e}, wf = {Wf}, deltac = {deltac}')
print(' ')
print(f'fpbh g mono: {fpbh_mono_g:.4}, fpbh ng mono: {fpbh_mono_resum:.4}')
print(f'fpeak g mono: {fpeak_mono_g:.4}, fpeak ng mono: {fpeak_mono_resum:.4}')
print(' ')
print(f'fpbh g ex: {fpbh_ex_g:.4}, fpbh ng ex: {fpbh_ex_resum:.4}')
print(f'fpeak g ex: {fpeak_ex_g:.4}, fpeak ng ex: {fpeak_ex_resum:.4}')
print(' ')
print(f'Mpeak g mono: {MH[np.argmax(f_mono_g)]:.4}, Mpeak ng mono: {MH[np.argmax(f_mono_resum)]:.4}')
print(f'Mpeak g ex: {MH[np.argmax(f_ex_g)]:.4}, Mpeak ng ex: {MH[np.argmax(f_ex_resum)]:.4}')
sys.exit()

print(f'Mpeak_mono_g_1: {Mpeak_mono_g_1:.2e},\nMpeak_mono_g_2: {Mpeak_mono_g_2:.2e},\nMpeak_mono_g_3: {Mpeak_mono_g_3:.2e},\nMpeak_mono_g_4: {Mpeak_mono_g_4:.2e}')
print(f'Mpeak_mono_resum_1: {Mpeak_mono_resum_1:.2e},\nMpeak_mono_resum_2: {Mpeak_mono_resum_2:.2e},\nMpeak_mono_resum_3: {Mpeak_mono_resum_3:.2e},\nMpeak_mono_resum_4: {Mpeak_mono_resum_4:.2e}')

# abuncance constraint, valid for 1e-16 to 1e-13 solar masses
Mbeta1 = np.argmin(np.abs(MH-1e-16))
Mbeta2 = np.argmin(np.abs(MH-1e-11))
Mbeta = MH[Mbeta2:Mbeta1]
beta_constraint = 1e-19*(Mbeta/(frac* 2.5137e-19))**0.5

from matplotlib.ticker import FuncFormatter

# Custom formatter function
def format_ticks1(y, pos):
    if y == 0:
        return '0'
    elif abs(y) < 1e-12:
        return f'{y:.0e}'
    else:
        return f'{y:.1e}'
# xi3
plt.plot(MH, (xi3_Wthtf), 'tab:orange',label='Wthtf')
plt.plot(MH, (xi3_Wg4), '--','tab:blue', label='Wg4')
plt.xscale('log')
plt.yscale('symlog')
plt.axhline(y=0, color='gray', linewidth=0.5)
plt.ylabel(r'$\xi_3/g_{eff}$')
plt.xlabel(r'$M_H/M_\odot$')
plt.xlim(5.6e-16, 2.8e-6)
plt.yticks(list(plt.yticks()[0]) + [0])

formatter = FuncFormatter(format_ticks1)
plt.gca().yaxis.set_major_formatter(formatter)

plt.legend()
plt.title('Smoothed 3-Point Connected Function')
# plt.savefig(f'xi3_vs_{kikf}.pdf')
plt.show()


# skewness

def format_ticks2(y, pos):
    if y == 0:
        return '0'
    elif abs(y) < 1e-8:
        return f'{y:.0e}'
    else:
        return f'{y:.1e}'


#########################################

# beta mono
plt.plot( MH, beta_mono_g_Wg4, label='Gaussian, Wg4')
plt.plot( MH, beta_mono_resum_Wg4, label='non-Gaussian, Wg4')
plt.plot( MH, beta_mono_g_Wthtf, label='Gaussian, Wthtf')
plt.plot( MH, beta_mono_resum_Wthtf, label='non-Gaussian, Wthtf')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_H\,/\,M_\odot$')
plt.legend()
plt.title(r'Monochromatic $\beta(MH)$' f', g={n}g$_c$, {Wf}')
plt.ylim(1e-27, 1e6)
# plt.savefig(f'beta_mono_vs_{n}g_{kikf}.pdf')
plt.show()



# beta extended
plt.plot( MH, beta_ex_g_Wg4, label='Gaussian, Wg4')
plt.plot( MH, beta_ex_resum_Wg4, label='non-Gaussian, Wg4')
plt.plot( MH, beta_ex_g_Wthtf, label='Gaussian, Wthtf')
plt.plot( MH, beta_ex_resum_Wthtf, label='non-Gaussian, Wthtf')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_H\,/\,M_\odot$')
plt.legend()
plt.title(r'Extended $\beta(MH)$' f', g={n}g$_c$, {Wf}')
plt.ylim(1e-27, 1e6)
# plt.savefig(f'beta_ex_vs_{n}g_{kikf}.pdf')
plt.show()



#########################################


# Define a custom formatter function
def custom_formatter_y(x, pos):
    if x == 1:
        return '1'
    else:
        return f'$10^{{{int(np.log10(x))}}}$' if x != 0 else '0'
import matplotlib.ticker as ticker 
# tab:orange color='#ff7f0e'
# linestyle=(0, (5, 10))
# celeste #00c3f3 ó #00bdeb ó #00BFFF  #00ABEB

plt.plot( MH, f_mono_g_Wg4, color='#00BFFF', alpha=1, linestyle='-', label = r'$f_G$, Wg4')
plt.plot( MH, f_mono_resum_Wg4, '--', color='tab:orange', label = r'$f_{NG}$, Wg4')
plt.plot( MH, f_mono_g_Wthtf, ':', color='k', label=r'$f_G$, Wthtf')
plt.plot( MH, f_mono_resum_Wthtf, color='#8C2EB9', linestyle=(0, (5, 2, 1, 2, 1, 2)) , label=r'$f_{NG}$, Wthtf')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_H\,/\,M_\odot$')
# plt.legend(loc='center right', bbox_to_anchor=(0.999, 0.81))
plt.legend()
plt.title(r'Monochromatic $f(MH)/f_{\rm PBH}$' f', g={n}g$_c$')
# plt.axhline(y=1, color='r', linestyle='--', linewidth=0.92)
# plt.yticks(list(plt.yticks()[0]) + [1])
plt.ylim(1e-17, 1e1)
plt.xlim(3e-18, 6e-9)
# Use FuncFormatter with the custom formatter function
# ax = plt.gca()
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter_y))
# plt.savefig(f'f_mono_vs_{n}g_fpbh_{kikf}.pdf')
plt.show()


# f extended
plt.plot( MH, f_ex_g_Wg4 , color='#00BFFF', alpha=1, linestyle='-', label = r'$f_G$, Wg4')
plt.plot( MH, f_ex_resum_Wg4 , '--', color='tab:orange', label = r'$f_{NG}$, Wg4')
plt.plot( MH, f_ex_g_Wthtf , ':', color='k', label=r'$f_G$, Wthtf')
plt.plot( MH, f_ex_resum_Wthtf , color='#8C2EB9', linestyle=(0, (5, 2, 1, 2, 1, 2)) , label=r'$f_{NG}$, Wthtf')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_H\, [M_\odot]$')
plt.legend()
plt.title(r'Extended $f(MH)$' f', g={n}g$_c$, {Wf}')
plt.ylim(1e-48, 1e2)
plt.xlim(1e-18, 3e-9)
# plt.savefig(f'f_ex_vs_{n}g_{kikf}.pdf')
plt.show()




# Transfer function plot
def Wth(k,q):
    a=3.*(np.sin(k/q)-k/q*np.cos(k/q))/(k/q)**3. 
    return a
def Wtf(k,q):
    b=3.*(np.sin(csrad*k/q)-csrad*k/q*np.cos(csrad*k/q))/(csrad*k/q)**3. 
    return b 
def Wthtf(k,q):
    a=3.*(np.sin(k/q)-k/q*np.cos(k/q))/(k/q)**3. 
    b=3.*(np.sin(csrad*k/q)-csrad*k/q*np.cos(csrad*k/q))/(csrad*k/q)**3. 
    return a*b 

kRHsym = np.linspace(-40,40,1000)
plt.plot(kRHsym, Wth(kRHsym,1), lw=0.98,label=r'$T(kR_H)$', color='k')
plt.plot(kRHsym, Wtf(kRHsym,1), label=r'$T(c_skR_H)$', color='tab:orange')
plt.plot(kRHsym, Wthtf(kRHsym,1), '-.',label=r'$T(kR_H)*T(c_skR_H)$', color='#00ABEB')
plt.legend()
plt.plot(kRHsym, Wth(kRHsym,1),lw=0.9, label=r'$T(kR_H)$', color='k')
plt.title('Fourier Transform of Spherical Real-Space Top-hat')
plt.xlabel(r'kR_H')
# plt.axvline(1,lw=0.5,color='gray')
plt.xlim(-37, 37)
# plt.savefig('TF_sym.pdf')
plt.show()

kRHpositive = np.linspace(0,14,1000)
# plt.fill_between(kRHpositive, Wthtf(kRHpositive, 1), where=(kRHpositive >= 1) & (kRHpositive <= 4.4934), color='#00ABEB', alpha=0.275, label='PBH forming scales')
plt.plot(kRHpositive, Wth(kRHpositive,1),lw=0.99, label=r'$T(kR_H)$', color='k')
plt.plot(kRHpositive, Wtf(kRHpositive,1),  label=r'$T(c_skR_H)$', color='tab:orange')
plt.plot(kRHpositive, Wthtf(kRHpositive,1), '-.',label=r'$T(kR_H)*T(c_skR_H)$', color='#00ABEB')
plt.legend()
plt.plot(kRHpositive, Wth(kRHpositive,1),lw=0.9, label=r'$T(kR_H)$', color='k')
plt.title('Fourier Transform of a Real-Space Spherical Top-hat Function ')
# plt.title('Fourier Transform of a Spherical Top-hat Function in Real Space')
plt.xlabel(r'$kR_H$')
plt.axvline(1,lw=0.5,color='gray')
plt.xticks(list(plt.xticks()[0]) + [1])
plt.xlim(0, 14)
# plt.savefig('TF_pos.pdf')
plt.show()
#############################################
#

# celeste color='#00BFFF' #00ABEB
# morado color='#781aa5 ' #9133BE #8C2EB9 
i0 = np.argmin(abs(MHofk(kpzmax)-MHsmall))

extraytick = (g0_3*(xi3small_3))[i0]
fig, ax1 = plt.subplots()
ax1.plot(MHsmall, g0_3*(xi3small_3) , label=r'Wg4, $\lambda_0=226.206$', color='k')
ax1.plot(MHsmall, g0_4*(xi3small_4) ,label=r'Wg4, $\lambda_0=226.518$', color='#00ABEB')
ax1.plot(MHsmall, g0_1*(xi3small_1) ,'-.', label=r'Wthtf, $\lambda_0=233.548$', color='#004080')
ax1.plot(MHsmall, g0_2*(xi3small_2) ,'-.', label=r'Wthtf, $\lambda_0=233.758$', color='tab:orange')
ax1.legend(loc='lower left')
ax1.set_ylabel(r'$\xi_3$')
ax1.set_xlabel(r'$M_H \, [M_\odot]$')
# plt.axhline(y=0, color='gray', linewidth=0.5)
# ax1.axvline(MHofk(kpzmax), color='k', linestyle='--', linewidth=0.5)
ax1.set_xscale('log')
ax1.set_yscale('symlog')
# ax1.set_legend()
yticks = list(ax1.get_yticks())
yticks.append(0)
yticks.append(extraytick)
ax1.set_yticks((yticks))
ax1.set_title(f'Smoothed 3-point Connected Function')
# Create a secondary x-axis
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
secax.set_xlim(ax1.get_xlim())
# major_ticks = np.geomspace(MHsmall[-1], MHsmall[0], 5)  # Adjust the number of major ticks as needed
major_ticks = [MHofk(k) for k in [1e15, 1e14, 1e13, 1e12]]
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
formatter = FuncFormatter(format_ticks1)
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
# plt.savefig('xi3_vs_lambda0.pdf')
ax1.axvline(MHofk(kpzmax), color='k', linestyle='--', linewidth=0.5)
plt.show()




# extraytick = [np.min(g0_3*(xi3small_3)/varsmall_3**1.5)]
# extraytick = [np.min(g0_1*(xi3small_1)/varsmall_1**1.5), np.min(g0_3*(xi3small_3)/varsmall_3**1.5)]

extraytick = [(g0_1*(xi3small_1)/varsmall_1**1.5)[i0], (g0_3*(xi3small_3)/varsmall_3**1.5)[i0]]
fig, ax1 = plt.subplots()
ax1.plot(MHsmall, g0_3*(xi3small_3)/varsmall_3**1.5,'-.', lw='1.1', label=r'Wg4, $\lambda_0=226.206$', color='k')
ax1.plot(MHsmall, g0_4*(xi3small_4)/varsmall_4**1.5,'-.',lw='1.1',  label=r'Wg4, $\lambda_0=226.518$', color='#00ABEB')
ax1.plot(MHsmall, g0_1*(xi3small_1)/varsmall_1**1.5,lw='0.98', label=r'Wthtf, $\lambda_0=233.548$', color='#004080')
ax1.plot(MHsmall, g0_2*(xi3small_2)/varsmall_2**1.5,lw='0.98',label=r'Wthtf, $\lambda_0=233.758$', color='tab:orange')
ax1.legend(loc='upper left')
ax1.set_ylabel(r'$\xi_3/\sigma^3$')
ax1.set_xlabel(r'$M_H \, [M_\odot]$')
# ax1.axhline(y=0, color='k', linestyle='--')
plt.axhline(y=0, color='gray', linewidth=0.5)
ax1.set_xscale('log')
ax1.set_yscale('symlog')
# ax1.set_legend()
yticks = list(ax1.get_yticks())
yticks.append(0)
# yticks.append(extraytick)
yticks = yticks+extraytick
ax1.set_yticks(yticks)
ax1.set_title(f'Skewness')
# Create a secondary x-axis
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
secax.set_xlim(ax1.get_xlim())
# major_ticks = np.geomspace(MHsmall[-1], MHsmall[0], 5)  # Adjust the number of major ticks as needed
major_ticks = [MHofk(k) for k in [1e15, 1e14, 1e13, 1e12]]
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
formatter = FuncFormatter(format_ticks1)
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
# plt.savefig('skewness_vs_lambda0.pdf')
plt.axvline(MHofk(kpzmax), color='k', linestyle='--', linewidth=0.5)
plt.show()



# extraytick = np.min((xi3small_4))
# fig, ax1 = plt.subplots()
# ax1.plot(MHsmall, (xi3small_3) , label=r'Wg4, $\lambda_0=226.206$', color='k')
# ax1.plot(MHsmall, (xi3small_4) ,label=r'Wg4, $\lambda_0=226.518$', color='#00ABEB')
# ax1.plot(MHsmall, (xi3small_1) ,'-.', label=r'Wthtf, $\lambda_0=233.548$', color='#003366')
# ax1.plot(MHsmall, (xi3small_2) ,'-.', label=r'Wthtf, $\lambda_0=233.758$', color='tab:orange')
# ax1.legend(loc='lower left')
# ax1.set_ylabel(r'$\xi_3$')
# ax1.set_xlabel(r'$M_H \, [M_\odot]$')
# # plt.axhline(y=0, color='gray', linewidth=0.5)
# # ax1.axvline(MHofk(kpzmax), color='k', linestyle='--', linewidth=0.5)
# ax1.set_xscale('log')
# ax1.set_yscale('symlog')
# # ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# # yticks.append(extraytick)
# ax1.set_yticks(yticks)
# ax1.set_title(f'Smoothed 3-point Connected Function')
# # Create a secondary x-axis
# secax = ax1.twiny()
# # Set the tick positions and labels for the secondary x-axis
# secax.set_xscale('log')
# secax.set_xlim(ax1.get_xlim())
# # major_ticks = np.geomspace(MHsmall[-1], MHsmall[0], 5)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(k) for k in [1e15, 1e14, 1e13, 1e12]]
# # Use the conversion function to set the secondary x-axis tick labels
# secax.set_xticks(major_ticks)
# secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
# secax.set_xlabel(r'k [Mpc$^{-1}$]')
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
# plt.tight_layout()
# # plt.savefig('xi3_vs_lambda0.pdf')
# plt.show()



# extraytick = np.min((xi3small_4)/varsmall_4**1.5)
# fig, ax1 = plt.subplots()
# ax1.plot(MHsmall, (xi3small_3)/varsmall_3**1.5, label=r'Wg4, $\lambda_0=226.206$', color='k')
# ax1.plot(MHsmall, (xi3small_4)/varsmall_4**1.5, label=r'Wg4, $\lambda_0=226.518$', color='#00ABEB')
# ax1.plot(MHsmall, (xi3small_1)/varsmall_1**1.5,'-.', label=r'Wthtf, $\lambda_0=233.548$', color='#003366')
# ax1.plot(MHsmall, (xi3small_2)/varsmall_2**1.5,'-.', label=r'Wthtf, $\lambda_0=233.758$', color='tab:orange')
# ax1.legend(loc='upper left')
# ax1.set_ylabel(r'$\xi_3/\sigma^3$')
# ax1.set_xlabel(r'$M_H \, [M_\odot]$')
# # ax1.axhline(y=0, color='k', linestyle='--')
# plt.axhline(y=0, color='gray', linewidth=0.5)
# ax1.set_xscale('log')
# ax1.set_yscale('symlog')
# # ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# yticks.append(extraytick)
# # plt.axhline(y=extraytick, color='gray', linewidth=0.5)
# ax1.set_yticks(yticks)
# ax1.set_title(f'Skewness')
# # Create a secondary x-axis
# secax = ax1.twiny()
# # Set the tick positions and labels for the secondary x-axis
# secax.set_xscale('log')
# secax.set_xlim(ax1.get_xlim())
# # major_ticks = np.geomspace(MHsmall[-1], MHsmall[0], 5)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(k) for k in [1e15, 1e14, 1e13, 1e12]]
# # Use the conversion function to set the secondary x-axis tick labels
# secax.set_xticks(major_ticks)
# secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
# secax.set_xlabel(r'k [Mpc$^{-1}$]')
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
# plt.tight_layout()
# # plt.savefig('skewness_vs_lambda0.pdf')
# plt.show()


###################################
########## plos ng final ##########
###################################
# beta_ex_resum_4
# fig, ax1 = plt.subplots()
# # ax1.plot(MH, wth_fmono, '--', color='tab:blue', label=r'Wth, $\lambda_0=228.210$')
# ax1.plot(MH, f_mono_g_1, color='tab:orange', label=r'Wthtf, $\lambda_0=233.548$')
# ax1.plot(MH, f_mono_g_2, '--', color='k', label=r'Wthtf, $\lambda_0=233.758$')
# ax1.plot(MH, f_mono_g_3, '-.',color='k', linewidth=1,label=r'Wg4, $\lambda_0=226.206$')
# ax1.plot(MH, f_mono_g_4, '-.',color='#00ABEB', linewidth=1,label=r'Wg4, $\lambda_0=226.518$')
# ax1.set_title(f'Monochromatic PBH Mass Function, Gaussian Statistics', pad=8)#, fontsize=16)
# ax1.set_xlabel(r'$M\, [M_\odot]$')
# ax1.set_ylabel(r'$f(M)$')
# # plt.axhline(y=1, color='r', linestyle='--')
# ax1.set_ylim(1e-48, 1e2)
# ax1.set_xlim(1e-17, 1e1)
# ax1.legend(loc='lower right')
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# # ax1.set_legend()
# # yticks = list(ax1.get_yticks())
# # yticks.append(0)
# # ax1.set_yticks(yticks)
# # Create a secondary x-axis
# secax = ax1.twiny()
# # Set the tick positions and labels for the secondary x-axis
# secax.set_xscale('log')
# secax.set_xlim(ax1.get_xlim())
# major_ticks = np.geomspace(MH[-1], MH[0], 10)  # Adjust the number of major ticks as needed
# # Use the conversion function to set the secondary x-axis tick labels
# secax.set_xticks(major_ticks)
# secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
# secax.set_xlabel(r'k [Mpc$^{-1}$]')
# # formatter = FuncFormatter(format_ticks1)
# # plt.gca().yaxis.set_major_formatter(formatter)
# ax1.set_ylim(1e-48, 1e2)
# ax1.set_xlim(1e-17, 1e1)
# plt.tight_layout()
# # plt.savefig('fmono.pdf')
# plt.show()



# ARREGLAR TICKS!!!!!!!!!
fig, ax1 = plt.subplots()
# ax1.plot(MH, f_ex_resum_2, color='k',label=r'Wthtf, $\lambda_0=233.758$')
# ax1.plot(MH, f_ex_g_2,color='tab:orange',  label=r'Wthtf, $\lambda_0=233.758$')
ax1.plot(MH, f_mono_g_1,'-.',color='k',  label=r'Wthtf, $f_{\rm \, PBH, ex}^{\,G}=$'f'{fpbh_mono_g_1:.3}'[:-1])
ax1.plot(MH, f_mono_resum_1, color='tab:orange',label=r'Wthtf, $f_{\rm PBH, ex}^{\,NG}=$'f'{fpbh_mono_resum_1:.3}'[:-1])
ax1.set_title(f'Monochromatic PBH Mass Function, non-Gaussian Statistics', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.legend(loc='lower left')
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
ax1.set_ylim(1e-14, 1e1)
ax1.set_xlim(8e-15, 2e-11)
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
xlims = ax1.get_xlim()
secax.set_xlim(xlims)
# major_ticks = np.geomspace(MHofk(1e13), MHofk(1e12), 2)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(k) for k in [1e13, 1e12]]
major_ticks = np.geomspace(MH[-1], MH[0], 10)
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func0(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
secax.set_xlim(xlims)
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
# plt.savefig('fmono_ng_wthtf.pdf')
plt.show()






# ARREGLAR TICKS!!!!!!!!!
fig, ax1 = plt.subplots()
# ax1.plot(MH, f_ex_resum_2, color='k',label=r'Wthtf, $\lambda_0=233.758$')
# ax1.plot(MH, f_ex_g_2,color='tab:orange',  label=r'Wthtf, $\lambda_0=233.758$')
ax1.plot(MH, f_ex_g_2,'-.',color='k',  label=r'Wthtf, $f_{\rm \, PBH, ex}^{\,G}=$'f'{fpbh_ex_g_2:.3}'[:-1])
ax1.plot(MH, f_ex_resum_2, color='tab:orange',label=r'Wthtf, $f_{\rm PBH, ex}^{\,NG}=$'f'{fpbh_ex_resum_2:.3}'[:-1])
ax1.set_title(f'Extended PBH Mass Function, non-Gaussian Statistics', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.legend(loc='lower left')
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
ax1.set_ylim(1e-14, 1e1)
ax1.set_xlim(8e-15, 2e-11)
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
xlims = ax1.get_xlim()
secax.set_xlim(xlims)
# major_ticks = np.geomspace(MHofk(1e13), MHofk(1e12), 2)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(k) for k in [1e13, 1e12]]
major_ticks = np.geomspace(MH[-1], MH[0], 10)
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func0(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
secax.set_xlim(xlims)
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
# plt.savefig('fex_ng_wthtf.pdf')
plt.show()



# ARREGLAR TICKS!!!!!!!!!
fig, ax1 = plt.subplots()
# ax1.plot(MH, f_ex_resum_2, color='k',label=r'Wthtf, $\lambda_0=233.758$')
# ax1.plot(MH, f_ex_g_2,color='tab:orange',  label=r'Wthtf, $\lambda_0=233.758$')
ax1.plot(MH, f_mono_g_3,'-.',color='k',  label=r'Wg4, $f_{\rm \, PBH, ex}^{\,G}=$'f'{fpbh_mono_g_3:.3}'[:-1])
ax1.plot(MH, f_mono_resum_3, color='tab:orange',label=r'Wg4, $f_{\rm PBH, ex}^{\,NG}=$'f'{fpbh_mono_resum_3:.3}'[:-1])
ax1.set_title(f'Monochromatic PBH Mass Function, non-Gaussian Statistics', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.legend(loc='lower left')
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
ax1.set_ylim(1e-14, 1e1)
ax1.set_xlim(8e-15, 2e-11)
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
xlims = ax1.get_xlim()
secax.set_xlim(xlims)
# major_ticks = np.geomspace(MHofk(1e13), MHofk(1e12), 2)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(k) for k in [1e13, 1e12]]
major_ticks = np.geomspace(MH[-1], MH[0], 10)
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func0(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
secax.set_xlim(xlims)
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
# plt.savefig('fex_ng_wg4.pdf')
plt.show()




# ARREGLAR TICKS!!!!!!!!!
fig, ax1 = plt.subplots()
# ax1.plot(MH, f_ex_resum_2, color='k',label=r'Wthtf, $\lambda_0=233.758$')
# ax1.plot(MH, f_ex_g_2,color='tab:orange',  label=r'Wthtf, $\lambda_0=233.758$')
ax1.plot(MH, f_ex_g_4,'-.',color='k',  label=r'Wg4, $f_{\rm \, PBH, ex}^{\,G}=$'f'{fpbh_ex_g_4:.3}'[:-1])
ax1.plot(MH, f_ex_resum_4, color='tab:orange',label=r'Wg4, $f_{\rm PBH, ex}^{\,NG}=$'f'{fpbh_ex_resum_4:.3}'[:-1])
ax1.set_title(f'Extended PBH Mass Function, non-Gaussian Statistics', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.legend(loc='lower left')
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
ax1.set_ylim(1e-14, 1e1)
ax1.set_xlim(8e-15, 2e-11)
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
xlims = ax1.get_xlim()
secax.set_xlim(xlims)
# major_ticks = np.geomspace(MHofk(1e13), MHofk(1e12), 2)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(k) for k in [1e13, 1e12]]
major_ticks = np.geomspace(MH[-1], MH[0], 10)
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func0(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
secax.set_xlim(xlims)
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
# plt.savefig('fex_ng_wg4.pdf')
plt.show()





# ARREGLAR TICKS!!!!!!!!!
fig, ax1 = plt.subplots()
# ax1.plot(MH, f_ex_resum_2, color='k',label=r'Wthtf, $\lambda_0=233.758$')
# ax1.plot(MH, f_ex_g_2,color='tab:orange',  label=r'Wthtf, $\lambda_0=233.758$')
ax1.plot(MH, beta_ex_g_4,'-.',color='k',  label=r'Wg4')
ax1.plot(MH, beta_ex_resum_4, color='tab:orange',label=r'Wg4')
ax1.set_title(f'Extended PBH Abundance, non-Gaussian Statistics', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$\beta(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.legend(loc='lower left')
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
ax1.set_ylim(1e-28, 1e-14)
ax1.set_xlim(8e-15, 2e-11)
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
xlims = ax1.get_xlim()
secax.set_xlim(xlims)
# major_ticks = np.geomspace(MHofk(1e13), MHofk(1e12), 2)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(k) for k in [1e13, 1e12]]
major_ticks = np.geomspace(MH[-1], MH[0], 10)
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func0(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
secax.set_xlim(xlims)
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
# plt.savefig('betaex_ng_wg4.pdf')
plt.show()




fig, ax1 = plt.subplots()
ax1.plot(MH, f_ex_g_2,linestyle=(0, (5, 2, 1, 2, 1, 2)),color='k',  label=r'Wthtf, $f_{\rm \, PBH, ex}^{\,G}=$'f'{fpbh_ex_g_2:.3}'[:-1])
ax1.plot(MH, f_ex_resum_2,'--', color='#00ABEB',label=r'Wthtf, $f_{\rm PBH, ex}^{\,NG}=$'f'{fpbh_ex_resum_2:.3}'[:-1])
ax1.plot(MH, f_ex_g_4,'-.',color='#8C2EB9',  label=r'Wg4, $f_{\rm \, PBH, ex}^{\,G}=$'f'{fpbh_ex_g_4:.3}'[:-1])
ax1.plot(MH, f_ex_resum_4, color='tab:orange',label=r'Wg4, $f_{\rm PBH, ex}^{\,NG}=$'f'{fpbh_ex_resum_4:.3}'[:-1])
ax1.set_title(f'Extended PBH Mass Function Comparison', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.legend(loc='lower left', bbox_to_anchor=(0.1, 0))
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
ax1.set_ylim(1e-14, 1e1)
ax1.set_xlim(2e-15, 3e-11)
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
xlims = ax1.get_xlim()
secax.set_xlim(xlims)
# major_ticks = np.geomspace(MHofk(1e13), MHofk(1e12), 2)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(k) for k in [1e13, 1e12]]
# major_ticks = np.geomspace(MH[-1], MH[0], 10)
major_ticks = [MHofk(1e14),MHofk(1e13),MHofk(1e12)]
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
secax.set_xlim(xlims)
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.savefig('fex_ng_bothW_cb.pdf')
plt.show()



from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LogLocator, NullFormatter
# ARREGLAR TICKS!!!!!!!!!
fig, ax1 = plt.subplots()
ax1.plot(MH, f_mono_g_1,linestyle=(0, (5, 2, 1, 2, 1, 2)),color='k',  label=r'Wthtf, $f_{\rm \, PBH, mc}^{\,G}=$'f'{fpbh_mono_g_1:.3}'[:-1])
ax1.plot(MH, f_mono_resum_1,'--', color='#00ABEB',label=r'Wthtf, $f_{\rm PBH, mc}^{\,NG}=$'f'{fpbh_mono_resum_1:.3}'[:-1])
ax1.plot(MH, f_mono_g_3,'-.',color='#8C2EB9',  label=r'Wg4, $f_{\rm \, PBH, mc}^{\,G}=$'f'{fpbh_mono_g_3:.3}'[:-1])
ax1.plot(MH, f_mono_resum_3, color='tab:orange',label=r'Wg4, $f_{\rm PBH, mc}^{\,NG}=$'f'{fpbh_mono_resum_3:.3}'[:-1])
ax1.set_title(f'Monochromatic PBH Mass Function Comparison', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.legend(loc='lower left')#, bbox_to_anchor=(0.12, 0))
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
ax1.set_ylim(1e-14, 1e1)
ax1.set_xlim(2e-15, 3e-11)
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
xlims = ax1.get_xlim()
secax.set_xlim(xlims)
# major_ticks = np.geomspace(MHofk(1e13), MHofk(1e12), 2)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(k) for k in [1e13, 1e12]]
# major_ticks = np.geomspace(MH[-1], MH[0], 10)
major_ticks = [MHofk(1e14),MHofk(1e13),MHofk(1e12)]  # Adjust the number of major ticks as needed
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
secax.set_xlim(xlims)
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
# secax.xaxis.set_minor_locator(LogLocator())  # Use log-based tick locations
# secax.xaxis.set_minor_formatter(NullFormatter())  # Disable minor tick labels
plt.tight_layout()
# inset_ax = inset_axes(ax1, width="40%", height="40%", loc='upper left')
# Adjust the bbox_to_anchor values for better positioning
inset_ax = inset_axes(ax1, width="40%", height="40%", loc='upper left', bbox_to_anchor=(0.08, -0.01, 1, 1), bbox_transform=ax1.transAxes)
inset_ax.loglog(MH, f_mono_g_2,linestyle=(0, (5, 2, 1, 2, 1, 2)),color='k')
inset_ax.loglog(MH, f_mono_resum_2, '--', color='#00ABEB')
inset_ax.loglog(MH, f_mono_resum_4, color='tab:orange')
inset_ax.loglog(MH, f_mono_g_4,'-.',color='#8C2EB9')

# inset_ax.set_title('Inset')
inset_ax.set_ylim(1e-1, 1e1)
inset_ax.set_xlim(4.5e-13, 2.7e-12)
# inset_ax.set_xticks([4e-13,2e-12])
# Remove minor ticks (if you don't want them)
# inset_ax.xaxis.set_minor_locator(LogLocator())  # Use log-based tick locations
# inset_ax.xaxis.set_minor_formatter(NullFormatter())  # Disable minor tick labels
# inset_ax.set_xticklabels([r'$4 \times 10^{-13}$', r'$2 \times 10^{-12}$'])
# inset_ax.set_yticks([])
plt.savefig('fmono_ng_bothW_inset_cb.pdf')
plt.show()




fig, ax1 = plt.subplots()
ax1.plot(MH, beta_mono_g_1,'-.',color='k',  label=r'Wthtf, $f_{\rm \, PBH, mc}^{\,G}=$'f'{fpbh_mono_g_1:.3}'[:-1])
ax1.plot(MH, beta_mono_resum_1, color='#00ABEB',label=r'Wthtf, $f_{\rm PBH, mc}^{\,NG}=$'f'{fpbh_mono_resum_1:.3}'[:-1])
ax1.plot(MH, beta_mono_g_3,'-.',color='#8C2EB9',  label=r'Wg4, $f_{\rm \, PBH, mc}^{\,G}=$'f'{fpbh_mono_g_3:.3}'[:-1])
ax1.plot(MH, beta_mono_resum_3, color='tab:orange',label=r'Wg4, $f_{\rm PBH, mc}^{\,NG}=$'f'{fpbh_mono_resum_3:.3}'[:-1])
ax1.set_title(f'Monochromatic PBH Mass Function Comparison', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.legend(loc='lower left')#, bbox_to_anchor=(0.12, 0))
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
# ax1.set_ylim(1e-14, 1e1)
# ax1.set_xlim(2e-15, 3e-11)
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
xlims = ax1.get_xlim()
secax.set_xlim(xlims)
# major_ticks = np.geomspace(MHofk(1e13), MHofk(1e12), 2)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(k) for k in [1e13, 1e12]]
# major_ticks = np.geomspace(MH[-1], MH[0], 10)
major_ticks = [MHofk(1e14),MHofk(1e13),MHofk(1e12)]  # Adjust the number of major ticks as needed
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
secax.set_xlim(xlims)
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
# inset_ax = inset_axes(ax1, width="30%", height="30%", loc='upper left')
# inset_ax.loglog(MH, f_mono_g_2,'-.',color='k')
# inset_ax.loglog(MH, f_mono_resum_2, color='#00ABEB')
# inset_ax.loglog(MH, f_mono_resum_4, color='tab:orange')
# inset_ax.loglog(MH, f_mono_g_4,'-.',color='#8C2EB9')
# # inset_ax.set_title('Inset')
# inset_ax.set_ylim(1e-2, 1e1)
# inset_ax.set_xlim(4.5e-13, 2.7e-12)
# inset_ax.set_xticks([])
# inset_ax.set_yticks([])
# plt.savefig('betamono_ng_bothW.pdf')
plt.show()


fig, ax1 = plt.subplots()
ax1.plot(MH, beta_ex_g_g4,'-.', color='tab:orange',label='Extended Wg4')
ax1.plot(MH, beta_ex_g_tf, color='#00ABEB',label='Extended Wthtf')
ax1.plot(MH, beta_mono_g_3,'--',color='#8C2EB9',  label='Monochromatic Wg4')
ax1.plot(MH, beta_mono_g_1, linestyle=(0, (5, 2, 1, 2, 1, 2)) ,color='k',  label= 'Monochromatic Wthtf')

ax1.set_title(f'Monochromatic and Extended PBH Abundance, Gaussian Statistics', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$\beta$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.legend(loc='lower left')#, bbox_to_anchor=(0.12, 0))
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_yticks(yticks)
# Create a secondary x-axis
ax1.set_ylim(1e-63, 1e-13)
ax1.set_xlim(1e-17, 1e-10)
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
xlims = ax1.get_xlim()
secax.set_xlim(xlims)
# major_ticks = np.geomspace(MHofk(1e13), MHofk(1e12), 2)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(k) for k in [1e13, 1e12]]
# major_ticks = np.geomspace(MH[-1], MH[0], 10)
major_ticks = [MHofk(1e15),MHofk(1e14),MHofk(1e13),MHofk(1e12)]  # Adjust the number of major ticks as needed
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
secax.set_xlim(xlims)
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
# plt.savefig('beta_monoex_g_bothW.pdf')
plt.show()






for i in n:
    plt.plot(frac*MH, f_mono_g, label='Gaussian')
    # plt.plot(frac*MH, f_mono_pert/fpbh_mono_pert,'o', label='pert')
    plt.plot(frac*MH, f_mono_resum, label='non-Gaussian')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$M_{\rm PBH}\,/\,M_\odot$')
    plt.legend()
    plt.title(f'Monochromatic f(M), {Wf}, ' r'$\lambda_0=$' f'{L}')
    plt.ylim(1e-14, 70)
    plt.xlim(3e-14, 3e-10)
    if frac == 1:
        plt.xlabel(r'$M_H\,/\,M_\odot$')
    else:
        plt.xlabel(f'{frac}*' r'$M_H\,/\,M_\odot$')
    # plt.axvline(x=frac*MHmax_mono, color='k')
    # plt.axvline(x=frac*MHngmax, color='k', linestyle='--')
    plt.show()
    plt.plot( MH, f_ex_g, label='Gaussian')
    # plt.plot(frac*MH, f_ex_pert/fpbh_ex_pert,'o', label='pert')
    plt.plot( MH, f_ex_resum, label='non-Gaussian')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$M_{\rm PBH}\,/\,M_\odot$')
    plt.ylabel(r'$f(M_{\rm PBH})$')
    plt.legend()
    plt.title(f'Extended PBH Mass Function, {Wf}, ' r'$\lambda_0=$' f'{L}')
    plt.ylim(1e-18,80)
    plt.xlim(1e-18, 3e-8)
    plt.show()