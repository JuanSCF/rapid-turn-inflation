
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
n=1
L0=251.327
L=n*L0
k0 = 1e11 # Mpc^-1 . '''this gives k_peak=1.287e13 Mpc^-1'''

# initial and final k that will be integrated
ki = 11
kf = 13
kp = k0*L/2.

kikf = str(ki)+str(kf)
ki = 3*10**ki 
kf = 3*10**kf
print(f'ki: {ki:.0e}, kf: {kf:.1e}, kp: {kp:.0e}') #, Mp: {MHofk(kp):.2e}')


# test Wg4tf
Wf = 'Wg4' # Wg, Wg4, Wth, Wthtf 

nkk = 350 #number of steps
size = 3000


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
xi3_file=f'xi3d-gth-{Wf}-{nkk}-steps-3e{kikf}-lambda-{n}L0.npy'
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
    

fgaussian_data_file = os.path.join(cwd, f'data\\gaussian-data-{Wf}.npz')
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


plt.plot(kzsmall, xi3,'o')
plt.plot(kzsmall, xi3)
plt.xscale('log')
plt.yscale('symlog')
plt.axhline(y=0, color='k', linestyle='--')
plt.title(r'$\xi_3$, 'f'{Wf}')
plt.xlabel(r'k [Mpc$^{-1}$]')
plt.show()

# es dificil implementar el savgol acá
# xi3f = savgol_filter(np.log(xi3), 31, 3)
# xi3f = savgol_filter((xi3), 31, 3)
# xi3fi = interp1d(MHsmall, xi3f,bounds_error=False, kind='cubic',fill_value="extrapolate")

# These lines are to extend the xi3 function to the whole range of MH
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


plt.plot(MH,xi3E,'o')
plt.plot(MH,xi3E)
plt.xscale('log')
plt.yscale('symlog')
plt.axhline(y=0, color='k', linestyle='--')
# plt.xlim(2.5e-16, 2.5e-8)
plt.yticks(list(plt.yticks()[0]) + [0])
# plt.legend()
plt.title(r'$\xi_3$, 'f'{Wf}')
plt.xlabel(r'$M_H/M_\odot$')
plt.show()


# lets compute a g value for every xi3 value
# gvec = 6.*varsmall**3/(deltac**3.) /abs(xi3)
# gvec = 6.*var**3/(deltac**3.) /abs(xi3E)

# maximum g value allowed according to the perturbation condition
# deltac = 1
gcrits = 6*np.amax(varsmall)**3/abs(xi3[np.argmax(varsmall)])/(deltac**3)
gcrit = 6*np.amax(var)**3/abs(xi3E[np.argmax(var)])/(deltac**3)
print(f'gcrit = {gcrit:.2e}, wf = {Wf}, deltac = {deltac}')

nn = 0.88 #*0.883
g0 = gcrits *nn
# g0 = 1
g0n = -gcrit #*nn
# g0 = gvec

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
def format_func(value, tick_number): #this one is for checking if the values of the ticks kz fit with the x values from MH
    return f'{kofMH(value):.1e}'
def format_func(value, tick_number):
    exponent = int(np.log10(kofMH(value)))
    return r'$10^{{{}}}$'.format(exponent)

fig, ax1 = plt.subplots()
ax1.plot(MHsmall, g0*(xi3)/varsmall**1.5, 'o', label='skew w/sigma**3')
ax1.plot(MHsmall, g0*(xi3)/varsmall**1.5, label='skew w/sigma**3')
ax1.set_ylabel(r'$\xi_3(M_H)$')
ax1.set_xlabel(r'$M_H/M_\odot$')
ax1.axhline(y=0, color='k', linestyle='--')
ax1.set_xscale('log')
ax1.set_yscale('symlog')
# ax1.set_legend()
yticks = list(ax1.get_yticks())
yticks.append(0)
ax1.set_yticks(yticks)
ax1.set_title(f'Skewness, {Wf}')
# Create a secondary x-axis
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')

secax.set_xlim(ax1.get_xlim())
major_ticks = np.geomspace(MHsmall[-1], MHsmall[0], 4)  # Adjust the number of major ticks as needed
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
plt.show()


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

beta_mono_resum = beta_mono_g*np.exp(g0*(xi3E)*(deltac/var)**3/6)
beta_mono_resumsmall = beta_mono_gsmall*np.exp(g0*(xi3)*(deltac/varsmall)**3/6)


# should i plot frac*MH or MH vs beta,f, etc?
plt.plot(frac*MH, beta_mono_g, label='gaussian')
# plt.plot(MH, beta_mono_pert, label='pert')
plt.plot(frac*MH, beta_mono_resum,  label='resum')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title(f'Monochromatic $\\beta(M_H)$, {Wf}')
# plt.ylim(1e-20, 1e3)
# plt.xlim(1e-14, 1e-10)
plt.xlabel(f'{frac}*' r'$M_H\,/\,M_\odot$')
plt.show()

plt.plot(frac*MHsmall, beta_mono_gsmall, 'o', label='gaussian')
# plt.plot(MHsmall, beta_mono_pertsmall, 'o', label='pert')
plt.plot(frac*MHsmall, beta_mono_resumsmall, 'o', label='resum')
plt.xscale('log')
plt.yscale('log')
plt.legend()
# plt.ylim(1e-20, 1e3)
plt.axvline(x=frac*MHngmax, color='k', linestyle='--')
plt.title(f'Monochromatic $\\beta(M_H)$ small, {Wf}')
plt.xlabel(f'{frac}*' r'$M_H\,/\,M_\odot$')
plt.show()


# monochromatic f

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

plt.plot(frac*MH, beta_mono_g/fpbh_mono_g, label='gaussian')
# plt.plot(MHsmall, beta_mono_pertsmall, 'o', label='pert')
plt.plot(frac*MH, beta_mono_resum/fpbh_mono_resum, label='resum')
plt.xscale('log')
plt.yscale('log')
# plt.ylim(1e-16, 4)
plt.ylim(1e-20, 1e-14)
# plt.axvline(x=0.2*MHngmax, color='k', linestyle='--')
plt.axvline(x=frac*MHsmall[np.argmax(beta_mono_resumsmall/fpbh_mono_resumsmall)], color='k', linestyle='--')
# plt.axhline(y=5.66e-16, color='cyan', linestyle='--', label='outdated present day femtolensing constraint')
plt.xlabel(r'$M_{\rm PBH}=$' f'{frac}' r'$\,M_H\,/\,M_\odot$')
plt.title(r'Monochromatic $\beta(M_H)/f_{PBH}$, ' f'{Wf}')
plt.legend()
plt.show()


print(f'OmegaPBH_mono_g: {OmegaPBH_mono_g:.2e}, OmegaPBH_mono_gsmall: {OmegaPBH_mono_gsmall:.2e}')
iMHmax_mono = np.argmax(f_mono_resum)
MHmax_mono = MH[np.argmax(f_mono_resum)]

# frac = 1
plt.plot(frac*MH, f_mono_g, label='gaussian')
# plt.plot(frac*MH, f_mono_pert/fpbh_mono_pert,'o', label='pert')
plt.plot(frac*MH, f_mono_resum, label='resum')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\rm PBH}\,/\,M_\odot$')
plt.legend()
plt.title(f'Monochromatic f(M), {Wf}')
plt.ylim(1e-14, 1e14)
plt.xlim(1e-15, 1e-11)
plt.axvline(x=frac*MHmax_mono, color='k')
plt.axvline(x=frac*MHngmax, color='k', linestyle='--')
plt.show()


plt.plot(frac*MH, f_mono_g/fpbh_mono_g, '--', label='gaussian')
# plt.plot(frac*MH, f_mono_pert/fpbh_mono_pert,'o', label='pert')
plt.plot(frac*MH, f_mono_resum/fpbh_mono_resum, label='resum')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\rm PBH}\,/\,M_\odot$')
plt.legend(loc='center right', bbox_to_anchor=(0.999, 0.86))
plt.title(f'Monochromatic f(M)/fpbh, {Wf}')
# plt.ylim(1e-10, 1e1)
plt.ylim(1e-17, 1e1)
plt.xlim(1e-19, 1e-5)
plt.axhline(y=1, color='r', linestyle='--')
# plt.yticks(list(plt.yticks()[0]) + [1])
# plt.axvline(x=frac*MHngmax, color='k', linestyle='--')
plt.axvline(x=frac*MH[np.argmax(f_mono_g/fpbh_mono_g)], color='k', linestyle=':')
plt.show()

print(' f_g mono/fpbh peak', np.amax(f_mono_g/fpbh_mono_g),'\n f_ng mono/fpbh peak', np.amax(f_mono_resum/fpbh_mono_resum))


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


# plt.plot( MH, beta_ex_pert, label='pert')
plt.plot( MH, beta_ex_g, label='Gaussian')
plt.plot( MH, beta_ex_resum, label='non-Gaussian')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_H\,/\,M_\odot$')
plt.legend()
plt.title(r'Extended $\beta(MH)$' f', g={nn}g$_c$, {Wf}')
plt.ylim(1e-27, 1e6)
# plt.savefig(f'beta_ex_{Wf}_{nn}g.pdf')
plt.show()


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


print(f'OmegaPBH_ex_g: {OmegaPBH_ex_g:.2e}')#), OmegaPBH_ex_gsmall: {OmegaPBH_ex_gsmall:.2e}')

# plt.plot( MH, beta_ex_g, label='Gaussian')
# plt.plot( MH, beta_ex_resum, label='non-Gaussian')
plt.plot( MH, beta_ex_g/fpbh_ex_g, label='gaussian')
plt.plot( MH, beta_ex_resum/fpbh_ex_resum, label='resum')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_H\,/\,M_\odot$')
plt.legend()
plt.title(r'Extended $\beta(MH)/f_{\rm PBH}$' f', g={nn}g$_c$, {Wf}')
plt.ylim(1e-39, 1e-7)
# plt.savefig(f'beta_ex_{Wf}_{nn}g_fpbh.pdf')
plt.show()


########################################################
''' DEBERÍA GRAFICAR CON RESPECTO A MPBH=0.2MH O MH? '''
########################################################
# note that M_peak ~ MHofk(L0*k0*0.5) = 1.77e-13 and maxMpbh = 2.37e-13

plt.plot( MH, f_ex_g, label='gaussian')
# plt.plot(frac*MH, f_ex_pert/fpbh_ex_pert,'o', label='pert')
plt.plot( MH, f_ex_resum, label='resum')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\rm PBH}\,/\,M_\odot$')
plt.legend()
plt.title(f'Extended f(M), {Wf}')
plt.ylim(1e-12, 1e13)
plt.show()



plt.plot(MH, f_ex_g/fpbh_ex_g, linestyle='--', color='#6EB5FF', label='gaussian')
# plt.plot(MH, f_ex_g/fpbh_ex_g,'k--', label='gaussian')
# plt.plot(frac*MH, f_ex_pert/fpbh_ex_pert,'o', label='pert')
plt.plot( MH, f_ex_resum/fpbh_ex_resum, color='#ff7f0e', label='resum')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\rm PBH}=$' f'{frac}*' r'$M_H\,/\,M_\odot$')
plt.legend(loc='center right', bbox_to_anchor=(0.999, 0.86))
plt.title(f'Extended f(M)/fpbh, {Wf}')
# plt.ylim(1e-10, 1e1)
plt.ylim(1e-17, 1e1)
plt.xlim(1e-19, 1e-5)
plt.axhline(y=1, color='r', linestyle='--')
# plt.axvline(x=MH[imaxM], color='k')
# plt.axvline(x=frac*MH[np.argmax(f_ex_g/fpbh_ex_g)], color='red')
# plt.axvline(x=MH[np.argmax(f_ex_resum/fpbh_ex_resum)], color='cyan')
# plt.yticks(list(plt.yticks()[0]) + [1])
plt.show()


print(np.amax(f_ex_g/fpbh_ex_g), np.amax(f_ex_resum/fpbh_ex_resum))



# plt.plot(frac*MH, f_mono_g/fpbh_mono_g,'--', label='gaussian mono')
# plt.plot(frac*MH, f_ex_g/fpbh_ex_g,'-.', label='gaussian ex')

plt.plot( frac*MH, f_mono_resum/fpbh_mono_resum, '--', label='resum mono')
plt.plot(MH, f_ex_resum/fpbh_ex_resum, label='resum ex')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\rm PBH}\,/\,M_\odot$')
plt.title(f'f(M)/fpbh, {Wf}')
plt.ylim(1e-17, 1e1)
plt.xlim(1e-19, 1e-5)
plt.axhline(y=1, color='r', linestyle='--')
# plt.axvline(x = MH[imaxM], color='k', label='max var')
# plt.axvline(x = MH[np.argmax(f_mono_resum/fpbh_mono_resum)], color='r', label='max f mono')
# plt.axvline(x = MH[np.argmax(f_ex_resum/fpbh_ex_resum)], color='cyan', label='max f ex')
# plt.yticks(list(plt.yticks()[0]) + [1])
plt.legend(loc='center right', bbox_to_anchor=(0.999, 0.86))
plt.show()

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

plt.plot(delta, g_pdf, label='Gaussian')
plt.plot(delta, ng_pdf, label='non-Gaussian')
plt.axvline(x=deltac, color='r', linestyle='--', label = f'$\delta_c={deltac}$')
plt.axhline(y=0, color='gray', linewidth=0.5)
plt.legend()
# plt.title(f'PDF for $M_H={MH[imaxM]:.2e}$, {Wf}')
plt.title(r'PDF for max $\sigma^2(M_H)$ value, 'f'{Wf}')
# plt.axvline(x=MHngmax, color='k', linestyle='--')
plt.show()


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

sys.exit()

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
plt.title('Smoothed Connected 3-Point Function')
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

# plt.plot(MH[1900:2766], (xi3_Wthtf[1900:2766])/var_Wthtf[1900:2766]**1.5)
# plt.plot(MH[1900:2766], (xi3_Wthtf[1900:2766])/var_Wthtf[1900:2766]**1.5, 'orange',label='Wthtf')
# plt.plot(MH[1900:2766], (xi3_Wg4[1900:2766])/var_Wg4[1900:2766]**1.5, 'b--', label='Wg4')

plt.plot(MH[2000:2666], (xi3_Wthtf[2000:2666])/var_Wthtf[2000:2666]**1.5, 'orange',label='Wthtf')
plt.plot(MH[2000:2666], (xi3_Wg4[2000:2666])/var_Wg4[2000:2666]**1.5, 'b--', label='Wg4')
plt.xscale('log')
plt.yscale('symlog')
plt.axhline(y=0, color='gray', linewidth=0.5)
plt.ylabel(r'$\kappa_3/g_{eff}$')
plt.xlabel(r'$M_H/M_\odot$')
plt.title(f'Skewness')
# plt.xlim(5.6e-16, 2.8e-6)
plt.yticks(list(plt.yticks()[0]) + [0])
formatter = FuncFormatter(format_ticks2)
plt.gca().yaxis.set_major_formatter(formatter)
plt.legend( )
# plt.savefig(f'skewness_vs_{kikf}.pdf')
plt.show()

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
plt.title(r'Monochromatic $\beta(MH)$' f', g={nn}g$_c$, {Wf}')
plt.ylim(1e-27, 1e6)
# plt.savefig(f'beta_mono_vs_{nn}g_{kikf}.pdf')
plt.show()


plt.plot( MH, beta_mono_g_Wg4/fpbh_ex_g_Wg4, label='Gaussian, Wg4')
plt.plot( MH, beta_mono_resum_Wg4/fpbh_ex_resum_Wg4, label='non-Gaussian, Wg4')
plt.plot( MH, beta_mono_g_Wthtf/fpbh_ex_g_Wthtf, label='Gaussian, Wthtf')
plt.plot( MH, beta_mono_resum_Wthtf/fpbh_ex_resum_Wthtf, label='non-Gaussian, Wthtf')
plt.plot(Mbeta, beta_constraint, 'k--', label='outdated femtolensing constraints')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_H\,/\,M_\odot$')
plt.legend()
plt.title(r'Monochromatic $\beta(MH)/f_{\rm PBH}$' f', g={nn}g$_c$, {Wf}')
plt.ylim(1e-21, 1e-14)
# plt.savefig(f'beta_mono_vs_{nn}g_fpbh_{kikf}.pdf')
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
plt.title(r'Extended $\beta(MH)$' f', g={nn}g$_c$, {Wf}')
plt.ylim(1e-27, 1e6)
# plt.savefig(f'beta_ex_vs_{nn}g_{kikf}.pdf')
plt.show()

plt.plot( MH, beta_ex_g_Wg4/fpbh_ex_g_Wg4, color='#00BFFF', alpha=1, linestyle='-', label = r'$\beta_G$, Wg4')
plt.plot( MH, beta_ex_resum_Wg4/fpbh_ex_resum_Wg4, '--', color='tab:orange', label = r'$\beta_{NG}$, Wg4')
plt.plot( MH, beta_ex_g_Wthtf/fpbh_ex_g_Wthtf, ':', color='k', label=r'$\beta_G$, Wthtf')
plt.plot( MH, beta_ex_resum_Wthtf/fpbh_ex_resum_Wthtf, color='#781aa5', linestyle=(0, (5, 2, 1, 2, 1, 2)) , label=r'$\beta_{NG}$, Wthtf')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_H\,/\,M_\odot$')
plt.legend()
plt.title(r'Extended $\beta(MH)/f_{\rm PBH}$' f', g={nn}g$_c$, {Wf}')
plt.ylim(1e-39, 1e-7)
# plt.savefig(f'beta_ex_vs_{nn}g_fpbh_{kikf}.pdf')
plt.show()


#########################################

# f mono
plt.plot( MH, f_ex_g_Wg4 , color='#00BFFF', alpha=1, linestyle='-', label = r'$f_G$, Wg4')
plt.plot( MH, f_ex_resum_Wg4 , '--', color='tab:orange', label = r'$f_{NG}$, Wg4')
plt.plot( MH, f_ex_g_Wthtf , ':', color='k', label=r'$f_G$, Wthtf')
plt.plot( MH, f_ex_resum_Wthtf , color='#781aa5', linestyle=(0, (5, 2, 1, 2, 1, 2)) , label=r'$f_{NG}$, Wthtf')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_H\,/\,M_\odot$')
plt.legend()
plt.title(r'Monochromatic $f(MH)$' f', g={nn}g$_c$, {Wf}')
plt.ylim(1e-17, 1e19)
# plt.savefig(f'f_mono_vs_{nn}g_{kikf}.pdf')
plt.show()


# Define a custom formatter function
def custom_formatter_y(x, pos):
    if x == 1:
        return '1'
    else:
        return f'$10^{{{int(np.log10(x))}}}$' if x != 0 else '0'
import matplotlib.ticker as ticker 
# tab:orange color='#ff7f0e'
# linestyle=(0, (5, 10))
# celeste #00c3f3 ó #00bdeb ó #00BFFF
plt.plot( MH, f_mono_g_Wg4/fpbh_mono_g_Wg4, color='#00BFFF', alpha=1, linestyle='-', label = r'$f_G$, Wg4')
plt.plot( MH, f_mono_resum_Wg4/fpbh_mono_resum_Wg4, '--', color='tab:orange', label = r'$f_{NG}$, Wg4')
plt.plot( MH, f_mono_g_Wthtf/fpbh_mono_g_Wthtf, ':', color='k', label=r'$f_G$, Wthtf')
plt.plot( MH, f_mono_resum_Wthtf/fpbh_mono_resum_Wthtf, color='#781aa5', linestyle=(0, (5, 2, 1, 2, 1, 2)) , label=r'$f_{NG}$, Wthtf')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_H\,/\,M_\odot$')
plt.legend(loc='center right', bbox_to_anchor=(0.999, 0.81))
plt.title(r'Monochromatic $f(MH)/f_{\rm PBH}$' f', g={nn}g$_c$')
plt.ylim(1e-17, 1e1)
plt.xlim(3e-14, 6e-9)
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.92)
plt.yticks(list(plt.yticks()[0]) + [1])
plt.ylim(1e-17, 1e1)
plt.xlim(3e-14, 6e-9)
# Use FuncFormatter with the custom formatter function
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter_y))
# plt.savefig(f'f_mono_vs_{nn}g_fpbh_{kikf}.pdf')
plt.show()


# f extended
plt.plot( MH, f_ex_g_Wg4 , color='#00BFFF', alpha=1, linestyle='-', label = r'$f_G$, Wg4')
plt.plot( MH, f_ex_resum_Wg4 , '--', color='tab:orange', label = r'$f_{NG}$, Wg4')
plt.plot( MH, f_ex_g_Wthtf , ':', color='k', label=r'$f_G$, Wthtf')
plt.plot( MH, f_ex_resum_Wthtf , color='#781aa5', linestyle=(0, (5, 2, 1, 2, 1, 2)) , label=r'$f_{NG}$, Wthtf')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_H\,/\,M_\odot$')
plt.legend()
plt.title(r'Extended $f(MH)$' f', g={nn}g$_c$, {Wf}')
plt.ylim(1e-17, 1e19)
# plt.savefig(f'f_ex_vs_{nn}g_{kikf}.pdf')
plt.show()


plt.plot( MH, f_ex_g_Wg4/fpbh_ex_g_Wg4, color='#00BFFF', alpha=1, linestyle='-', label = r'$f_G$, Wg4')
plt.plot( MH, f_ex_resum_Wg4/fpbh_ex_resum_Wg4, '--', color='tab:orange', label = r'$f_{NG}$, Wg4')
plt.plot( MH, f_ex_g_Wthtf/fpbh_ex_g_Wthtf, ':', color='k', label=r'$f_G$, Wthtf')
plt.plot( MH, f_ex_resum_Wthtf/fpbh_ex_resum_Wthtf, color='#781aa5', linestyle=(0, (5, 2, 1, 2, 1, 2)) , label=r'$\beta_{NG}$, Wthtf')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$f(MH)/f_{\rm PBH}$')
plt.xlabel(r'$M_H\,/\,M_\odot$')
plt.legend(loc='center right', bbox_to_anchor=(0.999, 0.81))
plt.title('Extended Mass function' f', g={nn}g$_c$')
plt.ylim(1e-17, 1e1)
plt.xlim(5e-17, 6e-8)
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.92)
plt.yticks(list(plt.yticks()[0]) + [1])
plt.ylim(1e-17, 1e1)
# plt.xlim(3e-14, 6e-9)
# Use FuncFormatter with the custom formatter function
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter_y))
# plt.savefig(f'f_ex_vs_{nn}g_fpbh_{kikf}.pdf')
plt.show()

