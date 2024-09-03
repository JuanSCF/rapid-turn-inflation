import math
import numpy as np
import scipy as sp
#from timeit import default_timer as timer
import time
import matplotlib.pyplot as plt
import cmath
import numpy as np
from scipy.special import j1
import os
#J
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.misc import derivative

import sys
from scipy.signal import savgol_filter
#J
import matplotlib                      
matplotlib.rcParams['text.usetex'] = True
#J
import tqdm as tqdm

# d = 3.75*np.pi
# enh=np.exp(2*d)/(4*d)**2.
# print(f'{enh:.2e}')
# 2*d*10

deltaN = 0.1
# L = 251.327 # -> f_pbh = 2.56e+11 d = 4.00*np.pi
# this L gives kpeak ~ 1.17e13 Mpc^-1
# ESTO DE ARRIBA SOLO PARA Wthtf con deltac=0.41 !!!!
# deltac=0.45 -> fpbh=1.378e-03, deltac=0.5 -> fpbh=1.633e-07

k00 = 1e11 # Mpc^-1
# L = 233.158 # -> mono: fpbh = 33.47% f_peak = 0.997, d = *np.pi
####################
#################################################
L = 233.548 # fpbh_ex = 56.2174%, fpbh_mono = 99.8%, fpeak_ex = 0.6304, fpeak_mono = 2.927, max ps = 1.61e-2
L = 233.758 # fpbh_ex = 99.7919%, fpbh_mono = 176.136%, fpeak_ex = 1.118, fpeak_mono = 5.123, max ps = 1.62e-2
#################################################
####################
# L = 233.5487 # fpbh_ex = 56.3266%, fpbh_mono = 99.9961%, fpeak_ex = 0.6316, fpeak_mono = 2.932, max ps = 1.61e-2
# L = 233.5488 # fpbh_ex = 56.3423%, fpbh_mono = 100.024%, fpeak_ex = 0.6318, fpeak_mono = 2.933, max ps = 1.61e-2
# L = 233.549 # fpbh_ex = 56.37%, fpbh_mono = 100.079%, fpeak_ex = 0.6321, fpeak_mono = 2.935, max ps = 1.61e-2

# L = 233.716 # -> ex: f_pbh = 89.12 fpeak = 0.9983, d = *np.pi
#################################
# L = 233.758 # fpbh_ex = 99.7919%, fpbh_mono = 176.136%, fpeak_ex = 1.118, fpeak_mono = 5.123, max ps = 1.62e-2
#################################
# L = 233.7587 # fpbh_ex = 99.9793%, fpbh_mono = 176.463%, fpeak_ex = 1.12, fpeak_mono = 5.132, max ps = 1.62e-2
# L = 233.75877 # fpbh_ex = 99.9981%, fpbh_mono = 176.496%, fpeak_ex = 1.12, fpeak_mono = 5.133, max ps = 1.62e-2
# L = 233.75878 # fpbh_ex = 100.01%, fpbh_mono = 176.5%, fpeak_ex = 1.12, fpeak_mono = 5.133, max ps = 1.63e-2
# L = 233.759 # fpbh_ex = 100.06%, fpbh_mono = 176.603%, fpeak_ex = 1.121, fpeak_mono = 5.136, max ps = 1.63e-2
#################################
############  Wg4  ##############
#################################
# L = 226.188 # fpbh mono 94.8%, fpbh ex 39.6%
# L = 226.204 # fpbh mono 97.6%, fpbh ex 40.7%
# L = 226.208 # fpbh mono 100.313%, fpbh ex 41.9%
#################################
L = 226.206 ##### # fpbh mono 99.752%, fpbh ex 41.6659%
L = 226.518 ##### #fpbh mono 236.731%, fpbh ex 99.7571%
#################################
# L = 226.519 # fpbh mono 237.38%, fpbh ex 100.033%
#################################
############  Wth  ##############
#################################
L=228.21 # Wth, fpbh mono=99.87%, fpbh ex=56.3%
# L=228.404 # Wth, fpbh mono=17x%, fpbh ex=99.67%
L=228.405 # Wth, fpbh mono=176.2%, fpbh ex=99.98%

# k00 = 1.57e7 # this is for Mpeak~1e-5

# L = 163.363
# k00 = 2.05e4 # this is for Mpeak~1e1
# playing with the values to find lambda0 similar to fnl values
# deltaN = 0.5
# L = 11.0103*2/deltaN # Wg4: this gives fpbh mono=101.1% 
# L = 11.3527*2/deltaN # Wthtf, this gives fpbh mono=99.99%, fpbh ex=56.95% 
# L = 11.3626*2/deltaN # Wthtf, this gives fpbh mono=173.4%, fpbh ex=99.91% 
# deltaN = 1
# L = 11.3626*2/deltaN # Wthtf, this gives fpbh mono=%, fpbh ex=% 
'''this gives k_peak=1.287e13 Mpc^-1'''
size = 4000
nps = 15000

Wf='Wth' # remember to change deltac and W2 in sigmaR2 calc below line~370
#sigma line~400

if Wf=='Wg4' or 'Wth':
  deltac = 0.41
if Wf=='Wg4':
  deltac = 0.18

gamma=0.36
C=4.
OmegaCDM=0.264

cwd = os.getcwd()
fgaussian_data_file = os.path.join(cwd, f'data\\gaussian-data-{Wf}-L-{L}.npz')
# fgaussian_data = np.load(fgaussian_data_file)


#ps cris en funcion de kappa=k/k0
#Pz(K)
def DeltaZetaAnalytic(K):
  a= abs(cmath.exp(2*1j*K*cmath.sinh(deltaN/2.))*(cmath.cos(deltaN*cmath.sqrt(-(L*K) + K**2)) + cmath.cos(deltaN*cmath.sqrt(L*K + K**2)) - (0.5j*(1 - L*K + 2*K**2)*cmath.sin(deltaN*cmath.sqrt(-(L*K) + K**2)))/(K*cmath.sqrt(-(L*K) + K**2)) - (0.5j*(1 + L*K + 2*K**2)*cmath.sin(deltaN*cmath.sqrt(L*K + K**2)))/(K*cmath.sqrt(L*K + K**2))) - (0.5j*cmath.exp(2*1j*K*cmath.cosh(deltaN/2.))*(((L*K - K**2 + (1j + K)**2)*cmath.sin(deltaN*cmath.sqrt(-(L*K) + K**2)))/cmath.sqrt(-(L*K) + K**2) + ((-(L*K) - K**2 + (1j + K)**2)*cmath.sin(deltaN*cmath.sqrt(L*K + K**2)))/cmath.sqrt(L*K + K**2)))/K)**2/8. + abs((cmath.exp(2*1j*K*cmath.sinh(deltaN/2.))*((1 + 2*K**2)*cmath.cos(deltaN*cmath.sqrt(-(L*K) + K**2)) + (-1 - 2*K**2)*cmath.cos(deltaN*cmath.sqrt(L*K + K**2)) - (((1 + 1j*K)*K**2 - (1 - 1j*K)*(-(L*K) + K**2))*cmath.sin(deltaN*cmath.sqrt(-(L*K) + K**2)))/cmath.sqrt(-(L*K) + K**2) + (((1 + 1j*K)*K**2 - (1 - 1j*K)*(L*K + K**2))*cmath.sin(deltaN*cmath.sqrt(L*K + K**2)))/cmath.sqrt(L*K + K**2)) + cmath.exp(2*1j*K*cmath.cosh(deltaN/2.))*((-1 + 2j*K)*cmath.cos(deltaN*cmath.sqrt(-(L*K) + K**2)) + (1 - 2j*K)*cmath.cos(deltaN*cmath.sqrt(L*K + K**2)) + (1 - 1j*K)*((L*K*cmath.sin(deltaN*cmath.sqrt(-(L*K) + K**2)))/cmath.sqrt(-(L*K) + K**2) + (L*K*cmath.sin(deltaN*cmath.sqrt(L*K + K**2)))/cmath.sqrt(L*K + K**2))))/K**2)**2/32.
  return a

# kk = np.geomspace(1e-3, 1e3, 5000)
# pz = [DeltaZetaAnalytic(k) for k in kk]
# pz = np.array(pz)

kpz_f = 3e15
kk = np.geomspace(3e7, kpz_f, nps)
'''aumentar rango de integracion''' #-> disminuye el valor de Mpeak
pz = [DeltaZetaAnalytic(k/k00) for k in kk]
'''computing ps for kappa=k/k0'''
pz = np.array(pz)


# Save the data to a .npy file
cwd = os.getcwd()
# np.save(cwd+'\my_data.npy', [kk, pz])

# Load the data from the .npy file
# kk, pz= np.load('my_data.npy')


# if Wf=='Wg':
#   deltac = 0.18
#   C=1.44
#   C=4
# else:
#   deltac = 0.5
#   C=4.




Nexit=np.log(kk)
Nexit0=np.linspace(np.log(3e-15),np.log(kpz_f), nps)#este 2do range podria tener más finura
#N goes from -34.5 to 34.5

# #plot ps vs e-folds
# plt.plot(Nexit, pz)
# plt.yscale('log')
# #plt.xscale('log')
# plt.show()



# plt.plot(kk, pz)

from matplotlib.ticker import FuncFormatter

Omegam = 0.315 #???
Meq=(2.8)*10**17. #solar masses
keq=0.01*(Omegam/0.31) #Mpc^-1
keq = 0.01


def kofMH(M):
    return keq*(Meq/M)**0.5
def MHofk(k):
    return (keq/k)**2.*Meq
def format_func0(value, tick_number): #this one is for checking if the values of the ticks kz fit with the x values from MH
    return f'{kofMH(value):.1e}'
def format_func(value, tick_number):
    exponent = int(np.log10(kofMH(value)))
    return r'$10^{{{}}}$'.format(exponent)

def format_ticks1(y, pos):
    if y == 0:
        return '0'
    elif abs(y) < 1e2:
        return f'{y:.0e}'
    else:
        return f'{y:.1e}'


# CMBindex=np.argmin(np.abs(Nexit0-(Nexit0[0]+3)))
CMBindex=np.argmin(np.abs(Nexit0-(-3))) 
'''changed cmb index to N=np.log(0.05) ~ -3'''
#Jacopo:
#here to calculate where to put the feat in order to have the peak for a given 
#mass Mfeat(even if in our case the peak will be shifted due to our mechanism)
MHCMB=Meq*(keq/0.05)**2 #solar Masses
#MH=1.7e13*k**-2
#horizon mass at cmb epoch
'''other way to compute MH'''
MHCMB2=17*(1.e6/0.05)**2
#MH=12.8e13*k**-2

NCMB=Nexit0[CMBindex]
#???
# Nfeat=0. #por qué fijar Nfeat=0? para dejarlo en el centro del plot?
'''probar que pasa si hago variar Nfeat, yo lo elijo como Nfeat=ln(k0)'''
#pasan cosas raras, se van haciendo cada vez mas negativos los valores de f. se pierde la cola derecha
#Nfeat=np.log(k00) # hace 
# Nfeat=15.
Nfeat=np.log(k00)
'''changed Nfeat to ln(k0)'''
Mfeat=MHCMB*np.exp(-2*(Nfeat-Nexit[CMBindex]))
featindex=np.argmin(np.abs(Nexit0-Nfeat))

k0=np.exp(Nexit0)

kfeat=k0[int(featindex)]
# print(kfeat)
# sys.exit()
# kfeat=1e11
kCMB=k0[CMBindex]
kn=np.exp(Nexit-NCMB)*0.05 # Mpc^-1 #jacopo: normalized the k of the CMB at the value 0.05
k=kn

kfeatn=kfeat/kCMB*0.05
kfeat=kfeatn
k0=k0/kCMB*0.05
Lk0=np.log(k0)
Lk=np.log(k)
P0 = 2.4*10**-9
P0 = 2.1e-09
P0 = np.ones(len(Nexit0))*P0
P0fk = interp1d(Lk0,P0,bounds_error=False, kind='cubic',fill_value="extrapolate")
#Jacopo:
#here I should import the Power spectrum numerical one with his k , interpolate and then extend to all the k0

norm = 2.1*10**(-9)/np.exp(P0[CMBindex])
norm = 10**(-9)/np.exp(P0[CMBindex])

#Numerics,

##############
#ver que pasa si saco el savgol_filter
############
P = savgol_filter(np.log(pz), 31, 3)
# P= np.log(pz) #js 
Pfk=interp1d(Lk,P,bounds_error=False, kind='cubic', fill_value="extrapolate")
nn = np.exp(P0fk(Lk[0]))/np.exp(Pfk(Lk[0]))  #Jacopo: #you can do this only if the rescaling happens in a region where there is no enhancement

###############################################

#Jacopo:
#Here I extend the power spectrum to the whole domain k0 and I normalize it
#########

PfkE=np.zeros(len(k0))
for i in range (0,len(k0)):
	if Lk0[i] < Lk[0] or Lk0[i] >= Lk[-1]:
		PfkE[i]=norm*np.exp(P0fk(Lk0[i]))#*norm
	else:
		PfkE[i]=np.exp(Pfk(Lk0[i]))*norm *nn
	if Lk0[i] >= Lk[-1]:
		PfkE[i]= np.exp(P0fk(Lk0[i]))*norm*nn*np.exp(Pfk(Lk[-1]))/np.exp(P0fk(Lk[-1]))
	else:
		pass
  
# plt.plot(Lk0,PfkE)
# plt.show() # este muestra el plot parecido a un pulso
# plt.plot(Lk0,PfkE)
# plt.xlim(24,32)
# plt.show() #este muestra el plot parecido a un pulso pero con un zoom en el eje x

Pfk = interp1d(Lk0, PfkE, bounds_error=False, kind='cubic', fill_value="extrapolate")
PfkL = interp1d(Lk0, np.log(PfkE), bounds_error=False, kind='cubic', fill_value="extrapolate")

# plt.plot(Lk0,PfkE,'b')
# plt.show()

# plt.plot(Lk0, np.exp(PfkL(Lk0)),'r')
# plt.yscale('log')
# #plt.xscale('log')
# plt.show()

q0 = np.geomspace(2.7, 3e15, 15000)
Lq0 = np.log(q0)
Pq0 = Pfk(np.log(q0))
# np.savez('Pq0.npz', q0=q0, Pq0=Pq0)

# sys.exit()
# plt.plot(k0,Pfk(Lk0),'b')
# #plt.plot(kr,np.exp(Pappr(Lkr)),'--')
# plt.plot(k0,np.exp(PfkL(Lk0)),'r')
# plt.plot(k0,np.exp(P0)*norm,'--')
# plt.plot(k0,PfkE, color = 'g',linestyle = '--')
# plt.yscale('log')
# plt.xscale('log')
# plt.show()

# Acmb = 2.0989031673191437e-9
Acmb = 2.1e-9
ns = 0.9649
Kpivot = 0.002 # (*Mpc^-1*)
cmbPS = Acmb*(k0/Kpivot)**(ns-1)
# plt.plot(k0,np.exp(P0)*norm,'--',color='r')
#plt.plot(kr,np.exp(Pappr(Lkr)),'--')
# plt.plot(k0,np.exp(PfkL(Lk0)),'r')
# plt.plot(k0,PfkE, color = 'g',linestyle = '--')
plt.plot(k0,Pfk(Lk0), label=r'$P_{\rm \mathcal R}$')
plt.plot(k0,cmbPS,'r--', label=r'$P_{\rm CMB}$')
plt.axhline(y=1e-2, color='purple', linewidth=1)
plt.yscale('log')
plt.xscale('log')
plt.xlim(1e-5,1e15)
plt.xlim(1e7,1e15)
ylims = plt.gca().get_ylim()
plt.fill_betweenx([1e-2, 1], 1, 1e15, color='purple', alpha=0.27)
plt.ylim(ylims)
plt.xlabel(r'$k\,\left[Mpc^{-1} \right]$')
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.897))
# plt.savefig('Pz-Pcmb.pdf')
plt.show()


#################
##########################################################################


Ppeak=np.amax(Pfk(Lk))
ppeak=np.argmin(np.abs(Pfk(Lk)-Ppeak))
kpeak=k[int(ppeak)]

Npeak=Nexit[int(ppeak)]

print('00')

plt.figure(00)
plt.plot(Nexit,Pfk(Lk),'--')
plt.axvline(x=Npeak)
plt.yscale('log')
# plt.show()

# #Jacopo:
# # with ax.twiny the two axis are sincronyzed but not perfectly, therefore if you want to 
# ju=np.amax(Pfk(Lk0)) # creo k no c usa
# fig, ax=LMH plt.subplots()
# plt.title(r'$\mathrm{Power\,\, Spectrum}$',fontsize=16)
# ax2=ax.twiny()
# ax.plot(Nexit0-NCMB, np.exp(P0)*norm, label='$P_{\zeta 0}$')
# ax.plot(Nexit0-NCMB,Pfk(Lk0),'g', label='$P_{\zeta}$')
# #
# ax.axvline(x=Npeak-NCMB, linewidth=1, color='k', linestyle='--')
# ax.set_xticks([0,10,20,Nfeat-NCMB,Npeak-NCMB,40,50,Nexit0[-1]-NCMB])
# ax.set_xticklabels(['$N_*=0$', 10, 20, '$N_{f}$', '$N_{p}$', 40, 50, '$N_{end}$'])
# ax2.plot(k0,0*np.exp(P0)*norm)
# ax2.set(ylabel='', xlabel='$k\,(Mpc^{-1})$') 
# ax.tick_params(axis="x", labelsize=14)
# ax.tick_params(axis="y", labelsize=14)
# ax2.tick_params(axis="x", labelsize=14) 
# ax2.set_xscale('log')
# plt.yscale('log')
# ax.legend(fontsize=16)
# plt.tight_layout()
# plt.show()

# print(Npeak,Nfeat)
#######################################
#################################################
############################################################
################  integration time  #############################
############################################################
#################################################
#######################################

print('Integration 1')

def Intarray(f, array):
  array
  S=[]
  for i in range(1, len(array)):
    S.append((array[i]-array[i-1])*0.5*(f[i]+f[i-1]))
  return np.sum(S)
  


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

#Jacopo:
# those are really useful when you use the full powerspectum to zoom on the relevant part (if you use the approx most of the time they are not doing anythin)
inM=np.argmin(np.abs(10**10*kfeat-k0))
inm=np.argmin(np.abs(10**(-5)*kfeat-k0))
inz=np.argmin(np.abs(10**(8.)*kfeat-k0))


q= k0[:inM]
kz=k0[inm:inz]
nq= len(q)

# aqui puedo optimizar memoria al no utilizar np.geomspace. buscar iteraor o algo
kzz=np.geomspace(kz[0],kz[-1], size)
kz=kzz
nkz=len(kz)
Lq =np.log(q)
W2 = np.zeros([nkz,nq])
integrand = np.zeros([nkz,nq])
integrandb = np.zeros([nkz,nq])
integrandc = np.zeros([nkz,nq])
sigmaR2 = np.zeros(nkz)
sigmaR2r=[]

Mz=(keq/kz)**2. * Meq
csrad=np.sqrt(1./3.)
'''
podria utilizar Mz en vez de kz y ver que pasa con la varianza
'''
for i in tqdm.tqdm(range(0, nkz)):
    # W2[i,:] = np.exp(-0.5*(q[:]/keq)**2*(Mz[i]/Meq)) # Gaussian 4, M as input
    # wg4
    # W2[i,:] = np.exp(-0.25*(q[:]/kz[i])**2)# Gaussian 4
    # W2[i,:] = np.exp(-0.25*(q[:]/kz[i])**2)*(3.*(np.sin(csrad*q[:]/kz[i])-csrad*q[:]/kz[i]*np.cos(csrad*q[:]/kz[i]))/(csrad*q[:]/kz[i])**3.) # Gaussian 4 + transfer
    # top-hat
    W2[i,:] = 3.*(np.sin(q[:]/kz[i])-q[:]/kz[i]*np.cos(q[:]/kz[i]))/(q[:]/kz[i])**3. # top-hat
    # wthtf
    # W2[i,:] = (3.*(np.sin(q[:]/kz[i])-q[:]/kz[i]*np.cos(q[:]/kz[i]))/(q[:]/kz[i])**3.)*(3.*(np.sin(csrad*q[:]/kz[i])-csrad*q[:]/kz[i]*np.cos(csrad*q[:]/kz[i]))/(csrad*q[:]/kz[i])**3.)
    W2[i,:] = (W2[i,:])**2
    integrand[i,:] = (16./81.)*Pfk(Lq[:])*W2[i,:]*(q[:])**3./(keq)**4. *(Mz[i]/Meq)**2
    sigmaR2[i]=Intarray_vec(integrand[i,:]*q[:],Lq[:])

kcmb=np.geomspace(1e-4,1e-1,1000) # Mpc^-1
cmbPS_short = Acmb*(kcmb/Kpivot)**(ns-1)
varcmb = Intarray_vec(cmbPS_short,np.log(kcmb))

plt.figure(2)
plt.plot(kz,sigmaR2,'k', label='$\sigma^2(k)$')
plt.plot(kcmb,varcmb*np.ones_like(kcmb),'red', label='$\sigma^2_{\\rm cmb}$')
plt.legend(fontsize=15)
plt.yscale('log')
plt.xscale('log')
plt.title(f'variance, Wf={Wf}, L={L}')
plt.show()

plt.figure(2)
plt.plot(Mz,sigmaR2,'k', label='$\sigma^2(M)$')
plt.legend(fontsize=15)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$M/M_\odot$')
plt.show()

MH = Meq*(keq/kz)**2
sigmapeak=np.amax(sigmaR2)
mhpeak=np.argmin(np.abs(sigmaR2-sigmapeak))
MHp=MH[int(mhpeak)]

mzpeak=np.argmin(np.abs(sigmaR2-sigmapeak))
Mzp=MH[int(mzpeak)]

# plt.figure(11)
# plt.plot(MH,sigmaR2,'k', label='$\sigma^2(MH)$')
# plt.legend(fontsize=15)
# plt.axvline(x=MHp)
# plt.plot(Mz,sigmaR2,color='orange', label='$\sigma^2(Mz)$')
# plt.axvline(x=Mzp,color='orange')
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel(r'$M/M_\odot$')
# # plt.ylabel('Y-axis Label')
# plt.show()

# plt.figure(11)
# plt.plot(Mz,sigmaR2,color='orange', label='$\sigma^2(Mz)$')
# plt.legend(fontsize=15)
# plt.axvline(x=Mzp,color='orange')
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel(r'$M/M_\odot$')
# # plt.ylabel('Y-axis Label')
# plt.show()

print('MHpeak=', MHp)

#############################
#################################
# #save variance
# # Save the data to a .npy file
# cwd = os.getcwd()
# np.save(cwd+'\\variance_data.npy', [q, kz, sigmaR2])
# np.savez(cwd+'\\bs\\variance_data_Wg.npz', [q, kz, sigmaR2])
#################################
#############################

print('Integration 2')

in2=np.argmin(np.abs(kfeat-kz))
MH = Meq*(keq/kz)**2

LMH = np.log(MH)
# # C=3.3 #4
# C=4
# gamma=0.36
# # deltac = 0.45 # 0.5, 0.55
# deltac= 0.5
# OmegaCDM=0.264
# # valores k y deltac, 2303.05248 1904.00984


# esta parte de abajo es memoria optimizable al no utilizar np.geomspace
M = np.geomspace(MH[-1], 10**(-1)*MH[0], size) # why compute M in this way?
mu = np.zeros([nkz, size])
# esta parte de abajo es memoria optimizable al no utilizar np.zeros([nkz,size])
Integrand2=np.zeros([nkz,size])
Integrand2b=np.zeros([nkz,size])
Integrand2d=np.zeros([nkz,size])
Integrand2c=np.zeros([nkz,size])
# esta parte de abajo es memoria optimizable al no utilizar np.zeros(size]
f=np.zeros(size)
fb=np.zeros(size)
fc=np.zeros(size)
fd=np.zeros(size)
f1=np.zeros(size)
kzr=[]

for j in range(0, size):
  mu[:,j]=(M[j]/(C*MH))
  Integrand2[:,j]=-2/(OmegaCDM)/(np.sqrt(np.pi*2*sigmaR2[:]))*np.exp(-(mu[:,j]**(1./gamma)+deltac)**2/(2*sigmaR2[:]))*M[j]/MH[:]*(1./gamma)*(M[j]/(C*MH[:]))**(1./gamma)*np.sqrt(Meq/MH[:])
  f[j]=Intarray_vec(Integrand2[:,j],LMH[:])
for i in range (0,nkz):    #this one is to check which k and sigma values are relevant for the computations
		if np.amax(-Integrand2[i,:]) > 0: #if at least one term in the integral is different from zero I keep that value of k and sigma
			sigmaR2r.append(sigmaR2[i])
			kzr.append(kz[i])
		else:
			pass
	#	f1[j]=np.trapz(Integrand2[:,j],LMH[:])
LM=np.log(M)

# ############################################
# f(M) monochromatic calculation
# ############################################
frac=1. # 0.2
betamono=2*frac*sigmaR2/deltac/np.sqrt(2*np.pi*sigmaR2)*np.exp(-deltac**2/(2*sigmaR2))
fmono=1./OmegaCDM*np.sqrt(Meq/(frac*MH))*betamono
Mpbh = frac*MH
plt.axhline(y=1,color='r')
plt.plot(Mpbh,fmono,label='$f(M)$')
# plt.plot(MH,fmono/Intarray_vec(-fmono, LMH), label='$f(M)$')
plt.yscale('log')
plt.xscale('log')
plt.title('f mono',fontsize=16)
# plt.ylim(1e-10, 1e15)
# plt.ylim(1e-20, 3)
plt.show()
# ############################################

# ############################################
# ############################################
# #save f(M)
# # Save the data to a .npy file
# cwd = os.getcwd()
# np.save(cwd+'\\f(M)_data.npy', [M, f])
# np.save(cwd+'\\bs\\f(M)_data.npy', [M, f])
# ############################################
# ############################################


#Ppeak=np.amax(Pfk(k))
#ppeak=np.argmin(np.abs(Pfk(k)-Ppeak))
#kpeak=k[int(ppeak)]
Lkzr=np.log(kzr)
plt.figure(22)
plt.plot(k,Pfk(Lk),'y',label='$P_z$')
plt.plot(kzr,Pfk(Lkzr),'k,--',label='relevant k',linewidth=1.3)
# plt.axvline(x=kpeak)
plt.plot(kz,sigmaR2,'r',label='$\sigma_R$')
plt.plot(kzr,sigmaR2r,'k,--',linewidth=1.3)
#plt.plot(M,f1,'--')
plt.legend(fontsize=10)
plt.yscale('log')
plt.xscale('log')
plt.axhline(y=0.01)
plt.ylim(1e-12, 1e-1)
plt.show()

fpeak=np.amax(f)
mpeak=np.argmin(np.abs(f-fpeak))
Mp=M[int(mpeak)]

plt.figure(3)
plt.plot(M,f,'o',label='$f(M)$')
plt.axvline(x=Mp)  #the PBH mass corresponding to the peak 
plt.axvline(x=Mfeat) #the horizon mass corresponding to the feature
#plt.plot(M,f1,'--')
plt.legend(fontsize=10)
plt.yscale('log')
plt.xscale('log')
plt.show()


OmegaPBH=Intarray_vec(f,LM)
print ('Omega_{PBH}=', OmegaPBH)

# plt.figure(4)
# plt.plot(M,f/OmegaPBH,'o',label='$f(M)$')
# plt.axvline(x=Mp)  #the PBH mass corresponding to the peak 
# plt.axvline(x=Mfeat) #the horizon mass corresponding to the feature
# #plt.plot(M,f1,'--')
# plt.legend(fontsize=10)
# # Limiting the plot
# plt.xlim(1e-17, 1e-8)  # Set limits for x-axis
# plt.ylim(1e-16, 5)  # Set limits for y-axis
# plt.yscale('log')
# plt.xscale('log')
# plt.title(r'$f(M)/f_{\rm PBH}$',fontsize=16)
# plt.show()

# xpeak=np.log(10.**-12)
# #evaluate f(M=xpeak)
# xpeakindex=np.argmin(np.abs(LM-xpeak))
# Mpeak=M[int(xpeakindex)]


# plt.figure(5)
# plt.plot(M,f/fpeak,'o',label='$f(M)$')
# plt.axvline(x=Mp)  #the PBH mass corresponding to the peak 
# plt.axvline(x=Mfeat) #the horizon mass corresponding to the feature
# #plt.plot(M,f1,'--')
# plt.legend(fontsize=10)
# # Limiting the plot
# plt.xlim(1e-17, 1e-8)  # Set limits for x-axis
# plt.ylim(1e-16, 5)  # Set limits for y-axis

# plt.title(r'$f(M)/f_{peak}$',fontsize=16)
# plt.yscale('log')
# plt.xscale('log')
# print('Mpeak=',Mp)
# plt.show()

# betaex=np.zeros(size)
# Integrandbeta=np.zeros([nkz,size])
# for j in range(0, size):
#   mu[:,j]=(M[j]/(C*MH))
#   Integrandbeta[:,j]=-2/(np.sqrt(np.pi*2*sigmaR2[:]))*np.exp(-(mu[:,j]**(1./gamma)+deltac)**2/(2*sigmaR2[:]))*M[j]/MH[:]*(1./gamma)*(M[j]/(C*MH[:]))**(1./gamma) 
#   betaex[j]=Intarray_vec(Integrand2[:,j],LM[:])

############# save data ################
# np.savez(fgaussian_data_file, kz=kz, sigmaR2=sigmaR2, kpz=k0, pz=PfkE, fmono=fmono, fex=f)


plt.plot(MH,fmono,'o', label='$fmono$')
plt.plot(M,f,'o', label='$f(M)$')
plt.yscale('log')
plt.xscale('log')
plt.title('f mono vs extended', fontsize=16)
plt.legend()
plt.show()

fpbhmono=-Intarray_vec(fmono,LMH)
fpbhex = Intarray_vec(f,LM)

# plt.plot(MH,fmono/fpbhmono, label='Monochromatic')
# plt.plot(MH, fmono_pnorm, '-.', label='Monochromatic')
# plt.plot(M, fex_pnorm, label='Extended')
plt.plot(MH, fmono, '-.', color='tab:blue', label='Monochromatic')
plt.plot(M, f, color='tab:orange',label='Extended')
# plt.plot(M, wthtf_f, color='tab:purple', label='Extended2')
plt.yscale('log')
plt.xscale('log')
plt.title(f'PBH Mass Function Gaussian Statistics {Wf}', fontsize=16)
plt.legend()
plt.xlabel(r'$M/M_\odot$')
plt.ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
plt.ylim(1e-64, 10)
plt.xlim(1e-17, 4e-9)
plt.ylim(1e-48, 1e2)
plt.xlim(1e-17, 1e-10)
plt.xlim(1e-17, 1e1)
# plt.axvline(x=MH[np.argmax(fmono_pnorm)], linestyle='-.')
# plt.axvline(x=M[np.argmax(fex_pnorm)])
# plt.yticks(list(plt.yticks()[0]) + [1])
# plt.savefig(f'C:\ZZZ\Laburos\Codes\\figss\Gaussian-f\\both\\Gaussian-f(M)-{Wf}-deltac-{deltac}.png')
# plt.savefig(f'C:\ZZZ\Laburos\Codes\\figss\Gaussian-f\\both\\Gaussian-f(M)-{Wf}-deltac-{deltac}.svg')
plt.show()




fig, ax1 = plt.subplots()
ax1.plot(MH, fmono,'--',  color='tab:blue', label=r'Monochromatic, $\lambda_0=$'f'{L}')
ax1.plot(M, f, color='tab:orange', label=r'Extended, $\lambda_0=$'f'{L}')
ax1.set_title(f'PBH Mass Function, Gaussian Statistics. {Wf}', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.set_ylim(1e-48, 1e2)
ax1.set_xlim(1e-17, 1e1)
ax1.legend(loc='lower left')
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
secax.set_xlim(ax1.get_xlim())
major_ticks = np.geomspace(MH[-1], MH[0], 10)  # Adjust the number of major ticks as needed
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
ax1.set_ylim(1e-48, 1e2)
ax1.set_xlim(1e-18, 1e-10)
plt.tight_layout()
# plt.savefig(f'fmonoex_{Wf}.pdf')
plt.show()

# wthtf_var = sigmaR2
# wth_var = sigmaR2
# wg_var = sigmaR2
# wgc4_var = sigmaR2
if Wf=='Wthtf' and L==233.548:
  wthtf_fmono = fmono
  wthtf_Mpmono = MH[np.argmax(fmono)]
elif Wf=='Wthtf' and L==233.758:
  wthtf_f = f
  wthtf_Mp = M[np.argmax(f)]
  wthtf_var = sigmaR2
elif Wf=='Wg4' and L==226.206:
  wg4_fmono = fmono
  wg4_Mpmono = MH[np.argmax(fmono)]
elif Wf=='Wg4' and L==226.518:
  wg4_f = f
  wg4_Mp = M[np.argmax(f)]
  wg4_var = sigmaR2
elif Wf=='Wth' and L==228.21:
  wth_fmono = fmono
  wth_Mpmono = M[np.argmax(f)]
elif Wf=='Wth' and L==228.405:  
  wth_f = f
  wth_Mp = MH[np.argmax(fmono)]
  wth_var = sigmaR2

# np.savez('fmono_data', MH=MH, wthtf_fmono=wthtf_fmono, wth_fmono=wth_fmono,wg4_fmono=wg4_fmono)
# np.savez('fex_data', M=M, wthtf_fex=wthtf_f,wth_fex=wth_f, wg4_fex=wg4_f)

fpbhmono = -Intarray_vec(fmono,LMH)
fpbhex = Intarray_vec(f,LM)
print(f'L = {L} \n size = {size} \n nps = {nps}')
print(f' ')
print (r'$f_{PBH,\, mono}$ =', f'{100*fpbhmono:.6} %')
print (r'$f_{PBH,\, ex}$ =', f'{100*fpbhex:.6} %, deltac = {deltac}')
 

print (r'$f_{mono peak}$ =', f'{max(fmono):.4}')
print (r'$f_{ex peak}$ =', f'{max(f):.4}')

print(f'Max PS = {np.max(PfkE):.2e}')

sys.exit()

# celeste #00ABEB morado: #8C2EB9
plt.plot(MH, wth_fmono, color='tab:blue', label=r'Wth, $\lambda_0=228.210$')
plt.plot(MH, wthtf_fmono, '--', color='tab:orange', label=r'Wthtf, $\lambda_0=233.548$')
plt.plot(MH, wg4_fmono, linestyle=(0, (5, 2, 1, 2, 1, 2)),color='k', linewidth=1,label=r'Wg4, $\lambda_0=226.206$')
plt.yscale('log')
plt.xscale('log')
plt.title(f'Monochromatic PBH Mass Function, Gaussian Statistics')#, fontsize=16)
plt.legend(loc='lower right')
plt.xlabel(r'$M\,[M_\odot]$')
plt.ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
plt.ylim(1e-48, 1e2)
plt.xlim(1e-17, 1e1)
# plt.yticks(list(plt.yticks()[0]) + [1])
plt.tight_layout()
# plt.savefig('fmono.pdf')
plt.show()


# linestyle=(0, (5, 2, 1, 2, 1, 2))
plt.plot(MH, wth_fmono, '--', color='tab:blue', label=r'Wth, $\lambda_0=228.210$')
plt.plot(MH, wthtf_fmono, color='tab:orange', label=r'Wthtf, $\lambda_0=233.548$')
plt.plot(MH, wg4_fmono, '-.',color='k', linewidth=1,label=r'Wg4, $\lambda_0=226.206$')
plt.yscale('log')
plt.xscale('log')
plt.title(f'Monochromatic PBH Mass Function, Gaussian Statistics')#, fontsize=16)
plt.legend(loc='lower right')
plt.xlabel(r'$M\,[M_\odot]$')
plt.ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
plt.ylim(1e-48, 1e2)
plt.xlim(1e-17, 1e1)
# plt.yticks(list(plt.yticks()[0]) + [1])
plt.tight_layout()
# plt.savefig('fmono.pdf')
plt.show()


plt.plot(M, wth_f, color='tab:blue', label=r'Wth, $\lambda_0=228.405$')
plt.plot(M, wthtf_f, '--',color='tab:orange', label=r'Wthtf, $\lambda_0=233.758$')
plt.plot(M, wg4_f, linestyle=(0, (5, 2, 1, 2, 1, 2)),color='k', linewidth=1,label=r'Wg4, $\lambda_0=226.518$')
plt.yscale('log')
plt.xscale('log')
plt.title(f'Extended PBH Mass Function, Gaussian Statistics')#, fontsize=16)
plt.legend(loc='lower right')
plt.xlabel(r'$M\,[M_\odot]$')
plt.ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
plt.ylim(1e-48, 1e2)
plt.xlim(1e-17, 1e1)
# plt.yticks(list(plt.yticks()[0]) + [1])
plt.tight_layout()
# plt.savefig('fex.pdf')
plt.show()


plt.plot(MH, wth_var, color='tab:blue', label=r'Wth')
plt.plot(MH, wthtf_var, '--',color='tab:orange', label=r'Wthtf')
plt.plot(MH, wg4_var, linestyle=(0, (5, 2, 1, 2, 1, 2)),color='k', linewidth=1,label=r'Gaussian')
plt.yscale('log')
plt.xscale('log')
plt.title(f'Variance of the Overdensity')#, fontsize=16)
plt.legend(loc='lower right')
plt.xlabel(r'$M\,[M_\odot]$')
plt.ylabel(r'$\sigma^2$')
# plt.axhline(y=1, color='r', linestyle='--')
plt.ylim(1e-12, 1e-1)
plt.xlim(1e-17, 1e1)
# plt.yticks(list(plt.yticks()[0]) + [1])
plt.tight_layout()
# plt.savefig('var.pdf')
plt.show()





MH2=MH
fmono2=fmono
M2=M
f2=f
L2=L

plt.plot(MH, fmono,'-.',  color='tab:blue', label='Monochromatic')
plt.plot(M, f, color='tab:orange', label='Extended')
plt.yscale('log')
plt.xscale('log')
plt.title(f'PBH Mass Function, Gaussian Statistics. {Wf}')#, fontsize=16)
plt.legend()
plt.xlabel(r'$M/M_\odot$')
plt.ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
plt.ylim(1e-48, 1e2)
plt.xlim(1e-17, 1e1)
plt.legend(loc='lower right')
plt.tight_layout()
# plt.savefig('fmonoex.pdf')
plt.show()




####### plot con ticks


# linestyle=(0, (5, 2, 1, 2, 1, 2))


fig, ax1 = plt.subplots()
ax1.plot(MH, wth_fmono, '--', color='tab:blue', label=r'Wth, $\lambda_0=228.210$')
ax1.plot(MH, wthtf_fmono, color='tab:orange', label=r'Wthtf, $\lambda_0=233.548$')
ax1.plot(MH, wg4_fmono, '-.',color='k', linewidth=1,label=r'Wg4, $\lambda_0=226.206$')
ax1.set_title(f'Monochromatic PBH Mass Function, Gaussian Statistics', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.set_ylim(1e-48, 1e2)
ax1.set_xlim(1e-17, 1e1)
ax1.legend(loc='lower right')
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
secax.set_xlim(ax1.get_xlim())
major_ticks = np.geomspace(MH[-1], MH[0], 10)  # Adjust the number of major ticks as needed
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
ax1.set_ylim(1e-48, 1e2)
ax1.set_xlim(1e-17, 1e1)
plt.tight_layout()
# plt.savefig('fmono.pdf')
plt.show()


fig, ax1 = plt.subplots()
ax1.plot(M, wth_f, '--',color='tab:blue', label=r'Wth, $\lambda_0=228.405$')
ax1.plot(M, wthtf_f, color='tab:orange', label=r'Wthtf, $\lambda_0=233.758$')
# ax1.plot(M, wg4_f, linestyle=(0, (5, 2, 1, 2, 1, 2)),color='k', linewidth=1,label=r'Wg4, $\lambda_0=226.518$')
ax1.plot(M, wg4_f, '-.',color='k', linewidth=1,label=r'Wg4, $\lambda_0=226.518$')
ax1.set_title(f'Extended PBH Mass Function, Gaussian Statistics', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.set_ylim(1e-48, 1e2)
ax1.set_xlim(1e-18, 1e1)
ax1.legend(loc='lower right')
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
secax.set_xlim(ax1.get_xlim())
major_ticks = np.geomspace(MH[-1], MH[0], 10)  # Adjust the number of major ticks as needed
# major_ticks = np.geomspace(ax1.get_xlim()[1], ax1.get_xlim()[0], 5)  # Adjust the number of major ticks as needed
# major_ticks = [MHofk(1e15),MHofk(1e14),MHofk(1e13),MHofk(1e12)]  # Adjust the number of major ticks as needed
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
# plt.savefig('fex.pdf')
plt.show()




fig, ax1 = plt.subplots()
ax1.plot(MH, wth_var, '--',color='tab:blue', label=r'Wth, $\lambda_0=228.405$')
ax1.plot(MH, wthtf_var, color='tab:orange', label=r'Wthtf, $\lambda_0=233.758$')
ax1.plot(MH, wg4_var, '-.',color='k', linewidth=1.33,label=r'Wg4, $\lambda_0=226.518$')
ax1.set_title(f'Smoothed Overdensity Variance', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$\sigma^2$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.set_ylim(1e-12, 1e-1)
ax1.set_xlim(1e-17, 1e1)
ax1.legend(loc='lower right')
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
secax.set_xlim(ax1.get_xlim())
major_ticks = np.geomspace(MH[-1], MH[0], 10)  # Adjust the number of major ticks as needed
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
ax1.set_ylim(1e-12, 1e-1)
ax1.set_xlim(1e-17, 1e1)
plt.tight_layout()
plt.savefig('var.pdf')
plt.show()


fig, ax1 = plt.subplots()
ax1.plot(MH, var_th, '--',color='tab:blue', label=r'Wth, $\lambda_0=230$')
ax1.plot(MH, var_tf, color='tab:orange', label=r'Wthtf, $\lambda_0=230$')
ax1.plot(MH, var_g4, '-.',color='k', linewidth=1.33,label=r'Wg4, $\lambda_0=230$')
ax1.set_title('Smoothed Overdensity Variance', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$\sigma^2$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.set_ylim(1e-12, 1e-1)
ax1.set_xlim(1e-17, 1e1)
ax1.legend(loc='lower right')
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
secax.set_xlim(ax1.get_xlim())
major_ticks = np.geomspace(MH[-1], MH[0], 10)  # Adjust the number of major ticks as needed
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
ax1.set_ylim(1e-12, 1e-1)
ax1.set_xlim(1e-17, 1e1)
plt.tight_layout()
plt.savefig('var2.pdf')
plt.show()





fig, ax1 = plt.subplots()
ax1.plot(MH, wg4_fmono,'--',color='#8C2EB9', label=r'Monochromatic Wg4, $\lambda_0=226.206$')
ax1.plot(MH, wthtf_fmono, color='k', linestyle=(0, (5, 2, 1, 2, 1, 2)), label=r'Monochromatic Wthtf, $\lambda_0=233.548$')
ax1.plot(M, wg4_f, '-.', color='tab:orange', label=r'Extended Wg4, $\lambda_0=226.518$')
ax1.plot(M, wthtf_f, color='#00ABEB', label=r'Extended Wthtf, $\lambda_0=233.758$')
ax1.set_title(f'PBH Mass Function, Gaussian Statistics. Wthtf', pad=8)#, fontsize=16)
ax1.set_xlabel(r'$M\, [M_\odot]$')
ax1.set_ylabel(r'$f(M)$')
# plt.axhline(y=1, color='r', linestyle='--')
ax1.set_ylim(1e-48, 1e2)
ax1.set_xlim(1e-17, 1e-10)
ax1.legend(loc='lower left')
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.set_legend()
# yticks = list(ax1.get_yticks())
# yticks.append(0)
# ax1.set_yticks(yticks)
# Create a secondary x-axis
secax = ax1.twiny()
# Set the tick positions and labels for the secondary x-axis
secax.set_xscale('log')
secax.set_xlim(ax1.get_xlim())
# major_ticks = np.geomspace(ax1.get_xlim()[1], ax1.get_xlim()[0], 5)  # Adjust the number of major ticks as needed
major_ticks = [MHofk(1e15),MHofk(1e14),MHofk(1e13),MHofk(1e12)]  # Adjust the number of major ticks as needed
# Use the conversion function to set the secondary x-axis tick labels
secax.set_xticks(major_ticks)
secax.set_xticklabels([format_func(tick, i) for i, tick in enumerate(major_ticks)])
secax.set_xlabel(r'k [Mpc$^{-1}$]')
# formatter = FuncFormatter(format_ticks1)
# plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
# plt.savefig('fmonoex_g.pdf')
plt.show()
