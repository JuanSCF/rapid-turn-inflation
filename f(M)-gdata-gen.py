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

deltaN = 0.1
L = 251.327
k00 = 1e11 # Mpc^-1
'''this gives k_peak=1.287e13 Mpc^-1'''

#ps cris en funcion de kappa=k/k0
#Pz(K)
def DeltaZetaAnalytic(K):
  a= abs(cmath.exp(2*1j*K*cmath.sinh(deltaN/2.))*(cmath.cos(deltaN*cmath.sqrt(-(L*K) + K**2)) + cmath.cos(deltaN*cmath.sqrt(L*K + K**2)) - (0.5j*(1 - L*K + 2*K**2)*cmath.sin(deltaN*cmath.sqrt(-(L*K) + K**2)))/(K*cmath.sqrt(-(L*K) + K**2)) - (0.5j*(1 + L*K + 2*K**2)*cmath.sin(deltaN*cmath.sqrt(L*K + K**2)))/(K*cmath.sqrt(L*K + K**2))) - (0.5j*cmath.exp(2*1j*K*cmath.cosh(deltaN/2.))*(((L*K - K**2 + (1j + K)**2)*cmath.sin(deltaN*cmath.sqrt(-(L*K) + K**2)))/cmath.sqrt(-(L*K) + K**2) + ((-(L*K) - K**2 + (1j + K)**2)*cmath.sin(deltaN*cmath.sqrt(L*K + K**2)))/cmath.sqrt(L*K + K**2)))/K)**2/8. + abs((cmath.exp(2*1j*K*cmath.sinh(deltaN/2.))*((1 + 2*K**2)*cmath.cos(deltaN*cmath.sqrt(-(L*K) + K**2)) + (-1 - 2*K**2)*cmath.cos(deltaN*cmath.sqrt(L*K + K**2)) - (((1 + 1j*K)*K**2 - (1 - 1j*K)*(-(L*K) + K**2))*cmath.sin(deltaN*cmath.sqrt(-(L*K) + K**2)))/cmath.sqrt(-(L*K) + K**2) + (((1 + 1j*K)*K**2 - (1 - 1j*K)*(L*K + K**2))*cmath.sin(deltaN*cmath.sqrt(L*K + K**2)))/cmath.sqrt(L*K + K**2)) + cmath.exp(2*1j*K*cmath.cosh(deltaN/2.))*((-1 + 2j*K)*cmath.cos(deltaN*cmath.sqrt(-(L*K) + K**2)) + (1 - 2j*K)*cmath.cos(deltaN*cmath.sqrt(L*K + K**2)) + (1 - 1j*K)*((L*K*cmath.sin(deltaN*cmath.sqrt(-(L*K) + K**2)))/cmath.sqrt(-(L*K) + K**2) + (L*K*cmath.sin(deltaN*cmath.sqrt(L*K + K**2)))/cmath.sqrt(L*K + K**2))))/K**2)**2/32.
  return a

# kk = np.geomspace(1e-3, 1e3, 5000)
# pz = [DeltaZetaAnalytic(k) for k in kk]
# pz = np.array(pz)

kk = np.geomspace(1e10, 1e15, 15000)
'''aumentar rango de integracion''' #-> disminuye el valor de Mpeak
pz = [DeltaZetaAnalytic(k/k00) for k in kk]
'''computing ps for kappa=k/k0'''
pz = np.array(pz)


# Save the data to a .npy file
cwd = os.getcwd()
# np.save(cwd+'\my_data.npy', [kk, pz])

# Load the data from the .npy file
# kk, pz= np.load('my_data.npy')

Wf='Wthtf'

gamma=0.36
if Wf=='Wg':
  deltac = 0.18
  C=1.44
else:
  deltac = 0.5
  C=4.
OmegaCDM=0.264



Nexit=np.log(kk)
Nexit0=np.linspace(np.log(1e-15),np.log(1e15),15000)#este 2do range podria tener más finura
#N goes from -34.5 to 34.5

#plot ps vs e-folds
plt.plot(Nexit, pz)
plt.yscale('log')
#plt.xscale('log')
plt.show()



# plt.plot(kk, pz)

Omegam = 0.315 #???
Meq=(2.8)*10**17. #solar masses
keq=0.01*(Omegam/0.31) #Mpc^-1

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
P0=2.4*10**-9
P0=np.ones(len(Nexit0))*P0
P0fk=interp1d(Lk0,P0,bounds_error=False, kind='cubic',fill_value="extrapolate")
#Jacopo:
#here I should import the Power spectrum numerical one with his k , interpolate and then extend to all the k0

norm=10**(-9)/np.exp(P0[CMBindex])

#Numerics,

##############
#ver que pasa si saco el savgol_filter
############
P= savgol_filter(np.log(pz), 31, 3)
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
		PfkE[i]=np.exp(P0fk(Lk0[i]))*norm
	else:
		PfkE[i]=np.exp(Pfk(Lk0[i]))*norm *nn
	if Lk0[i] >= Lk[-1]:
		PfkE[i]= np.exp(P0fk(Lk0[i]))*norm*nn*np.exp(Pfk(Lk[-1]))/np.exp(P0fk(Lk[-1]))
	else:
		pass
  
plt.plot(Lk0,PfkE)
plt.show() # este muestra el plot parecido a un pulso
plt.plot(Lk0,PfkE)
plt.xlim(24,32)
plt.show() #este muestra el plot parecido a un pulso pero con un zoom en el eje x

Pfk = interp1d(Lk0, PfkE, bounds_error=False, kind='cubic', fill_value="extrapolate")
PfkL = interp1d(Lk0, np.log(PfkE), bounds_error=False, kind='cubic', fill_value="extrapolate")

# plt.plot(Lk0,PfkE,'b')
# plt.show()

plt.plot(Lk0, np.exp(PfkL(Lk0)),'r')
plt.yscale('log')
#plt.xscale('log')
plt.show()
#sys.exit()
plt.plot(k0,Pfk(Lk0),'b')
#plt.plot(kr,np.exp(Pappr(Lkr)),'--')
plt.plot(k0,np.exp(PfkL(Lk0)),'r')
plt.plot(k0,np.exp(P0)*norm,'--')
plt.plot(k0,PfkE, color = 'g',linestyle = '--')
plt.yscale('log')
plt.xscale('log')
plt.show()
#sys.exit()
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

#Jacopo:
# with ax.twiny the two axis are sincronyzed but not perfectly, therefore if you want to 
ju=np.amax(Pfk(Lk0)) # creo k no c usa
fig, ax= plt.subplots()
plt.title(r'$\mathrm{Power\,\, Spectrum}$',fontsize=16)
ax2=ax.twiny()
ax.plot(Nexit0-NCMB, np.exp(P0)*norm, label='$P_{\zeta 0}$')
ax.plot(Nexit0-NCMB,Pfk(Lk0),'g', label='$P_{\zeta}$')
#
ax.axvline(x=Npeak-NCMB, linewidth=1, color='k', linestyle='--')
ax.set_xticks([0,10,20,Nfeat-NCMB,Npeak-NCMB,40,50,Nexit0[-1]-NCMB])
ax.set_xticklabels(['$N_*=0$', 10, 20, '$N_{f}$', '$N_{p}$', 40, 50, '$N_{end}$'])
ax2.plot(k0,0*np.exp(P0)*norm)
ax2.set(ylabel='', xlabel='$k\,(Mpc^{-1})$') 
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax2.tick_params(axis="x", labelsize=14) 
ax2.set_xscale('log')
plt.yscale('log')
ax.legend(fontsize=16)
plt.tight_layout()
plt.show()

print(Npeak,Nfeat)
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
  
size=4000

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


kzz=np.geomspace(kz[0],kz[-1],3000)
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
    # W2[i,:] = np.exp(-(q[:]/keq)**2*(Mz[i]/Meq)) # Gaussian
    # W2[i,:] = 3.*(np.sin(q[:]/kz[i])-q[:]/kz[i]*np.cos(q[:]/kz[i]))/(q[:]/kz[i])**3. # top-hat
    W2[i,:] = (3.*(np.sin(q[:]/kz[i])-q[:]/kz[i]*np.cos(q[:]/kz[i]))/(q[:]/kz[i])**3.)*(3.*(np.sin(csrad*q[:]/kz[i])-csrad*q[:]/kz[i]*np.cos(csrad*q[:]/kz[i]))/(csrad*q[:]/kz[i])**3.)
    W2[i,:] = (W2[i,:])**2
    integrand[i,:] = (16./81.)*Pfk(Lq[:])*W2[i,:]*(q[:])**3./(keq)**4. *(Mz[i]/Meq)**2
    sigmaR2[i]=Intarray_vec(integrand[i,:]*q[:],Lq[:])

plt.figure(2)
plt.plot(kz,sigmaR2,'k', label='$\sigma^2(k)$')
plt.legend(fontsize=15)
plt.yscale('log')
plt.xscale('log')
plt.show()

plt.figure(2)
plt.plot(Mz,sigmaR2,'k', label='$\sigma^2(M)$')
plt.legend(fontsize=15)
plt.yscale('log')
plt.xscale('log')
plt.show()

MH = Meq*(keq/kz)**2
sigmapeak=np.amax(sigmaR2)
mhpeak=np.argmin(np.abs(sigmaR2-sigmapeak))
MHp=MH[int(mhpeak)]

mzpeak=np.argmin(np.abs(sigmaR2-sigmapeak))
Mzp=MH[int(mzpeak)]

plt.figure(11)
plt.plot(MH,sigmaR2,'k', label='$\sigma^2(MH)$')
plt.legend(fontsize=15)
plt.axvline(x=MHp)
plt.plot(Mz,sigmaR2,color='orange', label='$\sigma^2(Mz)$')
plt.axvline(x=Mzp,color='orange')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$M/M_\odot$')
# plt.ylabel('Y-axis Label')
plt.show()

plt.figure(11)
plt.plot(Mz,sigmaR2,color='orange', label='$\sigma^2(Mz)$')
plt.legend(fontsize=15)
plt.axvline(x=Mzp,color='orange')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$M/M_\odot$')
# plt.ylabel('Y-axis Label')
plt.show()

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



M = np.geomspace(MH[-1], 10**(-1)*MH[0], size) # why compute M in this way?
mu = np.zeros([nkz, size])
Integrand2=np.zeros([nkz,size])
Integrand2b=np.zeros([nkz,size])
Integrand2d=np.zeros([nkz,size])
Integrand2c=np.zeros([nkz,size])
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
frac=1.
betamono=2*frac*sigmaR2/deltac/np.sqrt(2*np.pi*sigmaR2)*np.exp(-deltac**2/(2*sigmaR2))
fmono=1./OmegaCDM*np.sqrt(Meq/MH)*betamono

plt.plot(MH,fmono,'o',label='$f(M)$')
plt.yscale('log')
plt.xscale('log')
plt.title('f mono',fontsize=16)
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
plt.axvline(x=kpeak)
plt.plot(kz,sigmaR2,'r',label='$\sigma_R$')
plt.plot(kzr,sigmaR2r,'k,--',linewidth=1.3)
#plt.plot(M,f1,'--')
plt.legend(fontsize=10)
plt.yscale('log')
plt.xscale('log')
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

plt.figure(4)
plt.plot(M,f/OmegaPBH,'o',label='$f(M)$')
plt.axvline(x=Mp)  #the PBH mass corresponding to the peak 
plt.axvline(x=Mfeat) #the horizon mass corresponding to the feature
#plt.plot(M,f1,'--')
plt.legend(fontsize=10)
# Limiting the plot
plt.xlim(1e-17, 1e-8)  # Set limits for x-axis
plt.ylim(1e-16, 5)  # Set limits for y-axis
plt.yscale('log')
plt.xscale('log')
plt.title(r'$f(M)/f_{\rm PBH}$',fontsize=16)
plt.show()

# xpeak=np.log(10.**-12)
# #evaluate f(M=xpeak)
# xpeakindex=np.argmin(np.abs(LM-xpeak))
# Mpeak=M[int(xpeakindex)]


plt.figure(5)
plt.plot(M,f/fpeak,'o',label='$f(M)$')
plt.axvline(x=Mp)  #the PBH mass corresponding to the peak 
plt.axvline(x=Mfeat) #the horizon mass corresponding to the feature
#plt.plot(M,f1,'--')
plt.legend(fontsize=10)
# Limiting the plot
plt.xlim(1e-17, 1e-8)  # Set limits for x-axis
plt.ylim(1e-16, 5)  # Set limits for y-axis

plt.title(r'$f(M)/f_{peak}$',fontsize=16)
plt.yscale('log')
plt.xscale('log')
print('Mpeak=',Mp)
plt.show()

# np.savez(cwd+'\\bs\\data\\'+Wf+'-gaussian-data-C4-deltac05.npz', kz=kz, sigmaR2=sigmaR2, f=f, fpeak=fpeak, OmegaPBH=OmegaPBH, Mp=Mp)
# np.savez(cwd+'\\bs\\data\\gaussian-data-C'+str(C)+'-deltac'+str(deltac)+'-'+Wf+'.npz', kz=kz, sigmaR2=sigmaR2, f=f, fpeak=fpeak, OmegaPBH=OmegaPBH, Mp=Mp)
# np.savez('data\\gaussian-data-C'+str(C)+'-deltac'+str(deltac)+'-'+Wf+'.npz', kz=kz, sigmaR2=sigmaR2, f=f, fpeak=fpeak, OmegaPBH=OmegaPBH, Mp=Mp)

plt.plot(MH,fmono,'o', label='$fmono$')
plt.plot(M,f,'o', label='$f(M)$')
plt.yscale('log')
plt.xscale('log')
plt.title('f mono vs extended',fontsize=16)
plt.legend()
plt.show()

fpbhmono=-Intarray_vec(fmono,LMH)

plt.plot(MH,fmono/fpbhmono,'o', label='$fmono$')
plt.plot(M,f/OmegaPBH,'o', label='$f(M)$')
plt.yscale('log')
plt.xscale('log')
plt.title('f mono vs extended',fontsize=16)
plt.legend()
plt.show()