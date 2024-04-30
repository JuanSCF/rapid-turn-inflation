
import os, sys, time
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import interpn
from scipy.signal import savgol_filter
import dask.array as da
from scipy.optimize import minimize_scalar
############################################################################
############################# initialitiazion   ############################
############################################################################

# power spectrum parameters
deltaN=0.1
n=1
L0=251.327
L=n*L0
k00 = 1e11 # Mpc^-1 . '''this gives k_peak=1.287e13 Mpc^-1'''

# initial and final k that will be integrated
ki=6
kf=14
kikf=str(ki)+str(kf)
ki=1*10**ki 
kf=1*10**kf

Wf='Wthtf' # options: 'Wg', 'Wgc4', 'Wth', 'Wthtf'
nkk=300 #number of steps
spacing='geometric' # 'geometric' or 'linear'
size=3000


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

####################################################################
############################ File names ############################
####################################################################


gamma=0.36
if Wf=='Wg':
  deltac = 0.18
  C=1.44
else:
  deltac = 0.5
  C=4.
OmegaCDM=0.264



# File names
cwd = os.getcwd()

# Define the directory where you want to save the file
data_directory = os.path.join(cwd, 'data')

# File name to save bs data
databs_file = f'databs-gth-{nkk}-steps-{spacing}-spacing-{kikf}-lambda-{n}L0.npy'
databs_gth_file = f'databs-gth-{nkk}-steps-{spacing}-spacing-{kikf}-lambda-{n}L0.npy'

# Construct the full path including the directory
databs_file = os.path.join(data_directory, databs_file)


xi3_file=f'xi3-{Wf}-{nkk}-steps-{kikf}-{spacing}-spacing-lambda-{n}L0.npy'
xi3_file = os.path.join(data_directory, xi3_file)
xi3_gth_file=f'xi3-gth-{Wf}-{nkk}-steps-{kikf}-{spacing}-spacing-lambda-{n}L0.npy'
xi3_gth_file = os.path.join(data_directory, xi3_gth_file)

# databs=np.load(databs_file)
# databs_gth=np.load(databs_gth_file)


############################################################################
############################# initialitiazion   ############################
############################################################################

Omegam = 0.315 #???
Meq=(2.8)*10**17. #solar masses
keq=0.01*(Omegam/0.31) #Mpc^-1
# Mi=(keq/ki)**2. *Meq
# Mf=(keq/kf)**2. *Meq

def kofMH(M):
    return keq*(Meq/M)**0.5
def MHofk(k):
    return (keq/k)**2.*Meq

# np.savez(cwd+'\\bs\\data\\gaussian-data-C'+str(C)+'-deltac'+str(deltac)+Wf+'.npz', kz=kz, sigmaR2=sigmaR2, f=f, fpeak=fpeak, OmegaPBH=OmegaPBH, Mp=Mp)

fgaussian_data_file = os.path.join(cwd, f'data\\gaussian-data-C{C}-deltac{deltac}-{Wf}.npz')
fgaussian_data = np.load(fgaussian_data_file)

kz=fgaussian_data['kz']

####################################################################################################
# np.savez( xi3_file, xi3=xi3,f=f, f2=f2, fng=fng, t_xi3_MH=t_xi3_MH, Mp=Mp, Mpng=Mpng)
# xi3_data_Wthtf = np.load(xi3_file)
# xi3_Wthtf=xi3_data_Wthtf['xi3']
# f_Wthtf=xi3_data_Wthtf['f']
# f2_Wthtf=xi3_data_Wthtf['f2']
# fng_Wthtf=xi3_data_Wthtf['fng']
# Mp_Wthtf=xi3_data_Wthtf['Mp']
# Mpng_Wthtf=xi3_data_Wthtf['Mpng']

# xi3 = np.load(xi3_file)
xi3_gth = np.load(xi3_gth_file)

xi3=xi3_gth # +xi3

# xi3_data = np.load(xi3_file)
# xi3=xi3_data['xi3']
# f=xi3_data['f']
# f2=xi3_data['f2']
# fng=xi3_data['fng']
# Mp=xi3_data['Mp']
# Mpng1=xi3_data['Mpng1']
# Mpng2=xi3_data['Mpng2']
'''
añadir 2do peak ng
'''
####################################################################################################
MH = Meq*(keq/kz)**2

# Mmax=Mz[xi3maxindex]
sigmaR2=fgaussian_data['sigmaR2']

kzsmall =np.zeros(nkk)
varsmall =np.zeros(nkk)

# creo que esto está malo, revisar
# indices = np.argmin(np.abs(kk[:, np.newaxis] - kz), axis=1)
# kzsmall = kz[indices]
# varsmall = sigmaR2[indices]
for i in range(nkk):
    index=np.argmin(np.abs(kk[i]-kz)) ##  !!!
    kzsmall[i]=kz[index]
    varsmall[i]=sigmaR2[index]

# check if this is equivalent to what is above
# indices = np.argmin(np.abs(kk[:, np.newaxis] - kz), axis=1)
# kzsmall = kz[indices]
# varsmall = sigmaR2[indices]

MHsmall = MHofk(kzsmall)

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




LMH=np.log(MHsmall)


'''
perturbativity condition
np.amax(6.*varsmall**3/deltac**3/abs(xi3)) =2.68e+85
#
# lets compute a g value for every xi3 value
# gvec = 6.*varsmall**3/(deltac**3.) /abs(xi3)
g0 = -4e86 # (+-) approx max value that g0 may take ??
see +-4e83
g0 = -1e81 # (+-) between this value and the above, the pdf shows departures
g0 = 1e-44 # +-1e-36 interesting values, see pdf form
# between 1-2e-44 there appears neg values in f_ng(M)
g \in (-0.054, 0.013) ??
g0=-1e-36
'''
gmax = np.amax(6.*varsmall**3/deltac**3/abs(xi3))
# gval = -0.000004
# g0 = gmax*gval
g0 = 1e-20
gval = 1
pathfigs=cwd+f'\\figs\\gpivot\\g{gval}\\'

# plt.plot(MHsmall, xi3,'o')
# plt.plot(MHsmall, xi3)
# plt.title(f'xi3(MH), {Wf}')
# # plt.xscale('log')
# # plt.yscale('symlog')
# plt.ylabel(r'$\xi_3(M_H)$')
# plt.xlabel(r'$M_H/M_\odot$')
# # plt.savefig(pathfigs+f'xi3MH-{Wf}.png')
# # plt.savefig(pathfigs+f'xi3MH-{Wf}.svg')
# plt.show()

#3 plots
# plt.figure(00)
# plt.plot(MHsmall,g0*xi3,'o')
# plt.plot(MHsmall,g0*xi3 )
# plt.title(r'$g_0/g_{max}=$'+f'{gval}, {Wf}')
# plt.ylabel(r'$\xi_3(M_H)$')
# plt.xlabel(r'$M_H/M_\odot$')
# # plt.axvline(x=2e-13)
# # plt.axvline(x=60e-13)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.axhline(y=1, color='k', linestyle='--')
# plt.axhline(y=-1, color='k', linestyle='--')
# # plt.savefig(pathfigs+f'xi3MH-{Wf}-g{gval}.png')
# # plt.savefig(pathfigs+f'xi3MH-{Wf}-g{gval}.svg')
# plt.show()

# plt.figure(00)
# plt.plot(MHsmall,g0*xi3,'o')
# plt.plot(MHsmall,g0*xi3 )
# plt.title(r'$g_0/g_{max}=$'+f'{gval}, {Wf}')
# plt.ylabel(r'$\xi_3(M_H)$')
# plt.xlabel(r'$M_H/M_\odot$')
# # plt.axvline(x=2e-13)
# # plt.axvline(x=60e-13)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.axhline(y=1, color='k', linestyle='--')
# plt.axhline(y=-1, color='k', linestyle='--')
# plt.xscale('log')
# # plt.savefig(pathfigs+f'xi3MH-{Wf}-g{gval}-loglog.png')
# # plt.savefig(pathfigs+f'xi3MH-{Wf}-g{gval}-loglog.svg')
# plt.show()


# plt.figure(00)
# plt.plot(MHsmall,g0*xi3,'o')
# plt.plot(MHsmall,g0*xi3 )
# plt.title(r'$g_0/g_{max}=$'+f'{gval}, {Wf}')
# plt.ylabel(r'$\xi_3(M_H)$')
# plt.xlabel(r'$M_H/M_\odot$')
# # plt.axvline(x=2e-13)
# # plt.axvline(x=60e-13)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.axhline(y=1, color='k', linestyle='--')
# plt.axhline(y=-1, color='k', linestyle='--')
# plt.xscale('log')
# plt.yscale('symlog')
# plt.yticks(list(plt.yticks()[0]) + [0])
# # plt.savefig(pathfigs+f'xi3MH-{Wf}-g{gval}-symlog.png')
# # plt.savefig(pathfigs+f'xi3MH-{Wf}-g{gval}-symlog.svg')
# plt.show()


# MHsmall = MHsmall[::-1]
# varsmall = varsmall[::-1]
# xi3 = xi3[::-1]
def intfdeM(M,MHsmall,varsmall,xi3 ):
    xi3 = g0*xi3 #1e-78
    # xi3 = xi3[::-1]
    # MHsmall = MHsmall[::-1]
    # varsmall = varsmall[::-1]
    mu = (M/(C*MHsmall))
    Integrand_f=-2/(OmegaCDM)/(np.sqrt(np.pi*2*varsmall))*np.exp(-(mu**(1./gamma)+deltac)**2/(2*varsmall))*M/MHsmall*(1./gamma)*(M/(C*MHsmall))**(1./gamma)*np.sqrt(Meq/MHsmall)
    # f= Intarray_vec(Integrand1, LMH) # ojo: tengo abs() aca!
    # Integrand_f=Integrand_f*-0.5*keq*np.sqrt(Meq/MHsmall) # jacobian k->M_H?
    # deltaM=(mu**(1./gamma)+deltac)**2
    deltaM = mu**(1./gamma)+deltac
#   deltaM=deltac
    Integrand_ngcont=( (1./6.)*xi3*((deltaM/varsmall)**3.-3.*deltaM/varsmall**2.) )
    # Integrand_ngcont=Integrand_f*(1./6.)*xi3*(deltaM**3/varsmall**3-3*deltaM/varsmall**2)
    Integrand_ftot=Integrand_f*(1.+1./6.*xi3*((deltaM/varsmall)**3.-3.*deltaM/varsmall**2.) )
    Integrand_fresum=Integrand_f*np.exp(-(1./6.)*xi3*(3*deltaM/varsmall**2-(deltaM/varsmall)**3.))
    return Integrand_f, Integrand_ftot, Integrand_ngcont, Integrand_fresum, deltaM
'''
deltaM y la diferencia dentro de la parte ng nunca se hace negativas!
'''
MHvals = []
MHindexes = []
for i in [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
    index=np.argmin(np.abs(MHsmall-i)) ##  !!!
    MHindexes.append(index)
    MHvals.append(MHsmall[index])



def gaussian_pdf(alpha, i):
    return 1./(np.sqrt(2*np.pi*varsmall[i]))*np.exp(-alpha**2/(2*varsmall[i]))
def ng_pdf(alpha, i):
    return gaussian_pdf(alpha, i)*(1+(1./6.)*g0*xi3[i]*((alpha/varsmall[i])**3.-3.*alpha/varsmall[i]**2.) )
def ng_pdf_resum(alpha, i):
    return gaussian_pdf(alpha, i)*np.exp((1./6.)*g0*xi3[i]*((alpha/varsmall[i])**3.-3.*alpha/varsmall[i]**2.) )





# for i in MHindexes:
#     x = np.linspace(-5, 5, 3000)
#     gaussian_probf = gaussian_pdf(x, i)
#     ng_fprob = ng_pdf(x, i)
#     ng_fprob_resum = ng_pdf_resum(x, i)
#     plt.plot(x, gaussian_probf, label='gaussian')
#     plt.plot(x, ng_fprob, label='ng')
#     plt.plot(x, ng_fprob_resum, label='ng resummed')
#     # plt.xscale('log')
#     # plt.yscale('symlog')
#     plt.legend()
#     plt.ylabel(r'$P[\alpha]$')
#     plt.xlabel(r'$\alpha$')
#     plt.title(f'Monochrome PDF g={g0}, MH={MHsmall[i]}, {Wf}')
#     plt.axhline(y=0, color='r', linestyle='--')
#     plt.yticks(list(plt.yticks()[0]) + [0])
#     # plt.savefig(pathfigs+f'mono-pdf-{Wf}-g{gval}-symlog.png')
#     # plt.savefig(pathfigs+f'mono-pdf-{Wf}-g{gval}-symlog.svg')
#     plt.show()

def gaussian_pdf_ex(Mpbh, i):
    alpha = (Mpbh/(C*MHsmall[i]))**(1./gamma)+deltac
    return 1./(np.sqrt(2*np.pi*varsmall[i]))*np.exp(-alpha**2/(2*varsmall[i]))
def ng_pdf_ex(Mpbh, i):
    alpha = (Mpbh/(C*MHsmall[i]))**(1./gamma)+deltac
    return gaussian_pdf_ex(Mpbh, i)*(1+(1./6.)*g0*xi3[i]*((alpha/varsmall[i])**3.-3.*alpha/varsmall[i]**2.) )
def ng_pdf_resum_ex(Mpbh, i):
    alpha = (Mpbh/(C*MHsmall[i]))**(1./gamma)+deltac
    return gaussian_pdf_ex(Mpbh, i)*np.exp((1./6.)*g0*xi3[i]*((alpha/varsmall[i])**3.-3.*alpha/varsmall[i]**2.) )

def ng_pdf_resum_ex2(Mpbh, i):
    alpha = (Mpbh/(C*MHsmall[i]))**(1./gamma)+deltac
    return 1./(np.sqrt(2*np.pi*varsmall[i]))*np.exp(-alpha**2/(2*varsmall[i])*(1-1./6.*g0*xi3[i]/varsmall[i]**2.*2*alpha) )

for i in MHindexes:
    x = np.geomspace(1e-18, 1e-10, 3000)
    gaussian_probf_ex = gaussian_pdf_ex(x, i)
    ng_fprob_ex = ng_pdf_ex(x, i)
    ng_fprob_resum_ex = ng_pdf_resum_ex(x, i)
    ng_fprob_ex2 = ng_pdf_resum_ex2(x, i)
    plt.plot(x, gaussian_probf_ex, label='gaussian')
    plt.plot(x, ng_fprob_ex, label='ng')
    # plt.plot(x, ng_fprob_resum_ex, label='ng resummed')
    plt.plot(x, ng_fprob_ex2, label='ng resummed2')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    plt.ylabel(r'$P[\delta(M)]$')
    plt.xlabel(r'$M_{pbh}/M_\odot$')
    plt.title(f'Extended PDF g={g0}, MH={MHsmall[i]}, {Wf}')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.yticks(list(plt.yticks()[0]) + [0])
    # plt.savefig(pathfigs+f'mono-pdf-{Wf}-g{gval}-symlog.png')
    # plt.savefig(pathfigs+f'mono-pdf-{Wf}-g{gval}-symlog.svg')
    plt.show()



sys.exit()
# this part is for plotting the integrand of the ng contribution
f=np.zeros(size)
fng=np.zeros(size)
fresum=np.zeros(size)
ngcont=[]
deltaMz=[]
# f2 = []
'''
this for may be optimizable.
eye with the fact that the length of Mz is way bigger than the length of xi3
'''

for i in tqdm.tqdm(range(0, len(Mz))):
    a,b,c,d, e = intfdeM(Mz[i],MHsmall,varsmall,xi3)
    f[i] = Intarray_vec( a, LMH)
    fng[i] = Intarray_vec( b, LMH)
    ngcont.append(c)
    fresum[i] = Intarray_vec( d, LMH)
    deltaMz.append(e)
    # f2.append(a)
ngcont = np.array(ngcont)
deltaMz = np.array(deltaMz)
def find_crossing_index(ngcont_3000, ngcont_300):
    for i in range(len(ngcont_3000)):
        if np.sign(ngcont_3000[i]) != np.sign(ngcont_300[i]):
            return i
    return -1  # If no crossing is found
##
# plor for the pdf of delta(Mpbh)
#

for i in range(5):
    deltam = deltaMz[200*i]
    g_pdf_ex = 1./(np.sqrt(2*np.pi*varsmall))*np.exp(-deltam**2/(2*varsmall))
    ng_pdf_ex = g_pdf_ex*(1+(1./6.)*g0*xi3*((deltam/varsmall)**3.-3*deltam/varsmall**2) )
    # ng_pdf_ex2=g_pdf_ex*np.exp((1./6.)*g0*xi3*((deltam/varsmall)**3.-3*deltam/varsmall**2) )
    plt.plot(MHsmall, g_pdf_ex, label='gaussian')
    plt.plot(MHsmall, ng_pdf_ex, label='ng')
    # plt.plot(MHsmall, ng_pdf_ex2, label='ng_resum')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    plt.ylabel(r'$P[R]$')
    plt.xlabel(r'$M_H/M_\odot$')
    plt.title(f'Extended PDF Mpbh={Mz[100*i]} g={gval}, {Wf}')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
# plt.axhline(y=0, color='r', linestyle='--')
# plt.show()
# m0=MHofk(k00)
for i in range(10):
    plt.axvline(x=Mz[585], color='red')
    plt.plot(MHsmall, ngcont[100*i],'o', label=f'Mpbh={ Mz[100*i] }')
    plt.plot(MHsmall, ngcont[100*i])
    # plt.show()
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=1, color='k', linestyle='--')
    plt.axhline(y=-1, color='k', linestyle='--')
    plt.legend()
    plt.title(f'f_ngcont(MH) para Mpbh fijo, {Wf}')
    plt.xlabel(r'$M_{H}$')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.show()    

for i in range(10): #10
    plt.plot(MHsmall, ngcont[300*i], label=f'Mpbh={ Mz[300*i] }')
    plt.xscale('log')
    plt.yscale('symlog')
    # plt.axhline(y=0, color='r', linestyle='--')
    # plt.show()
plt.title(f'ng contribution g={gval}, {Wf}')    
# plt.axvline(x=Mp)
plt.xlabel(r'$M_{H}$')
# plt.axvline(x=1e-10)
# plt.axvline(x=2e-13)
# plt.axvline(x=60e-13)
plt.axvline(x=Mz[585], color='red')
plt.axhline(y=0, color='r', linestyle='--')
plt.axhline(y=1, color='k', linestyle='--')
plt.axhline(y=-1, color='k', linestyle='--')
plt.legend()
# plt.savefig(pathfigs+f'ngcont-{Wf}-g{gval}-symlog.png')
# plt.savefig(pathfigs+f'ngcont-{Wf}-g{gval}-symlog.svg')
plt.show()    


for i in range(14): #10
    ni = 50*i
    plt.plot(MHsmall, ngcont[ni], label=f'Mpbh={ Mz[ni] }')
    plt.xscale('log')
    plt.yscale('symlog')
    # plt.axhline(y=0, color='r', linestyle='--')
    # plt.show()
plt.title(f'ng contribution g={gval}, {Wf}')    
# plt.axvline(x=Mp)
plt.xlabel(r'$M_{H}$')
# plt.axvline(x=1e-10)
# plt.axvline(x=2e-13)
# plt.axvline(x=60e-13)
plt.axvline(x=Mz[585], color='red')
plt.axhline(y=0, color='r', linestyle='--')
plt.axhline(y=1, color='k', linestyle='--')
plt.axhline(y=-1, color='k', linestyle='--')
plt.xlim(Mz[0], 1e-11)
plt.legend()
# plt.savefig(pathfigs+f'ngcont-{Wf}-g{gval}-symlog.png')
# plt.savefig(pathfigs+f'ngcont-{Wf}-g{gval}-symlog.svg')
plt.show()  

# function for studying whether all values of an array are less than zero

def all_values_less_than_zero(arr):
    for num in arr:
        if num >= 0:
            return False
    return True
q=0
for i in range(size):
    if all_values_less_than_zero(ngcont[i]):
        q=q+1
print(q)

# f_noabs = f
# fng_noabs = fng # values until index 653 are positive, then negative values

# plt.plot(Mz, f_noabs,'o',label='f')
# plt.plot(Mz, fng_noabs,'o',label='f_ng')
# # plt.plot(MHsmall,g0,'o',label='f_ng')
# plt.legend()
# plt.title(f'no abs f(M), {Wf} Smoothing function')
# plt.yscale('symlog')
# plt.xscale('log')
# plt.show()


def any_neg_value(array):
    if np.any(array < 0):
        print("There are negative values in the array")
    else:
        print("All values are not negative")



# sys.exit()

# f=abs(f)
# fng=abs(fng)


LM=np.log(Mz)
f_pbh= Intarray_vec(f,LM)
fng_pbh= Intarray_vec(fng,LM)


fpeak=np.amax(abs(f))
mpeak=np.argmin(np.abs(abs(f)-fpeak)) 

Mp=Mz[int(mpeak)]


fngpeak=np.amax(fng)
mngpeak=np.argmin(np.abs(fng-fngpeak))
Mpng=Mz[int(mngpeak)]

# # Find the maximum value of fng before and after mpeak
# fng_before_mpeak = np.amax(fng[:mpeak])
# fng_after_mpeak = np.amax(fng[mpeak:])

# # Find the corresponding indices of these maximum values
# mngpeak1 = np.where(fng == fng_before_mpeak)[0][-1]  # Last occurrence before mpeak
# mngpeak2 = np.where(fng == fng_after_mpeak)[0][0]   # First occurrence after mpeak

# # Retrieve the corresponding values of Mz at these indices
# Mpng1 = Mz[mngpeak1]
# Mpng2 = Mz[mngpeak2]

# print("Mng1:", Mpng1)
# print("Mng2:", Mpng2)

# np.savez( xi3_file, xi3=xi3,f=f, f2=f2, fng=fng, t_xi3_MH=t_xi3_MH, Mp=Mp, Mpng=Mpng)

# np.savez(xi3_file, xi3=xi3,f=f, f2=f2, fng=fng, Mp=Mp, Mpng1=Mpng1, Mpng2=Mpng2)
# np.savez(xi3_file, xi3=xi3,f=f, f2=f2, fng=fng, Mp=Mp, Mpng1=Mpng1, Mpng2=Mpng2, f_pbh=f_pbh, fng_pbh=fng_pbh, fpeak=fpeak, fngpeak=fngpeak)

plt.plot(Mz,f, 'o',label='f')
plt.plot(Mz,fng, 'o',label='f_ng')
plt.plot(Mz,fng, label='f_ng')
plt.xscale('log')
plt.yscale('symlog')
plt.axhline(y=1, color='r', linestyle='--')
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=Mp)
plt.axvline(x=Mpng, color='orange')
# plt.axvline(x=Mpng2, color='orange')
plt.title(f'f(M), {Wf} Smoothing function')
plt.legend(loc='right', bbox_to_anchor=(1, 0.85))
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f(M)$')
# plt.ylim(1e-9,5e12)
plt.yticks(list(plt.yticks()[0]) + [0])
# plt.savefig(pathfigs+f'f(M)-{Wf}-g{gval}-symlog.png')
# plt.savefig(pathfigs+f'f(M)-{Wf}-g{gval}-symlog.svg')
plt.show()

any_neg_value(fng)


plt.plot(Mz,f/f_pbh,'o',label='f')
plt.plot(Mz,fng/fng_pbh,'o',label='f_ng')
plt.xscale('log')
plt.yscale('symlog')
plt.legend(loc='right', bbox_to_anchor=(1, 0.85))
plt.axhline(y=1, color='r', linestyle='--')
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=Mp)
plt.axvline(x=Mpng, color='orange')
# plt.axvline(x=Mpng2, color='orange')
plt.title(f'f(M)/f_pbh, {Wf} Smoothing function')
# plt.ylim(1e-14,5e0)
# plt.axhline(y=1, color='r', linestyle='--')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f(M)/f_{\rm PBH}$')
plt.yticks(list(plt.yticks()[0]) + [0])
# plt.savefig(pathfigs+f'f(M)-{Wf}-norm-g{gval}-symlog.png')
# plt.savefig(pathfigs+f'f(M)-{Wf}-norm-g{gval}-symlog.svg')
plt.show()

# plt.plot(Mz,f/fpeak,'o',label='f')
# plt.plot(Mz,fng/fngpeak ,'o',label='f_ng')
# plt.xscale('log')
# plt.yscale('symlog')
# plt.legend(loc='right', bbox_to_anchor=(1, 0.85))
# plt.axhline(y=1, color='r', linestyle='--')
# plt.axhline(y=0, color='k', linestyle='--')
# plt.axvline(x=Mp)
# plt.axvline(x=Mpng, color='orange')
# # plt.axvline(x=Mpng2, color='orange')
# plt.title(f'f(M)/f_peak, {Wf} Smoothing function')
# # plt.ylim(1e-14,5e0)
# # plt.axhline(y=1, color='r', linestyle='--')
# plt.xlabel(r'$M_{\rm PBH}$')
# plt.ylabel(r'$f(M)/f_{\rm peak}$')
# plt.show()


# # plot without v lines
# plt.loglog(Mz,f/fpeak,label='f')
# plt.loglog(Mz,fng/fngpeak ,label='f_ng')
# # plt.axvline(x=Mp)
# # plt.axvline(x=Mpng1, color='orange')
# # plt.axvline(x=Mpng2, color='orange')
# plt.title(f'f(M)/f_peak, {Wf} Smoothing function')
# plt.ylim(1e-14,5e0)
# plt.axhline(y=1, color='r', linestyle='--')
# plt.xlabel(r'$M_{\rm PBH}$')
# plt.ylabel(r'$f(M)/f_{\rm peak}$')
# plt.legend(loc='right', bbox_to_anchor=(1, 0.85))
# # plt.savefig(pathfigs+f'f(M)-f_pbh-{Wf}.png')
# # plt.savefig(pathfigs+f'f(M)-f_pbh-{Wf}.svg')
# plt.show()

# # plot with v lines
# plt.loglog(Mz,f/fpeak,label='f')
# plt.loglog(Mz,fng/fngpeak ,label='f_ng')
# plt.axvline(x=Mp)
# plt.axvline(x=Mpng1, color='orange')
# plt.axvline(x=Mpng2, color='orange')
# plt.title(f'f(M)/f_peak, {Wf} Smoothing function')
# plt.ylim(1e-14,5e0)
# plt.axhline(y=1, color='r', linestyle='--')
# plt.xlabel(r'$M_{\rm PBH}$')
# plt.ylabel(r'$f(M)/f_{\rm peak}$')
# plt.legend(loc='right', bbox_to_anchor=(1, 0.85))
# # plt.savefig(pathfigs+f'f(M)-f_pbh-vlines-{Wf}.png')
# # plt.savefig(pathfigs+f'f(M)-f_pbh-vlines-{Wf}.svg')
# plt.show()


print(f'{Wf}_Mp: {Mp}')
print(f'{Wf}_Mpng: {Mpng}')
# print(f'{Wf}_Mpng1: {Mpng1}')
# print(f'{Wf}_Mpng2: {Mpng2}')
print(f'{Wf}_fpbh: {f_pbh}')
print(f'{Wf}_fngpbh: {fng_pbh}')
print(f'{Wf}_fpeak: {fpeak}')
print(f'{Wf}_fngpeak: {fngpeak}')

# sys.exit()
fresumpeak = np.amax(fresum)
# -------------------------------------------
# f_resum is almost equal to the gaussian result
# 
# plt.loglog(Mz,f/fpeak,label='f')
# plt.loglog(Mz,fng/fngpeak ,label='f_ng')
# plt.plot(Mz,fresum/fresumpeak ,label='f_resum')
# plt.yscale('log')
# plt.xscale('log')
# # plt.axvline(x=Mp)
# # plt.axvline(x=Mpng1, color='orange')
# # plt.axvline(x=Mpng2, color='orange')
# plt.title(f'f_resum/f_peak, {Wf} Smoothing function')
# plt.ylim(1e-14,5e0)
# plt.axhline(y=1, color='r', linestyle='--')
# plt.xlabel(r'$M_{\rm PBH}$')
# plt.ylabel(r'$f(M)/f_{\rm peak}$')
# plt.legend(loc='right', bbox_to_anchor=(1, 0.85))
# # plt.savefig(pathfigs+f'f(M)-f_pbh-{Wf}.png')
# # plt.savefig(pathfigs+f'f(M)-f_pbh-{Wf}.svg')
# plt.show()


# -------------------------------------------
if Wf == 'Wg':
    Wg_gxi3 = g0*xi3
    Wg_f_fpbh = f/f_pbh
    Wg_fng_fpbh = fng/fng_pbh
    Wg_f_fpeak = f/fpeak
    Wg_fng_fpeak = fng/fngpeak

    Wg_Mp = Mp
    # Wg_Mpng1 = Mpng1
    # Wg_Mpng2 = Mpng2

    Wg_fpbh=f_pbh
    Wg_fngpbh=fng_pbh
    Wg_fpeak = fpeak
    Wg_fngpeak = fngpeak
elif Wf == 'Wgc4':
    Wgc4_gxi3 = g0*xi3
    Wgc4_f_fpbh = f/f_pbh
    Wgc4_fng_fpbh = fng/fng_pbh
    Wgc4_f_fpeak = f/fpeak
    Wgc4_fng_fpeak = fng/fngpeak

    Wgc4_Mp = Mp
    # Wgc4_Mpng1 = Mpng1
    # Wgc4_Mpng2 = Mpng2

    Wgc4_fpbh=f_pbh
    Wgc4_fngpbh=fng_pbh
    Wgc4_fpeak = fpeak
    Wgc4_fngpeak = fngpeak
elif Wf == 'Wth':
    Wth_gxi3 = g0*xi3
    Wth_f_fpbh = f/f_pbh
    Wth_fng_fpbh = fng/fng_pbh
    Wth_f_fpeak = f/fpeak
    Wth_fng_fpeak = fng/fngpeak

    Wth_Mp = Mp
    # Wth_Mpng1 = Mpng1
    # Wth_Mpng2 = Mpng2

    Wth_fpbh=f_pbh
    Wth_fngpbh=fng_pbh
    Wth_fpeak = fpeak
    Wth_fngpeak = fngpeak
elif Wf == 'Wthtf':
    Wthtf_gxi3 = g0*xi3
    Wthtf_f_fpbh = f/f_pbh
    Wthtf_fng_fpbh = fng/fng_pbh
    Wthtf_f_fpeak = f/fpeak
    Wthtf_fng_fpeak = fng/fngpeak

    Wthtf_Mp = Mp
    # Wthtf_Mpng1 = Mpng1
    # Wthtf_Mpng2 = Mpng2

    Wthtf_fpbh=f_pbh
    Wthtf_fngpbh=fng_pbh
    Wthtf_fpeak = fpeak
    Wthtf_fngpeak = fngpeak

sys.exit()

plt.plot(MHsmall,Wg_gxi3/g0, label='Wg')
plt.plot(MHsmall,Wgc4_gxi3/g0, label='Wgc4')
plt.plot(MHsmall,Wth_gxi3/g0, label='Wth')
plt.plot(MHsmall,Wthtf_gxi3/g0, label='Wthtf')
plt.legend(loc='right')
#plt.legend(loc='lower right', bbox_to_anchor=(0.98, 0.08))
plt.title(f' xi3 for different Smoothing functions')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$g_i\xi3(MH)$')
plt.xscale('log')
plt.yscale('symlog')
# plt.savefig(pathfigs+'fullW-gxi3.png')
# plt.savefig(pathfigs+'fullW-gxi3.svg')
plt.show()

plt.plot(MHsmall,Wg_gxi3, label='Wg')
plt.plot(MHsmall,Wgc4_gxi3, label='Wgc4')
plt.plot(MHsmall,Wth_gxi3, label='Wth')
plt.plot(MHsmall,Wthtf_gxi3, label='Wthtf')
plt.legend(loc='right')
#plt.legend(loc='lower right', bbox_to_anchor=(0.98, 0.08))
plt.title(f'g_i*xi3 for different Smoothing functions')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$g_i\xi3(MH)$')
plt.xscale('log')
plt.yscale('symlog')
# plt.savefig(pathfigs+'fullW-gxi3.png')
# plt.savefig(pathfigs+'fullW-gxi3.svg')
plt.show()



# plt.loglog(Mz,Wg_f_fpbh, label='Wg')
# plt.loglog(Mz,Wgc4_f_fpbh, label='Wgc4')
# plt.loglog(Mz,Wth_f_fpbh, label='Wth')
# plt.loglog(Mz,Wthtf_f_fpbh, label='Wthtf')
# plt.legend(loc='right')
# plt.title(f'f/f_pbh for different Smoothing functions')
# plt.xlabel(r'$M_{\rm PBH}$')
# plt.ylabel(r'$f/f_{\rm PBH}$')
# plt.savefig(pathfigs+'fullW-f-f_peak.png')
# plt.savefig(pathfigs+'fullW-f-f_peak.svg')
# plt.show()


plt.loglog(Mz,Wg_fng_fpbh, label='Wg')
plt.loglog(Mz,Wgc4_fng_fpbh, label='Wgc4')
plt.loglog(Mz,Wth_fng_fpbh, label='Wth')
plt.loglog(Mz,Wthtf_fng_fpbh, label='Wthtf')
plt.legend(loc='right')
plt.title(f'f_ng/f_pbh for different Smoothing functions')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f_{\rm ng}/f_{\rm PBH}$')
# plt.savefig(pathfigs+'fullW-fng-f_peak.png')
# plt.savefig(pathfigs+'fullW-fng-f_peak.svg')
plt.show()


# plt.loglog(Mz,Wg_f_fpeak, label='Wg')
# plt.loglog(Mz,Wth_f_fpeak, label='Wth')
# plt.loglog(Mz,Wthtf_f_fpeak, label='Wthtf')
# plt.legend(loc='right')
# plt.title(f'f/f_peak for different Smoothing functions')
# plt.xlabel(r'$M_{\rm PBH}$')
# plt.ylabel(r'$f/f_{\rm peak}$')
# plt.savefig(pathfigs+'fullW-f-f_peak.png')
# plt.savefig(pathfigs+'fullW-f-f_peak.svg')
# plt.show()


plt.loglog(Mz,Wg_fng_fpeak, label='Wg')
plt.loglog(Mz,Wgc4_fng_fpeak, label='Wgc4')
plt.loglog(Mz,Wth_fng_fpeak, label='Wth')
plt.loglog(Mz,Wthtf_fng_fpeak, label='Wthtf')
plt.legend(loc='right')
plt.title(f'f_ng/f_peak for different Smoothing functions')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f_{\rm ng}/f_{\rm peak}$')
# plt.savefig(pathfigs+'fullW-fng-f_peak.png')
# plt.savefig(pathfigs+'fullW-fng-f_peak.svg')
plt.show()












# all plots, zoom


plt.loglog(Mz,Wg_f_fpbh, label='Wg')
plt.loglog(Mz,Wgc4_f_fpbh, label='Wgc4')
plt.loglog(Mz,Wth_f_fpbh, label='Wth')
plt.loglog(Mz,Wthtf_f_fpbh, label='Wthtf')
plt.legend(loc='upper right')
plt.ylim(1e-14,5e0)
plt.title(f'f/f_pbh for different Smoothing functions')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f/f_{\rm PBH}$')
# plt.savefig(pathfigs+'fullW-zoom-f-f_peak.png')
# plt.savefig(pathfigs+'fullW-zoom-f-f_peak.svg')
plt.show()

plt.loglog(Mz,Wg_fng_fpbh, label='Wg')
plt.loglog(Mz,Wgc4_fng_fpbh, label='Wgc4')
plt.loglog(Mz,Wth_fng_fpbh, label='Wth')
plt.loglog(Mz,Wthtf_fng_fpbh, label='Wthtf')
plt.legend(loc='upper right')
plt.ylim(1e-14,5e0)
plt.title(f'f_ng/f_pbh for different Smoothing functions')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f_{\rm ng}/f_{\rm PBH}$')
# plt.savefig(pathfigs+'fullW-zoom-fng-f_peak.png')
# plt.savefig(pathfigs+'fullW-zoom-fng-f_peak.svg')
plt.show()

plt.loglog(Mz,Wg_f_fpeak, label='Wg')
plt.loglog(Mz,Wgc4_f_fpeak, label='Wgc4')
plt.loglog(Mz,Wth_f_fpeak, label='Wth')
plt.loglog(Mz,Wthtf_f_fpeak, label='Wthtf')
plt.legend(loc='upper right')
plt.ylim(1e-14,5e0)
plt.title(f'f/f_peak for different Smoothing functions')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f/f_{\rm peak}$')
# plt.savefig(pathfigs+'fullW-zoom-f-f_pbh.png')
# plt.savefig(pathfigs+'fullW-zoom-f-f_pbh.svg')
plt.show()

plt.loglog(Mz,Wg_fng_fpeak, label='Wg')
plt.loglog(Mz,Wgc4_fng_fpeak, label='Wgc4')
plt.loglog(Mz,Wth_fng_fpeak, label='Wth')
plt.loglog(Mz,Wthtf_fng_fpeak, label='Wthtf')
plt.legend(loc='upper right')
plt.ylim(1e-14,5e0)
plt.title(f'f_ng/f_peak for different Smoothing functions')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f_{\rm ng}/f_{\rm peak}$')
# plt.savefig(pathfigs+'fullW-zoom-fng-f_pbh.png')
# plt.savefig(pathfigs+'fullW-zoom-fng-f_pbh.svg')
plt.show()









