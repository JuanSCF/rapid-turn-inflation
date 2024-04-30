
import os, sys, time
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import interpn
from scipy.signal import savgol_filter
import dask.array as da
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
ki=6 
kf=14
kikf=str(ki)+str(kf)
ki=1*10**ki 
kf=1*10**kf

Wf='Wthtf' # Wg, Wgc4, Wth, Wthtf 
print(f'Wf: {Wf}')
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

if Wf=='Wg':
  deltac = 0.18
  C=1.44
else:
  deltac = 0.5
  C=4.

# deltac = 0.5
# C = 4.

gamma=0.36
OmegaCDM = 0.264





# File names
cwd = os.getcwd()

# Define the directory where you want to save the file
data_directory = os.path.join(cwd, 'data')

# File name to save bs data
databs_file = f'databs-gth-{nkk}-steps-{spacing}-spacing-{kikf}-lambda-{n}L0.npy'

# Construct the full path including the directory
databs_file = os.path.join(data_directory, databs_file)

# xi3_file=f'xi3-{Wf}-{nkk}-steps-{kikf}-{spacing}-spacing-lambda-{n}L0-c4-d05.npz'
xi3_file=f'xi3-gth-{Wf}-{nkk}-steps-{kikf}-{spacing}-spacing-lambda-{n}L0'
xi3_file = os.path.join(data_directory, xi3_file)

# databs=np.load(databs_file)
databs = np.load(f'C:\ZZZ\Laburos\Codes\databs-gth-{nkk}-steps-geometric-spacing-{kikf}-lambda-1L0.npy')


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


########################################################################
# Smoothing function as a function of the collapsing scale q=RH^-1
########################################################################

    # Gaussian
if Wf=='Wg' or Wf=='Wgc4':
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
    csrad=np.sqrt(1./3.)
    def W(k,q):
        a=3.*(np.sin(k/q)-k/q*np.cos(k/q))/(k/q)**3. 
        b=3.*(np.sin(csrad*k/q)-csrad*k/q*np.cos(csrad*k/q))/(csrad*k/q)**3. 
        return a*b

    # q=q[:, None, None]
    # q=kofMH(MH)
    
    


def Mcal(k,q):
    # q=q[:, None, None]
    m=4./9. *(k/q)**2.  *W(k,q)
    return m
########################################################################
########################################################################

# vectorized
# def int_xi3(m1,m2,wx):
#     a = k0**2*0.5/12.*m1[:, None, None]*m2[None, :, None]*wx*databs
#     return a

# def integrandxi3(Mh,k1,k2,x):
#     k1=k1/k0
#     k2=k2/k0
#     q=kofMH(Mh)/k0
#     m1=Mcal(k1,q)
#     m2=Mcal(k2,q)
#     k12x = np.sqrt(k1[:, None, None]**2 + k2[None, :, None]**2 - 2*k1[:, None, None]*k2[None, :, None]*x[None, None, :])
#     wx=4./9.*q**(-2)*W(k12x,q)

#     condition = (k12x < L) & (k1[:, None, None] < L) & (k2[None, :, None] < L)
#     a=np.zeros_like(databs)
#     a = np.where(condition, int_xi3(m1,m2,wx), a)
#     return a

def int_xi3(m1,m2,wx):
    a = 0.5/12.*m1[:, None, None]*m2[None, :, None]*wx*databs
    return a

def integrandxi3(Mh,k1,k2,x):
    q=kofMH(Mh)
    m1=Mcal(k1,q)
    m2=Mcal(k2,q)
    k12x = np.sqrt(k1[:, None, None]**2 + k2[None, :, None]**2 - 2*k1[:, None, None]*k2[None, :, None]*x[None, None, :])
    wx=4./9.*q**(-2.)*W(k12x,q)

    condition = (k12x < L*k0) & (k1[:, None, None] < L*k0) & (k2[None, :, None] < L*k0)
    a=np.zeros_like(databs)
    a = np.where(condition, int_xi3(m1,m2,wx), a)
    return a


LMH=np.log(MHsmall)
ti= time.time()
xi3 = np.zeros(len(MHsmall))

print('xi3 calc')
initial_time_str = time.strftime('%H:%M:%S', time.localtime(ti))
print('Initial time:', initial_time_str)
'''
this for is optimizable
'''
# MHsmall = MHsmall[::-1]
'''
integrar con respecto a k1/k0, k2/k0, x ????
'''
for i in tqdm.tqdm(range(len(MHsmall))):
    xi3[i] = Intarray3D_vec(integrandxi3(MHsmall[i], k1, k2, x), k1, k2, x)

# def xi3calc():
#     for i in tqdm.tqdm(range(len(MHsmall))):
#         xi3[i] = Intarray3D_vec(integrandxi3(MHsmall[i], k1, k2, x), [k1, k2, x])

tf = time.time()
duration = tf - ti
# t_xi3_MH=duration

# Convert initial time to hh:mm:ss format
final_time_str = time.strftime('%H:%M:%S', time.localtime(tf))
print('Final time:', final_time_str)

print(f"Computation of xi3 completed in {duration:.2f} seconds")

# np.save(xi3_file, xi3)
# np.save(f'C:\\ZZZ\\Laburos\\Codes\\xi3-gth-{Wf}-{nkk}-steps-{kikf}-geometric-spacing-lambda-1L0.npy', xi3)


################################################
# extremely large memory consumption

################################################
# ti = time.time()
# ###############################################
# # Create 4D arrays using broadcasting
# MH_broadcasted = MHsmall[:, np.newaxis, np.newaxis, np.newaxis]
# k1_broadcasted = k1[np.newaxis, :, np.newaxis, np.newaxis]
# k2_broadcasted = k2[np.newaxis, np.newaxis, :, np.newaxis]
# x_broadcasted = x[np.newaxis, np.newaxis, np.newaxis, :]

# # Compute the integrand using broadcasting
# integrand_broadcasted = integrandxi3(MH_broadcasted, k1_broadcasted, k2_broadcasted, x_broadcasted)

# # Calculate the 3D integral using vectorized functions
# xi3 = Intarray3D_vec(integrand_broadcasted, k1_broadcasted, k2_broadcasted, x_broadcasted)

# # Sum over the last dimension to get xi3 for each MHsmall value
# xi3_2 = np.sum(xi3, axis=-1)
# ################################################
# tf = time.time()
# duration = tf - ti
# print(f"Computation of xi3 completed in {duration:.2f} seconds")
################################################

################################################
# chunk_size = 50  # You can adjust this value based on your available memory

# xi3_list = []

# for i in range(0, len(MHsmall), chunk_size):
#     end_idx = min(i + chunk_size, len(MHsmall))
    
#     # Create chunked 4D arrays using broadcasting
#     MH_chunk = MHsmall[i:end_idx, np.newaxis, np.newaxis, np.newaxis]
#     k1_chunk = k1[np.newaxis, :, np.newaxis, np.newaxis]
#     k2_chunk = k2[np.newaxis, np.newaxis, :, np.newaxis]
#     x_chunk = x[np.newaxis, np.newaxis, np.newaxis, :]
    
#     # Compute the integrand using broadcasting for the chunk
#     integrand_chunk = integrandxi3(MH_chunk, k1_chunk, k2_chunk, x_chunk)
    
#     # Calculate the 3D integral using vectorized functions for the chunk
#     xi3_chunk = Intarray3D_vec(integrand_chunk, k1_chunk, k2_chunk, x_chunk)
    
#     # Sum over the last dimension to get xi3 for each MHsmall value in the chunk
#     xi3_chunk = np.sum(xi3_chunk, axis=-1)
    
#     xi3_list.append(xi3_chunk)

# # Concatenate the xi3 chunks to get the final xi3 array
# xi3_2 = np.concatenate(xi3_list)
###############################################

# plt.loglog(kzsmall,abs(xi3),'o')
# plt.plot(kzsmall,abs(xi3))
# plt.title(f'abs( xi3(k) ), {Wf}')
# plt.show()

# plt.figure(00)
# plt.plot(kzsmall,xi3)
# plt.plot(kzsmall,xi3,'o')
# plt.title(f'xi3(k), {Wf}')
# plt.xscale('log')
# plt.yscale('symlog')
# plt.show()

# plt.figure(00)
plt.plot(MHsmall,xi3,'o')
plt.plot(MHsmall,xi3)
plt.title(f'xi3(MH), {Wf}')
plt.xscale('log')
plt.yscale('symlog')
plt.show()

# np.save(xi3_file, xi3ofk)



# to apply the perturbatibity condition I look for the maximum of abs(xi3) and its index.
# then, i look for the value of the variance at that index.
# afterwards, i find the maximun value for g that satisfies the perturbativity condition.

kstar_index=np.argmin(np.abs(kzsmall-k0)) # with this line i'm neglecting k<k0
xi3max=max(abs(xi3[kstar_index:]))
xi3maxindex=np.argmax(abs(xi3[kstar_index:]))+kstar_index # accounting for the neglected indexes

varmax=varsmall[xi3maxindex]
# S3max=xi3max/varmax**2
# g = 6.*varmax/(deltac**3.) /S3max

# lets compute a g value for every xi3 value
# gvec = 6.*varsmall**3/(deltac**3.) /abs(xi3)
g0 = 1

plt.figure(00)
plt.plot(MHsmall, g0*xi3)
plt.plot(MHsmall, g0*xi3,'o')
plt.title(f'g_i*xi3(MH), {Wf}')
plt.xscale('log')
plt.yscale('symlog')
plt.show()

# sys.exit()
def intfdeM(M,MHsmall,varsmall,xi3 ):
    xi3=g0*xi3
    mu = (M/(C*MHsmall))
    Integrand_f=-2/(OmegaCDM)/(np.sqrt(np.pi*2*varsmall))*np.exp(-(mu**(1./gamma)+deltac)**2/(2*varsmall))*M/MHsmall*(1./gamma)*(M/(C*MHsmall))**(1./gamma)*np.sqrt(Meq/MHsmall)
    # f= Intarray_vec(Integrand1, LMH) # ojo: tengo abs() aca!
    # Integrand_f=Integrand_f*-0.5*keq*np.sqrt(Meq/MHsmall) # jacobian k->M_H?
    # deltaM=(mu**(1./gamma)+deltac)**2
    deltaM = mu**(1./gamma)+deltac
#   deltaM=deltac
    # Integrand_ngcont=(deltaM**3/varsmall**3-3*deltaM/varsmall**2)
    Integrand_ngcont = (1./6.)*xi3*( (deltaM/varsmall)**3. - 3.*deltaM/varsmall**2)
    Integrand_ftot = Integrand_f*(1+(1./6.)*xi3*( (deltaM/varsmall)**3. - 3.*deltaM/varsmall**2) )
    # f2=  Intarray_vec(Integrand2, LMH)  # ojo: tengo abs() aca!
    # fng= Intarray_vec(Integrand4, LMH)  # ojo: tengo abs() aca!
    return Integrand_f, Integrand_ngcont, Integrand_ftot #, deltaM

'''
deltaM y la diferencia dentro de la parte ng nunca se hace negativas!
'''

ti= time.time()
print('f(M) calc')
initial_time_str = time.strftime('%H:%M:%S', time.localtime(ti))
print('Initial time:', initial_time_str)

deltaM=[]
f=np.zeros(size)
fng=np.zeros(size)
f2=np.zeros(size)
f_ngcont=[]
'''
this for may be optimizable.
eye with the fact that the length of Mz is way bigger than the length of xi3
'''
for i in tqdm.tqdm(range(0, len(Mz))):
    a,b,c = intfdeM(Mz[i],MHsmall,varsmall,xi3)
    f[i] = Intarray_vec( a, LMH)
    f_ngcont.append(b)
    # f2[i] = Intarray_vec( b, LMH)
    fng[i] = Intarray_vec( c, LMH)
    #deltaM.append(d)

for i in range(10):
    # plt.loglog(MHsmall, abs(f_ngcont[10*i]))
    # plt.show()
    plt.plot(MHsmall,f_ngcont[300*i], 'o', label=f'M={Mz[300*i]}')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.title(f'f_ngcont(MH) para Mpbh fijo, {Wf}')
    plt.xlabel(r'$M_{H}$')
    plt.legend()
    plt.show()
# for i in range(30):
#     plt.loglog(MHsmall,deltaM[100*i])
#     plt.show()
# plt.loglog(MHsmall,deltaM[2999])
# plt.show()
 


tf = time.time()
duration = tf - ti
t_f_ng=duration
final_time_str = time.strftime('%H:%M:%S', time.localtime(tf))

print(f"Computation of f(M) completed in {duration:.2f} seconds")


plt.plot(Mz,f,'o',label='f')
plt.plot(Mz,fng,'o',label='f_ng')
plt.legend()
plt.title(f'no abs f(M), {Wf} Smoothing function')
plt.yscale('symlog')
plt.xscale('log')
plt.show()

f_noabs=f
fng_noabs=fng

'''listo'''

f=abs(f)
fng=abs(fng)

LM=np.log(Mz)
f_pbh= Intarray_vec(f,LM)
fng_pbh= Intarray_vec(fng,LM)


fpeak=np.amax(abs(f))
mpeak=np.argmin(np.abs(abs(f)-fpeak)) 

Mp=Mz[int(mpeak)]


fngpeak=np.amax(fng)
mngpeak=np.argmin(np.abs(fng-fngpeak))
Mpng=Mz[int(mngpeak)]

# Find the maximum value of fng before and after mpeak
fng_before_mpeak = np.amax(fng[:mpeak])
fng_after_mpeak = np.amax(fng[mpeak:])

# Find the corresponding indices of these maximum values
mngpeak1 = np.where(fng == fng_before_mpeak)[0][-1]  # Last occurrence before mpeak
mngpeak2 = np.where(fng == fng_after_mpeak)[0][0]   # First occurrence after mpeak

# Retrieve the corresponding values of Mz at these indices
Mpng1 = Mz[mngpeak1]
Mpng2 = Mz[mngpeak2]

# np.savez(xi3_file, xi3=xi3,f=f, f2=f2, fng=fng, t_xi3_MH=t_xi3_MH, Mp=Mp, Mpng1=Mpng1, Mpng2=Mpng2)
# np.savez(xi3_file, xi3=xi3,f=f, f2=f2, fng=fng, Mp=Mp, Mpng1=Mpng1, Mpng2=Mpng2, f_pbh=f_pbh, fng_pbh=fng_pbh, fpeak=fpeak, fngpeak=fngpeak)

plt.loglog(Mz,f, 'o',label='f')
plt.loglog(Mz,fng, 'o',label='f_ng')
plt.axvline(x=Mp)
plt.axvline(x=Mpng, color='orange')
plt.title(f'f(M), {Wf} Smoothing function')
plt.legend()
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f(M)$')
plt.ylim(1e-9,5e13)
plt.show()


plt.loglog(Mz,f/f_pbh,'o',label='f')
plt.loglog(Mz,fng/fng_pbh,'o',label='f_ng')
plt.legend()
plt.axvline(x=Mp)
plt.axvline(x=Mpng, color='orange')
plt.title(f'f(M)/f_pbh, {Wf} Smoothing function')
plt.ylim(1e-14,5e0)
# plt.axhline(y=1, color='r', linestyle='--')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f(M)/f_{\rm PBH}$')
plt.show()



plt.loglog(Mz,f/f_pbh,label='f')
plt.loglog(Mz,fng/fng_pbh,label='f_ng')
plt.legend()
# plt.axvline(x=Mp)
# plt.axvline(x=Mpng1, color='orange')
# plt.axvline(x=Mpng2, color='orange')
plt.title(f'f(M)/f_pbh, {Wf} Smoothing function')
plt.ylim(1e-14,5e0)
plt.axhline(y=1, color='r', linestyle='--')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f(M)/f_{\rm PBH}$')
plt.show()



plt.loglog(Mz,f/fpeak,'o',label='f')
plt.loglog(Mz,fng/fngpeak ,'o',label='f_ng')
plt.legend()
plt.axvline(x=Mp)
plt.axvline(x=Mpng, color='orange')
plt.title(f'f(M)/f_peak, {Wf} Smoothing function')
plt.ylim(1e-14,5e0)
# plt.axhline(y=1, color='r', linestyle='--')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f(M)/f_{\rm peak}$')
plt.show()



plt.loglog(Mz,f/fpeak,label='f')
plt.loglog(Mz,fng/fngpeak ,label='f_ng')
plt.legend()
# plt.axvline(x=Mp)
# plt.axvline(x=Mpng1, color='orange')
# plt.axvline(x=Mpng2, color='orange')
plt.title(f'f(M)/f_peak, {Wf} Smoothing function')
plt.ylim(1e-14,5e0)
plt.axhline(y=1, color='r', linestyle='--')
plt.xlabel(r'$M_{\rm PBH}$')
plt.ylabel(r'$f(M)/f_{\rm peak}$')
plt.show()


frac=1.
betamono=2*frac*varsmall/deltac/np.sqrt(2*np.pi*varsmall)*np.exp(-deltac**2/(2*varsmall))
fmono=1./OmegaCDM*np.sqrt(Meq/MHsmall)*betamono

betang1= betamono*(1.+xi3/6/varsmall**2 *deltac*(deltac**2/varsmall**2-1))
betang2= betamono*(1.+xi3/6/varsmall**2 *deltac*(deltac**2/varsmall**2))

betaresum1 = betamono*np.exp(xi3/6/varsmall**2 *deltac*(deltac**2/varsmall**2-1))
betaresum2 = betamono*np.exp(xi3/6/varsmall**2 *deltac*(deltac**2/varsmall**2))

fmonoNG1=1./OmegaCDM*np.sqrt(Meq/MHsmall)*betang1
fmonoNG2=1./OmegaCDM*np.sqrt(Meq/MHsmall)*betang2

fmonoR1=1./OmegaCDM*np.sqrt(Meq/MHsmall)*betaresum1
fmonoR2=1./OmegaCDM*np.sqrt(Meq/MHsmall)*betaresum2

plt.plot(MHsmall,fmono,'o',label='f_mono')
plt.plot(MHsmall,fmonoNG1,'o',label='f_monoNG1')
plt.plot(MHsmall,fmonoNG2,'o',label='f_monoNG2')
plt.plot(MHsmall,fmonoR1,'o',label='f_monoR1')
plt.plot(MHsmall,fmonoR2,'o',label='f_monoR2')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()