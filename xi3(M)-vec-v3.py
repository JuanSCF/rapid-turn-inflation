
import os, sys, time
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
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

# L = 213.628
# k0 = 1.57e7 # this is for Mpeak~1e-5 M_\odot

# L = 163.363
# k00 = 2.05e4 # this is for Mpeak~1e1 M_\odot

# initial and final k that will be integrated
ki = 10
kf = 13


kikf=str(ki)+str(kf)
ki = 3*10**ki 
kf = 3*10**kf
print(f'ki: {ki:.0e}, kf: {kf:.0e}')


Wf='Wg4' # Wg, Wg4, Wth, Wthtf 


nkk=350 #number of steps
size=3000


kk = np.geomspace(ki, kf, nkk, dtype='float64')
# kk = np.linspace(ki, kf, nkk,dtype='float64', endpoint=False)
k1 = kk
k2 = kk
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
databs_file = f'databs-gth-{nkk}-steps-{kikf}-lambda-{n}L0.npy'

# Construct the full path including the directory
databs_file = os.path.join(data_directory, databs_file)
# databs = np.load(f'C:\ZZZ\Laburos\Codes\\newdata\databs-gth-{nkk}-steps-{kikf}-lambda-{n}L0.npy')
databs = np.load(f'C:\ZZZ\Laburos\Codes\\newdata\datadbs-gth-{nkk}-steps-3e{kikf}-lambda-{n}L0.npy')

# File name to save xi3 data
# xi3_file=f'xi3-gth-{Wf}-{nkk}-steps-{kikf}-lambda-{n}L0'
xi3_file=f'xi3d-gth-{Wf}-{nkk}-steps-3e{kikf}-lambda-{n}L0'

xi3_file = os.path.join(data_directory, xi3_file)

# databs=np.load(databs_file)



############################################################################
############################# initialitiazion   ############################
############################################################################

Omegam = 0.315 #???
Meq=(2.8)*10**17. #solar masses
keq=0.01*(Omegam/0.31) #Mpc^-1
keq = 0.01
# Mi=(keq/ki)**2. *Meq
# Mf=(keq/kf)**2. *Meq

def kofMH(M):
    return keq*(Meq/M)**0.5
def MHofk(k):
    return (keq/k)**2.*Meq



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
    

# np.savez(cwd+'\\bs\\data\\gaussian-data-C'+str(C)+'-deltac'+str(deltac)+Wf+'.npz', kz=kz, sigmaR2=sigmaR2, f=f, fpeak=fpeak, OmegaPBH=OmegaPBH, Mp=Mp)

fgaussian_data_file = os.path.join(cwd, f'data\\gaussian-data-{Wf}.npz')
fgaussian_data = np.load(fgaussian_data_file)

kz=fgaussian_data['kz']
sigmaR2=fgaussian_data['sigmaR2']
MH = MHofk(kz)

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
Lkz = np.log(kzsmall)
LMH = np.log(MHsmall)

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
########################################################################

# the definition of Mcal \equiv \mathcal{M} is between eqs. 6.15 and 6.16
def Mcal(k,q):
    # q=q[:, None, None]
    m=4./9. *(k/q)**2.  *W(k,q)
    return m

# see eq 6.20 for the def with the dbs and 6.19 for the bs with dimensions
def integrandxi3(Mh,k1,k2,x):
    q=kofMH(Mh)
    m1=Mcal(k1,q)
    m2=Mcal(k2,q)
    k12x = np.sqrt(k1[:, None, None]**2 + k2[None, :, None]**2 - 2*k1[:, None, None]*k2[None, :, None]*x[None, None, :])
    # wx=4./9.*q**(-2.)*W(k12x,q)
    m12x=Mcal(k12x ,q)

    # a = 2./(2.*np.pi)**4*k1[:, None, None]**2.*k2[None, :, None]**2.*m1[:, None, None]*m2[None, :, None]*m12x*databs
    a = 0.5*m1[:, None, None]*m2[None, :, None]*m12x*databs/k12x**2. # dimensionless bs
    return a
# is it alright to make the conversion MH->q here? Am i overlooking something?
# i believe it is ok, due to the definition of the functions Mcal and W(k,q)
# Besides im not integrating MH here, but k1, k2 and x. Check if im considering the jacobian when integrating MH!!!

LMH=np.log(MHsmall)
ti= time.time()

print('xi3 calc')
initial_time_str = time.strftime('%H:%M:%S', time.localtime(ti))
print('Initial time:', initial_time_str)
'''
this for is optimizable, but requires a lot of ram
may try to do something with dask
'''
# MHsmall = MHsmall[::-1]
xi3 = np.zeros(len(MHsmall))
for i in tqdm.tqdm(range(len(MHsmall))):
    xi3[i] = Intarray3D_vec(integrandxi3(MHsmall[i], k1, k2, x), k1, k2, x)


tf = time.time()
duration = tf - ti
# t_xi3_MH=duration

# Convert initial time to hh:mm:ss format
final_time_str = time.strftime('%H:%M:%S', time.localtime(tf))
print('Final time:', final_time_str)

print(f"Computation of xi3 completed in {duration:.2f} seconds")

np.save(xi3_file, xi3)

# xi3 = np.load(xi3_file)
plt.plot(MHsmall,xi3,'o')
plt.plot(MHsmall,xi3)
plt.title(f'xi3(MH), {Wf}')
plt.xscale('log')
plt.yscale('symlog')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel(r'$M_H/M_\odot$')
plt.show()

plt.plot(kofMH(MHsmall),xi3,'o')
plt.plot(kofMH(MHsmall),xi3)
plt.title(f'xi3($k_H$), {Wf}')
plt.xscale('log')
plt.yscale('symlog')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('k [Mpc$^{-1}$]')
plt.show()

# lets compute a g value for every xi3 value
# gvec = 6.*varsmall**3/(deltac**3.) /abs(xi3)

# maximum g value allowed according to the perturbation condition
# deltac = 1
gcrit = 6*varsmall[np.argmax(varsmall)]**3/abs(xi3[np.argmax(varsmall)])/(deltac**3)
print(f'gcrit = {gcrit:.1e}, wf = {Wf}, deltac = {deltac}')

# g0 = 6e8
g0 = gcrit
plt.figure(00)
skew = (xi3)/varsmall**1.5
plt.plot(MHsmall, g0*skew,'o')
plt.plot(MHsmall, g0*skew)
# plt.plot(kz(MHsmall), g0*skew,'o')
# plt.plot(kz(MHsmall), g0*skew)
# plt.plot(MHsmall, g0*xi3/varsmall**2, label='skew sigma**2' )
# plt.plot(MHsmall, varsmall**1.5, label='variance**1.5')
plt.title(f'Skewness, {Wf}')
plt.ylabel(r'$\xi_3$')
plt.xlabel(r'$M_H/M_\odot$')
# plt.xlabel('k [Mpc$^{-1}$]')

plt.axhline(y=0, color='r', linestyle='--')
plt.xscale('log')
plt.yscale('symlog')
# plt.legend()
plt.yticks(list(plt.yticks()[0]) + [0])
# plt.savefig(f'skewness-vs-k-{Wf}-1e10-1e15.pdf')
plt.show()
