
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
L=251.327
k00 = 1e11 # Mpc^-1 . '''this gives k_peak=1.287e13 Mpc^-1'''

# initial and final k that will be integrated
ki=10 
kf=14
kikf=str(ki)+str(kf)
ki=1*10**ki 
kf=1*10**kf

Wf='Wthtf'
nkk=100 #number of steps
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

# choices for different configurations

# equilateral
# x=np.array([0.5])

# squeezed
# x=np.array([0.95,0.99,0.999,0.9999])
# theta =[18.19, 8.11, 2.56, 0.81]

# folded
# x=-1*np.array([0.95,0.99,0.999,0.9999])
# theta =[161.8, 171.8, 177.44, 179.19]
# x=-x[::-1]

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
databs_file = f'databs-{nkk}-steps-{spacing}-spacing-{kikf}-lambda-{L}.npy'

# Construct the full path including the directory
databs_file = os.path.join(data_directory, databs_file)

xi3_file=f'xi3-{Wf}-{nkk}-steps-{kikf}-{spacing}-spacing.npy'
xi3_file = os.path.join(data_directory, xi3_file)

databs=np.load(databs_file)


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

Mz = np.geomspace(MHsmall[-1], 10**(-1)*MHsmall[0], size) 

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
if Wf=='Wg':
    def W(k,q):    
        return np.exp(-(k/q)**2.)   
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

#
#not vectorized
#
# def integrandxi3(Mh,k1,k2,x):
#     a=np.zeros_like(databs)
#     k1=k1/k00
#     k2=k2/k00
#     q=kofMH(Mh)/k00
#     m1=Mcal(k1,q)
#     m2=Mcal(k2,q)
#     for i in range(len(k1)):
#         for j in range(len(k2)):
#             for l in range(len(x)):
#                 # if np.sqrt(k1[i]**2 + k2[j]**2 - 2*k1[i]*k2[j]*x[l]) < L*k00 and k1[i]<L*k00 and k2[j]<L*k00:
#                 if np.sqrt(k1[i]**2 + k2[j]**2 - 2*k1[i]*k2[j]*x[l]) < L and k1[i]<L and k2[j]<L:
#                     wx=np.zeros(len(x))
#                     wx[l]=4./9.*q**(-2)*W(np.sqrt(k1[i]**2 + k2[j]**2 - 2*k1[i]*k2[j]*x[l]), q)
#                     a[i,j,l] = k00**2*0.5*m1[i]*m2[j]*wx[l]*databs[i,j,l]
#     return a





# vectorized
def int_xi3(m1,m2,wx):
    a = k00**2*0.5/12.*m1[:, None, None]*m2[None, :, None]*wx*databs
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




LMH=np.log(MHsmall)
ti= time.time()
xi3 = np.zeros(len(MHsmall))

print('xi3 calc')
initial_time_str = time.strftime('%H:%M:%S', time.localtime(ti))
print('Initial time:', initial_time_str)
'''
this for is optimizable
'''
for i in tqdm.tqdm(range(len(MHsmall))):
    xi3[i] = Intarray3D_vec(integrandxi3(MHsmall[i], k1, k2, x), k1, k2, x)

# def xi3calc():
#     for i in tqdm.tqdm(range(len(MHsmall))):
#         xi3[i] = Intarray3D_vec(integrandxi3(MHsmall[i], k1, k2, x), [k1, k2, x])

tf = time.time()
duration = tf - ti
t_xi3_MH=duration

# Convert initial time to hh:mm:ss format
final_time_str = time.strftime('%H:%M:%S', time.localtime(tf))

print(f"Computation of xi3 completed in {duration:.2f} seconds")







plt.loglog(kzsmall,abs(xi3),'o')
plt.plot(kzsmall,abs(xi3))
plt.title(f'abs( xi3(k) ), {Wf}')
plt.show()

plt.figure(00)
plt.plot(kzsmall,xi3)
plt.plot(kzsmall,xi3,'o')
plt.title(f'xi3(k), {Wf}')
plt.xscale('log')
plt.yscale('symlog')
plt.show()

plt.figure(00)
plt.plot(MHsmall,xi3)
plt.plot(MHsmall,xi3,'o')
plt.title(f'xi3(M), {Wf}')
plt.xscale('log')
plt.yscale('symlog')
plt.show()

# np.save(xi3_file, xi3ofk)






'''listo'''


mu = np.zeros([len(MHsmall), size])
Integrand2=np.zeros([nkk,size])
Integrand3=np.zeros([nkk,size])
Integrand4=np.zeros([nkk,size])
# Integrand2d=np.zeros([nkk,nqq])
# Integrand2c=np.zeros([nkk,nqq])
f=np.zeros(size)
fng=np.zeros(size)
fng2=np.zeros(size)
# fc=np.zeros(nqq)
# fd=np.zeros(nqq)
f2=np.zeros(size)
# f3=np.zeros(nqq)
kzr=[]
sigmaR2r=[]


# to apply the perturbatibity condition I look for the maximum of abs(xi3) and its index.
# then, i look for the value of the variance at that index.
# afterwards, i find the maximun value for g that satisfies the perturbativity condition.
xi3max=max(abs(xi3))
xi3maxindex=np.argmax(abs(xi3))
# kmax=kk[xi3maxindex]
varmax=varsmall[xi3maxindex]
S3max=xi3max/varmax**2
g=12.*6.*varmax/(0.45**3.) /S3max
xi3norm=xi3*g

g=1e20
xi3norm=xi3*g



#este no importa tanto vectorizarlo
# for j in range(0, size):
#   mu[:,j]=(Mz[j]/(C*MHsmall))
#   Integrand2[:,j]=-2/(OmegaCDM)/(np.sqrt(np.pi*2*varsmall[:]))*np.exp(-(mu[:,j]**(1./gamma)+deltac)**2/(2*varsmall[:]))*Mz[j]/MHsmall[:]*(1./gamma)*(Mz[j]/(C*MHsmall[:]))**(1./gamma)*np.sqrt(Meq/MHsmall[:])
#   f[j]= abs(Intarray_vec(Integrand2[:,j], LMH[:]) ) # ojo: tengo abs() aca!
#   deltaM=(mu[:,j]**(1./gamma)+deltac)**2
# #   deltaM=deltac
#   Integrand3[:,j]=Integrand2[:,j]*(1./6.)*xi3[:]*(deltaM**3/varsmall[:]**3-3*deltaM/varsmall[:]**2)
#   Integrand4[:,j]=Integrand2[:,j]*(1+(1./6.)*xi3[:]*(deltaM**3/varsmall[:]**3-3*deltaM/varsmall[:]**2) )
#   # Integrand3[:,j]=Integrand2[:,j]*(1./6.)*xi3norm[:]*(deltaM**3/varsmall[:]**3-3*deltaM/varsmall[:]**2)
#   f2[j]= abs( Intarray_vec(Integrand3[:,j], LMH[:]) ) # ojo: tengo abs() aca!
# #   Integrand4[:,j]=Integrand2[:,j]*np.exp((1./6.)*xi3[:]*(deltaM**3/varsmall[:]**3-3*deltaM/varsmall[:]**2))
# #   fng2[j]=Intarray(Integrand4[:,j],LMH[:])
# #   fng[j]=f[j]+f2[j]
#   fng[j]= abs( Intarray_vec(Integrand4[:,j], LMH[:]) ) # ojo: tengo abs() aca!



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
'''
this for may be optimizable.
eye with the fact that the length of Mz is way bigger than the length of xi3
'''
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
final_time_str = time.strftime('%H:%M:%S', time.localtime(tf))

print(f"Computation of f(M) completed in {duration:.2f} seconds")


# np.savez( xi3_file, xi3=xi3,f=f, f2=f2, fng=fng, t_xi3_MH=t_xi3_MH)

'''listo'''

f=abs(f)
fng=abs(fng)

LM=np.log(Mz)
f_pbh= Intarray_vec(f,LM)
f_pbh_ng= Intarray_vec(fng,LM)


fpeak=np.amax(abs(f))
mpeak=np.argmin(np.abs(abs(f)-fpeak)) 

Mp=Mz[int(mpeak)]


fngpeak=np.amax(fng)
mngpeak=np.argmin(np.abs(fng-fngpeak))
Mpng=Mz[int(mngpeak)]


plt.loglog(Mz,f, 'o',label='f')
plt.loglog(Mz,fng, 'o',label='f_ng')
plt.axvline(x=Mp)
plt.axvline(x=Mpng, color='orange')
plt.legend()
plt.show()

plt.loglog(Mz,f/f_pbh, 'o',label='f')
plt.loglog(Mz,fng/f_pbh_ng, 'o',label='f_ng')
plt.legend()
plt.axvline(x=Mp)
plt.axvline(x=Mpng, color='orange')
plt.show()