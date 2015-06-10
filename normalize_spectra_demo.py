import numpy as np
import asciidata
from matplotlib import pyplot as plt


spec_synth = asciidata.open('low_res_synth_spectrum_Jband.dat')
wave_synth = spec_synth[0].tonumpy()
flux_synth = spec_synth[1].tonumpy()



#Continuum normalization (assuming continuum at 1.226 microns as in Bonnefoy et al. 2010)
index = min(range(len(wave_synth)), key = lambda i: abs(wave_synth[i] - 1.226))

flux_synth_norm = flux_synth / flux_synth[index]
plt.plot(wave_synth, flux_synth_norm)
plt.xlabel('Wavelength [microns]')
plt.ylabel('Normalized flux')
plt.show()




#THIS OPTION IS TO BE DISCUSSED
#
#The continuum is presented as a combination of sin and cos functions.
#The matrix equation F = AX, where F is flux, A is coefficient matrix (values for sin and cos ar each wavelenght) and X is solutution matrix

#K = 2
#L = (max(wave_synth) - min(wave_synth)) *2
#A = np.vstack([np.vstack([np.cos(2*np.pi*k*wave_synth/L) for k in range(K)]),
#               np.vstack([np.sin(2*np.pi*k*wave_synth/L) for k in range(1,K)])]) #coefficient matrix A
#
#ATA = np.dot(A, A.T)
#ATy = np.dot(A, flux_synth)
#X = np.linalg.solve(ATA, ATy)#solution
#F = np.dot(X, A) #continuum



