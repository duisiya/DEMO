import numpy as np
import asciidata
import os
import matplotlib.pyplot as plt



    

spec_obs = asciidata.open('J_band_obs.dat')
wave_obs = spec_obs[0].tonumpy()
flux_obs = spec_obs[1].tonumpy()
sigma_obs = spec_obs[2].tonumpy()

spec_synth = asciidata.open('low_res_synth_norm_Jband.dat')
wave_synth = spec_synth[0].tonumpy()
flux_synth = spec_synth[1].tonumpy()

ivar = 1 / np.power(sigma_obs, 2)
L = (max(wave_synth) - min(wave_synth)) * 2

K = 10
hs = np.vstack([np.vstack([np.cos(2 * np.pi * k * wave_synth/L) for k in range(K)]), np.vstack([np.sin(2 * np.pi * k * wave_synth/L) for k in range(1,K)])]) #create base matrix

A = hs * flux_synth
ATA = np.dot(A, (ivar * A).T)
ATy = np.dot(A, ivar * flux_obs)
x = np.linalg.solve(ATA, ATy)
fit_best = np.dot(A.T, x)


func = np.dot(x, hs)

plt.plot(wave_obs, flux_obs, 'k-', label='data')
plt.plot(wave_synth, flux_synth, label='model')
plt.plot(wave_synth, fit_best, label='fitted model')
plt.plot(wave_synth, func, label='fitting function')
plt.xlabel('Wavelength [microns]')
plt.ylabel('Normalized flux')
plt.legend(loc='lower right')
plt.show()

