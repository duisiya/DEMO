#This script degrades spectral resolution of synthetic spectra and put them on the observed wavelength grid

from matplotlib import pyplot as plt
import numpy as np
import asciidata

from scipy import ndimage, interpolate

#read synthetic spectrum
spec_synth = asciidata.open('J_band_mod.dat')
wave_synth = spec_synth[0].tonumpy()
flux_synth = spec_synth[1].tonumpy()

#read observed spectrum
spec_obs = asciidata.open('J_band_obs.dat')
wave_obs = spec_obs[0].tonumpy()
flux_obs = spec_obs[1].tonumpy()

dwave = (max(wave_obs) - min(wave_obs)) / len(wave_obs) 
sigma_dwave = dwave / (2 * np.sqrt(2 * np.log(2)))


#Convolution of the flux with a Gaussian that has a variable sigma
#def gauss(x, mu, sig):
#    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / np.sqrt(2 * np.pi * np.power(sig, 2))
#
#gauss_matr = np. hstack([np.vstack([gauss(x, mu, sig) for x in wave_synth]) for mu, sig in zip(wave_synth, G_sigma)])
#print gauss_matr

flux_conv = ndimage.filters.gaussian_filter(flux_synth,sigma_dwave)

#interpolating on the observed wavelength grid (to get same sizes in the wavelength dimension)
func_interp = interpolate.splrep(wave_synth, flux_conv) #searching for interpolation function
flux_synth_smooth = interpolate.splev(wave_obs, func_interp) #interpolating on a new grid


R = 2000. #resolution of observed spectra in J band
FWHM = wave_obs / R
G_sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))

def gauss(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / np.sqrt(2 * np.pi * np.power(sig, 2))           
gauss_matr = np.vstack([np.hstack([gauss(x, mu, sig) for x in wave_obs]) for mu, sig in zip(wave_obs, G_sigma)])
flux_synth_conv = np.dot(flux_synth_smooth * dwave, gauss_matr)   

plt.plot(wave_synth, flux_synth, label='Original')    
plt.plot(wave_obs,flux_synth_smooth, label='Binned')
plt.plot(wave_obs, flux_synth_conv, label='Low resolution')
plt.xlabel('Microns')
plt.ylabel('erg/s/m^2/Micron')
plt.legend(loc='upper left')
plt.show()



