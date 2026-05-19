import math
import numpy as np
import numpy.matlib

def ramp_filter(sinogram, scale, alpha=0.001):
	""" Ram-Lak filter with raised-cosine for CT reconstruction

	fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
	using a Ram-Lak filter.

	fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
	cosine raised to the power given by alpha."""

	# get input dimensions
	angles = sinogram.shape[0]
	n = sinogram.shape[1]

	#Set up filter to be at least twice as long as input
	m = np.ceil(np.log(2*n-1) / np.log(2))
	m = int(2 ** m)

	# apply filter to all angles
	print('Ramp filtering')
      
	omega = 2 * np.pi * np.fft.fftfreq(m, d=scale)
	omega_max = np.pi / scale # Nyquist frequency

	filt = np.abs(omega) / (2 * np.pi)
	window = np.cos((omega * np.pi) / (2 * omega_max)) ** alpha
	f = filt * window

	f[0] = f[1] / 6.0

	sinogram_fft = np.fft.fft(sinogram, n=m, axis=1)
	filtered_fft = sinogram_fft * f
	filtered_sinogram = np.fft.ifft(filtered_fft, axis=1).real
    
	return filtered_sinogram[:, :n]
	