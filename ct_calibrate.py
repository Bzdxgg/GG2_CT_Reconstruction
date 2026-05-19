import numpy as np
import scipy
from scipy import interpolate
from ct_detect import ct_detect

def ct_calibrate(photons, material, sinogram, scale):

	""" ct_calibrate convert CT detections to linearised attenuation
	sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
	in x (angles x samples) and returns a linear attenuation sinogram
	(angles x samples). photons is the source energy distribution, material is the
	material structure containing names, linear attenuation coefficients and
	energies in mev, and scale is the size of each pixel in x, in cm."""

	# Get dimensions and work out detection for just air of twice the side
	# length (has to be the same as in ct_scan.py)
	n = sinogram.shape[1]

	thickness_air =  np.full(n, 2 * n * scale)

	# detector counts for air (I0)
	I0 = ct_detect(photons, material.coeff('Air'), thickness_air)
	
	# linearise the sinogram using the air counts
	p = -np.log(np.maximum(sinogram / I0, 1e-10))

	return p