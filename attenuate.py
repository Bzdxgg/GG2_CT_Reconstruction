import numpy as np
import math

def attenuate(original_energy, coeff, thickness):
	"""calculates residual photons for a particular material and thicknesses
	attenuate(original_energy, coeff, thickness, mas) takes the original_energy
	(energy, samples) and works out the residual_energy (energy, samples)
	for a particular material with linear attenuation coefficients given
	by coeff (energies), and a set of thicknesses given by thickness (samples)

	It is more efficient to calculate this for a range of samples rather then
	one at a time
	"""

	# check original energy is energy x samples
	if type(original_energy) != np.ndarray:
		original_energy = np.array([original_energy]).reshape((1, 1))
	elif original_energy.ndim == 1:
		original_energy = original_energy.reshape((len(original_energy), 1))
	elif original_energy.ndim != 2:
		raise ValueError('input original_energy has more than two dimensions')
	energies = original_energy.shape[0]
	samples = original_energy.shape[1]

	# check coeff is vector of energies
	if type(coeff) != np.ndarray:
		coeff = np.array([coeff])
	elif coeff.ndim != 1:
		raise ValueError('input coeffs has more than one dimension')
	if len(coeff) != energies:
		raise ValueError('input coeff has different number of energies to input original_energy')

	# check thickness is vector of samples
	if type(thickness) != np.ndarray:
		thickness = np.array([thickness])
	elif thickness.ndim != 1:
		raise ValueError('input thickness has more than one dimension')
	if len(thickness) != samples:
		raise ValueError('input thickness has different number of samples to input original_energy')

	# Work out residual energy for each thickness and at each energy
	residual_energy = original_energy * np.exp(-np.outer(coeff, thickness))


	return residual_energy
