import numpy as np
from attenuate import attenuate

def ct_detect(p, coeffs, thickness, mas=10000):

	"""ct_detect returns detector photons for given material thicknesses.
	y = ct_detect(p, coeffs, thickness, mas) takes a source energy
	distribution photons (energies), a set of material linear attenuation
	coefficients coeffs (materials, energies), and a set of material thicknesses
	in thickness (materials, samples) and returns the detections at each sample
	in y (samples).

	mas defines the current-time-product which affects the noise distribution
	for the linear attenuation"""

	# check p for number of energies
	if type(p) != np.ndarray:
		p = np.array([p])
	if p.ndim > 1:
		raise ValueError('input p has more than one dimension')
	energies = len(p)

	# check coeffs is of (materials, energies)
	if type(coeffs) != np.ndarray:
		coeffs = np.array([coeffs]).reshape((1, 1))
	elif coeffs.ndim == 1:
		coeffs = coeffs.reshape((1, len(coeffs)))
	elif coeffs.ndim != 2:
		raise ValueError('input coeffs has more than two dimensions')
	if coeffs.shape[1] != energies:
		raise ValueError('input coeffs has different number of energies to input p')
	materials = coeffs.shape[0]

	# check thickness is of (materials, samples)
	if type(thickness) != np.ndarray:
		thickness = np.array([thickness]).reshape((1,1))
	elif thickness.ndim == 1:
		if materials == 1:
			thickness = thickness.reshape(1, len(thickness))
		else:
			thickness = thickness.reshape(len(thickness), 1)
	elif thickness.ndim != 2:
		raise ValueError('input thickness has more than two dimensions')
	if thickness.shape[0] != materials:
		raise ValueError('input thickness has different number of materials to input coeffs')
	samples = thickness.shape[1]

	# extend source photon array so it covers all samples
	detector_photons = np.zeros([energies, samples])
	for e in range(energies):
		detector_photons[e] = p[e]

	# calculate array of residual mev x samples for each material in turn
	for m in range(materials):
		detector_photons = attenuate(detector_photons, coeffs[m], thickness[m])

	# sum this over energies
	detector_photons = np.sum(detector_photons, axis=0)

	# model noise

	# minimum detection is one photon
	detector_photons = np.clip(detector_photons, 1, None)

	return detector_photons